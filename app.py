import os
from typing import List, Dict, Any, Optional, Tuple

from flask import Flask, jsonify, render_template, request, session
from dotenv import load_dotenv
import google.generativeai as genai
import requests
import math
import re


app = Flask(__name__)
# Use a random-ish default but recommend overriding via env in production
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-me")


load_dotenv()

OSM_CONTACT_EMAIL = os.environ.get("OSM_CONTACT_EMAIL", "contact@example.com")
OSM_USER_AGENT = f"MedAssistant/1.0 ({OSM_CONTACT_EMAIL})"


def get_model():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set. Put it in a .env file or your env.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")


# ----------------------------- OSM helpers ---------------------------------

def geocode_zip(zip_code: str) -> Optional[Tuple[float, float]]:
    """Return (lat, lon) for a US ZIP code using Nominatim."""
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": zip_code,
                "countrycodes": "us",
                "format": "json",
                "limit": 1,
                "addressdetails": 1,
            },
            headers={"User-Agent": OSM_USER_AGENT, "Accept": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        items = resp.json()
        if not items:
            return None
        lat = float(items[0]["lat"])  # type: ignore[index]
        lon = float(items[0]["lon"])  # type: ignore[index]
        return lat, lon
    except Exception:
        return None


def overpass_find_pharmacies(lat: float, lon: float, radius_m: int = 5000) -> List[Dict[str, Any]]:
    """Find pharmacies near coordinates using Overpass. Returns list of elements."""
    query = f"""
    [out:json][timeout:25];
    node["amenity"="pharmacy"](around:{radius_m},{lat},{lon});
    out body;
    """
    try:
        resp = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            headers={"User-Agent": OSM_USER_AGENT, "Accept": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("elements", [])
    except Exception:
        return []


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def format_pharmacy_list(lat: float, lon: float, elements: List[Dict[str, Any]], limit: int = 10) -> str:
    rows = []
    for el in elements:
        tags = el.get("tags", {})
        name = tags.get("name") or "Pharmacy"
        el_lat = el.get("lat")
        el_lon = el.get("lon")
        if el_lat is None or el_lon is None:
            continue
        distance_km = haversine_km(lat, lon, float(el_lat), float(el_lon))
        address_parts = [
            tags.get("addr:housenumber"),
            tags.get("addr:street"),
            tags.get("addr:city"),
            tags.get("addr:state"),
            tags.get("addr:postcode"),
        ]
        address = ", ".join([p for p in address_parts if p]) or "Address unavailable"
        map_link = f"https://www.openstreetmap.org/?mlat={el_lat}&mlon={el_lon}#map=18/{el_lat}/{el_lon}"
        rows.append((distance_km, f"{name} — {address} — {distance_km:.1f} km\n{map_link}"))

    rows.sort(key=lambda x: x[0])
    if not rows:
        return "No pharmacies found within 5 km. Try a different ZIP or larger radius."

    lines = [f"Here are pharmacies near you (sorted by distance):"]
    for _, line in rows[:limit]:
        lines.append(f"- {line}")
    return "\n".join(lines)


ZIP_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")

# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/chat")
def chat_api():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    # Flow: detect pharmacy intent or await ZIP, otherwise pass to LLM
    awaiting = session.get("awaiting_zip")
    lower = user_message.lower()

    if awaiting == "pharmacies":
        m = ZIP_RE.search(user_message)
        if not m:
            return jsonify({"reply": "Please enter a valid 5-digit ZIP code (e.g., 10001)."})
        zip_code = m.group(0)
        coords = geocode_zip(zip_code)
        if not coords:
            session.pop("awaiting_zip", None)
            return jsonify({"reply": "Sorry, I could not find that ZIP code. Please try another."})
        lat, lon = coords
        elements = overpass_find_pharmacies(lat, lon)
        reply_text = format_pharmacy_list(lat, lon, elements)
        session.pop("awaiting_zip", None)
        # Track history as normal
        history: List[Dict[str, Any]] = session.get("history", [])
        history.append({"role": "user", "parts": [{"text": user_message}]})
        history.append({"role": "model", "parts": [{"text": reply_text}]})
        session["history"] = history
        return jsonify({"reply": reply_text})

    if "pharmacy" in lower:
        session["awaiting_zip"] = "pharmacies"
        return jsonify({"reply": "Please enter your 5-digit ZIP code to find nearby pharmacies."})

    # Default: route to Gemini
    history: List[Dict[str, Any]] = session.get("history", [])
    model = get_model()
    chat = model.start_chat(history=history)

    try:
        response = chat.send_message(user_message)
        bot_text = response.text
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    history.append({"role": "user", "parts": [{"text": user_message}]})
    history.append({"role": "model", "parts": [{"text": bot_text}]})
    session["history"] = history

    return jsonify({"reply": bot_text})


@app.post("/api/reset")
def reset_chat():
    session.pop("history", None)
    session.pop("awaiting_zip", None)
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
