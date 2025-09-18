import os
from typing import List, Dict, Any, Optional, Tuple

from flask import Flask, jsonify, render_template, request, session
from dotenv import load_dotenv
import google.generativeai as genai
import requests
import math
import re
from urllib.parse import urlparse


app = Flask(__name__)
# Use a random-ish default but recommend overriding via env in production
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-me")


load_dotenv()

OSM_CONTACT_EMAIL = os.environ.get("OSM_CONTACT_EMAIL", "contact@example.com")
OSM_USER_AGENT = f"MedAssistant/1.0 ({OSM_CONTACT_EMAIL})"


MAX_HISTORY_ITEMS = 8
MAX_HISTORY_CHARS = 200

# ------------------------- Google CSE configuration -------------------------
GOOGLE_CSE_API_KEY = os.environ.get("GOOGLE_CSE_API_KEY")
GOOGLE_CSE_CX = os.environ.get("GOOGLE_CSE_CX")
# Optional hard allowlist, comma-separated domains like: example.com, who.int
HEALTH_ALLOWED_DOMAINS = {
    d.strip().lower()
    for d in (os.environ.get("HEALTH_ALLOWED_DOMAINS", "").split(","))
    if d.strip()
}


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


def overpass_find_pharmacies(lat: float, lon: float, radius_m: int = 8000) -> List[Dict[str, Any]]:
    """Find pharmacies near coordinates using Overpass. Returns list of elements (node/way/relation)."""
    query = f"""
    [out:json][timeout:35];
    (
      node["amenity"="pharmacy"](around:{radius_m},{lat},{lon});
      way["amenity"="pharmacy"](around:{radius_m},{lat},{lon});
      relation["amenity"="pharmacy"](around:{radius_m},{lat},{lon});
    );
    out center;
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


def _element_coords(el: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    if "lat" in el and "lon" in el:
        return float(el["lat"]), float(el["lon"])  # node
    center = el.get("center")
    if isinstance(center, dict) and "lat" in center and "lon" in center:
        return float(center["lat"]), float(center["lon"])  # way/relation center
    return None


def format_pharmacy_list(lat: float, lon: float, elements: List[Dict[str, Any]], limit: int = 10) -> str:
    rows = []
    for el in elements:
        tags = el.get("tags", {})
        name = tags.get("name") or "Pharmacy"
        coords = _element_coords(el)
        if not coords:
            continue
        el_lat, el_lon = coords
        distance_km = haversine_km(lat, lon, el_lat, el_lon)
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
        return "No pharmacies found within 8 km. Try a different ZIP or larger radius."

    lines = [f"Here are pharmacies near you (sorted by distance):"]
    for _, line in rows[:limit]:
        lines.append(f"- {line}")
    return "\n".join(lines)


ZIP_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")
PHARM_INTENT_RE = re.compile(r"\b(pharmacy|pharmacies|drug ?store|chemist|chemists)\b", re.IGNORECASE)
COUNT_RE = re.compile(r"\b(\d{1,2})\b")
FOLLOWUP_WORDS_RE = re.compile(r"\b(closest|nearest|nearby|more)\b", re.IGNORECASE)

# Health intent detection - made more inclusive
HEALTH_INTENT_RE = re.compile(
    r"\b(ibuprofen|acetaminophen|aspirin|tylenol|advil|motrin|aleve|"
    r"symptom|symptoms|diagnos|treat|treatment|dose|dosage|side\s*effect|"
    r"contraindicat|interact|medication|medicine|drug|vaccine|vaccination|"
    r"condition|disease|infection|injury|pain|fever|cough|diarrhea|asthma|"
    r"hypertension|diabetes|cancer|migraine|allergy|dermatitis|rash|anxiety|depression)\b",
    re.IGNORECASE,
)


def extract_requested_count(text: str, default: int = 10) -> int:
    m = COUNT_RE.search(text)
    if not m:
        return default
    try:
        n = int(m.group(1))
    except Exception:
        return default
    return max(1, min(20, n))


def _truncate(text: str, max_len: int = MAX_HISTORY_CHARS) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "…"


def append_history(user_message: str, reply_text: str) -> None:
    # Keep a very small, truncated history to avoid cookie bloat
    history: List[Dict[str, Any]] = session.get("history", [])
    history.append({"role": "user", "parts": [{"text": _truncate(user_message)}]})
    history.append({"role": "model", "parts": [{"text": _truncate(reply_text)}]})
    if len(history) > MAX_HISTORY_ITEMS:
        history = history[-MAX_HISTORY_ITEMS:]
    session["history"] = history

# ------------------------------ Health helpers ------------------------------

def is_health_question(text: str) -> bool:
    """Detect health-related questions (excluding pharmacy locator intent)."""
    if PHARM_INTENT_RE.search(text):
        return False
    return bool(HEALTH_INTENT_RE.search(text))


def _is_allowed_domain(url: str) -> bool:
    if not url:
        return False
    try:
        hostname = urlparse(url).hostname or ""
        hostname = hostname.lower()
        if not HEALTH_ALLOWED_DOMAINS:
            # No explicit allowlist provided; rely on CSE restriction
            return True
        # Allow if hostname matches or is a subdomain of any allowed domain
        for allowed in HEALTH_ALLOWED_DOMAINS:
            if hostname == allowed or hostname.endswith("." + allowed):
                return True
        return False
    except Exception:
        return False


def cse_search(query: str, num: int = 3) -> list[dict[str, str]]:
    """Query Google CSE and return a list of {title, link, snippet}."""
    if not GOOGLE_CSE_API_KEY or not GOOGLE_CSE_CX:
        print("[DEBUG] CSE API key or CX not configured")
        return []
    try:
        print(f"[DEBUG] CSE searching for: {query}")
        resp = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "key": GOOGLE_CSE_API_KEY,
                "cx": GOOGLE_CSE_CX,
                "q": query,
                "num": max(1, min(10, num)),
                "safe": "active",
                "hl": "en",
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items") or []
        print(f"[DEBUG] CSE returned {len(items)} results")
        results: list[dict[str, str]] = []
        for it in items:
            title = (it.get("title") or "").strip()
            link = (it.get("link") or "").strip()
            snippet = (it.get("snippet") or "").strip()
            if not link or not title:
                continue
            if not _is_allowed_domain(link):
                print(f"[DEBUG] Blocked non-allowed domain: {link}")
                continue
            print(f"[DEBUG] Adding result: {title} - {link}")
            results.append({"title": title, "link": link, "snippet": snippet})
        return results
    except Exception as e:
        print(f"[DEBUG] CSE error: {e}")
        return []


def answer_health_with_cse(user_question: str) -> Optional[str]:
    """Use CSE results to answer the health question with citations, or None if insufficient."""
    results = cse_search(user_question, num=5)  # Get more results to increase chances
    if not results:
        return None

    # Build a constrained prompt using only titles/snippets/URLs
    lines: list[str] = [
        "You are a cautious health information assistant.",
        "You must use ONLY the sources provided below.",
        "If the sources are insufficient or off-topic, say you cannot answer.",
        "Cite sources as [1], [2], etc., and include the URL next to each citation.",
        "Avoid diagnosis or prescribing. Provide general, educational guidance only.",
        "Add this disclaimer at the end: This is educational information, not medical advice.",
        "",
        f"Question: {user_question}",
        "",
        "Sources:",
    ]

    for idx, r in enumerate(results, start=1):
        title = r["title"]
        url = r["link"]
        snippet = r.get("snippet") or ""
        lines.append(f"[{idx}] {title} — {url}")
        if snippet:
            lines.append(f"Snippet: {snippet}")
        lines.append("")

    instructions = [
        "Instructions:",
        "- Answer concisely in 4-8 sentences.",
        "- Use only information supported by the sources.",
        "- Cite claims with [number] and include the URL in parentheses.",
        "- If evidence is insufficient, say you cannot answer.",
        "- End with the disclaimer.",
    ]
    prompt = "\n".join(lines + instructions)

    model = get_model()
    chat = model.start_chat(history=[])
    try:
        resp = chat.send_message(prompt)
        bot_text = (resp.text or "").strip()
        if not bot_text:
            return None
        return bot_text
    except Exception:
        return None

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

    # Flow: always require a ZIP for pharmacy searches; do not reuse past coords
    awaiting = session.get("awaiting_zip")
    m_zip = ZIP_RE.search(user_message)

    has_pharm_intent = bool(PHARM_INTENT_RE.search(user_message))
    looks_like_pharmacy_request = has_pharm_intent or bool(FOLLOWUP_WORDS_RE.search(user_message))

    if awaiting == "pharmacies":
        if not m_zip:
            return jsonify({"reply": "Please enter a valid 5-digit ZIP code (e.g., 10001)."})
        zip_code = m_zip.group(0)
        coords = geocode_zip(zip_code)
        if not coords:
            session.pop("awaiting_zip", None)
            session.pop("pharmacy_limit", None)
            return jsonify({"reply": "Sorry, I could not find that ZIP code. Please try another."})
        lat, lon = coords
        elements = overpass_find_pharmacies(lat, lon)
        limit = int(session.pop("pharmacy_limit", 10))
        reply_text = format_pharmacy_list(lat, lon, elements, limit=limit)
        session.pop("awaiting_zip", None)
        append_history(user_message, reply_text)
        return jsonify({"reply": reply_text})

    # One-shot: "5 pharmacies 92620" or "pharmacies 92620"
    if looks_like_pharmacy_request and m_zip:
        zip_code = m_zip.group(0)
        coords = geocode_zip(zip_code)
        if not coords:
            return jsonify({"reply": "Sorry, I could not find that ZIP code. Please try another."})
        lat, lon = coords
        limit = extract_requested_count(user_message, default=10)
        elements = overpass_find_pharmacies(lat, lon)
        reply_text = format_pharmacy_list(lat, lon, elements, limit=limit)
        append_history(user_message, reply_text)
        return jsonify({"reply": reply_text})

    if looks_like_pharmacy_request:
        session["awaiting_zip"] = "pharmacies"
        session["pharmacy_limit"] = extract_requested_count(user_message, default=10)
        return jsonify({"reply": "Please enter your 5-digit ZIP code to find nearby pharmacies."})

    # Health Q&A via Google CSE (non-pharmacy)
    if is_health_question(user_message):
        print(f"[DEBUG] Health question detected: {user_message}")
        answer = answer_health_with_cse(user_message)
        if answer:
            append_history(user_message, answer)
            return jsonify({"reply": answer})
        else:
            print("[DEBUG] CSE returned no answer")
            decline = (
                "I can't answer that health question based on the approved sources. "
                "Please consult a qualified healthcare professional."
            )
            append_history(user_message, decline)
            return jsonify({"reply": decline})

    # Default: route to Gemini with a tiny context
    history: List[Dict[str, Any]] = session.get("history", [])
    # Use at most last 4 messages to keep prompts small
    short_history = history[-4:] if len(history) > 4 else history
    model = get_model()
    chat = model.start_chat(history=short_history)

    try:
        response = chat.send_message(user_message)
        bot_text = response.text
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    append_history(user_message, bot_text)
    return jsonify({"reply": bot_text})


@app.post("/api/reset")
def reset_chat():
    session.pop("history", None)
    session.pop("awaiting_zip", None)
    session.pop("pharmacy_limit", None)
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
