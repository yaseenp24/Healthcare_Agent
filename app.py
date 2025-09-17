import os
from typing import List, Dict, Any

from flask import Flask, jsonify, render_template, request, session
from dotenv import load_dotenv
import google.generativeai as genai


app = Flask(__name__)
# Use a random-ish default but recommend overriding via env in production
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-me")


load_dotenv()

def get_model():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set. Put it in a .env file or your env.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/chat")
def chat_api():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    # Retrieve or initialize chat history from session
    history: List[Dict[str, Any]] = session.get("history", [])

    # Build chat session
    model = get_model()
    chat = model.start_chat(history=history)

    try:
        response = chat.send_message(user_message)
        bot_text = response.text
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    # Append to history and persist
    history.append({"role": "user", "parts": [{"text": user_message}]})
    history.append({"role": "model", "parts": [{"text": bot_text}]})
    session["history"] = history

    return jsonify({"reply": bot_text})


@app.post("/api/reset")
def reset_chat():
    session.pop("history", None)
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
