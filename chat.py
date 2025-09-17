import os
import sys

from dotenv import load_dotenv
import google.generativeai as genai


def main() -> None:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY is not set. Put it in a .env file or your env.")
        print("Example .env line: GEMINI_API_KEY=your_key_here")
        sys.exit(1)

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-1.5-flash")
    chat = model.start_chat(history=[])

    print("Gemini Chatbot. Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user_input.lower() in {"exit", "quit", ":q", "q"}:
            print("Bye!")
            break

        if not user_input:
            continue

        try:
            response = chat.send_message(user_input)
            # The SDK returns a response with candidates; text() extracts concatenated text
            print(f"Bot: {response.text}\n")
        except Exception as exc:
            print(f"Error from Gemini API: {exc}")


if __name__ == "__main__":
    main()
