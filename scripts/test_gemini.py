from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai


def main() -> None:
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY is missing in .env")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Reply with exactly: Gemini connection ok",
    )
    text = (response.text or "").strip()
    print(text)


if __name__ == "__main__":
    main()

