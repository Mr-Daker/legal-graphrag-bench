from __future__ import annotations

from gemini_client import DEFAULT_XAI_MODEL, generate_text


def main() -> None:
    result = generate_text(
        prompt="Reply with exactly: xAI connection ok",
        system_instruction="You are a concise test responder.",
        model=DEFAULT_XAI_MODEL,
        provider="xai",
    )
    print(result.answer)


if __name__ == "__main__":
    main()
