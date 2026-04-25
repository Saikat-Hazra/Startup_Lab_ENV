"""Gemini model interface with retry and safe error handling."""

import os
import time

from google import genai

class ModelInterface:
    """Wrapper around Gemini API."""

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        genai.Client(api_key=api_key)
        self.model_name = 'gemini-1.5-flash'

    def generate(self, prompt: str) -> str:
        """Generate text with up to 2 attempts."""
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                response = self.model.generate_content(prompt)
                text = getattr(response, "text", None)
                if text:
                    return text.strip()
                return "analyze_market"
            except Exception as exc:  # pragma: no cover - external API variability
                last_error = exc
                if attempt == 0:
                    time.sleep(0.5)
        # Never crash the app flow on model-side errors; let agent fallback handle it.
        return "analyze_market"
