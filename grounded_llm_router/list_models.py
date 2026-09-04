"""
Check which Gemini models this API key can actually reach right now.

Free-tier model availability shifts — names get deprecated, renamed, or
rate-limited without much notice. Run this before a real eval pass instead
of assuming CHEAP_MODEL / STRONG_MODEL in core.py are still valid.

Usage:
    python list_models.py
"""

import os
import time
from getpass import getpass

import google.generativeai as genai


def main():
    api_key = os.environ.get("GEMINI_API_KEY") or getpass("Gemini API key: ")
    genai.configure(api_key=api_key)

    print("Models registered for this key:\n")
    all_models = [
        m.name for m in genai.list_models()
        if "generateContent" in m.supported_generation_methods
    ]
    for name in all_models:
        print(f"  {name}")

    print("\nTesting reachability (one call per model, 3s apart)...\n")
    working = []
    for model_name in all_models:
        short_name = model_name.replace("models/", "")
        try:
            model = genai.GenerativeModel(short_name)
            response = model.generate_content("Say OK")
            print(f"  {short_name}: WORKS — {response.text.strip()}")
            working.append(short_name)
        except Exception as e:
            print(f"  {short_name}: FAILED ({str(e)[:80]})")
        time.sleep(3)

    print(f"\nCurrently working: {working}")


if __name__ == "__main__":
    main()
