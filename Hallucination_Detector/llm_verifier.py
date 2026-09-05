"""
LLM-as-judge claim verifier, using Gemini.

Free-tier model names and rate limits shift without much notice - see the
retry logic here, which parses the wait time the API itself suggests
rather than guessing a fixed backoff.
"""

import re
import time

import google.generativeai as genai

JUDGE_PROMPT = """You are checking a single factual claim against a source document.

Source document:
{source}

Claim: "{claim}"

Does the source document SUPPORT this claim, CONTRADICT it, or is it UNVERIFIABLE (not mentioned in the source at all)?
Answer with exactly one word: SUPPORTED, CONTRADICTED, or UNVERIFIABLE."""


def parse_retry_delay(error_str: str, default: float = 20.0) -> float:
    match = re.search(r"retry in (\d+\.?\d*)s", error_str)
    return float(match.group(1)) + 2 if match else default


def llm_verify(source: str, claim: str, model_name: str = "gemini-3.1-flash-lite",
                max_retries: int = 4) -> str:
    prompt = JUDGE_PROMPT.format(source=source, claim=claim)
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            text = response.text.strip().upper()
            for label in ["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"]:
                if label in text:
                    return label
            return "UNVERIFIABLE"
        except Exception as e:
            wait = parse_retry_delay(str(e))
            print(f"    retry {attempt + 1}/{max_retries}, waiting {wait:.0f}s")
            time.sleep(wait)
    return "FAILED"
