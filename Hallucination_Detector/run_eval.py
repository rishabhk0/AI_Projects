"""
Runs the eval set through all three verification modes and reports
per-category accuracy: NLI (whole-doc premise), NLI (sentence-level
retrieval), and LLM judge.

Usage:
    python run_eval.py
"""

import json
import os
import time
from getpass import getpass
from pathlib import Path

import pandas as pd
import google.generativeai as genai

from nli_verifier import NLIVerifier
from llm_verifier import llm_verify

DOCS_DIR = Path("policy_docs")
SECONDS_BETWEEN_LLM_CALLS = 5


def main():
    api_key = os.environ.get("GEMINI_API_KEY") or getpass("Gemini API key: ")
    genai.configure(api_key=api_key)

    eval_claims = json.loads(Path("eval_claims.json").read_text())
    verifier = NLIVerifier()

    doc_texts = {}
    for doc_name in {c["source_doc"] for c in eval_claims}:
        text = (DOCS_DIR / doc_name).read_text()
        doc_texts[doc_name] = text
        verifier.index_doc(doc_name, text)

    whole_doc_rows = []
    for i, item in enumerate(eval_claims):
        print(f"[{i + 1}/{len(eval_claims)}] {item['claim'][:60]}")
        source_text = doc_texts[item["source_doc"]]

        nli_pred = verifier.verify(source_text, item["claim"])
        llm_pred = llm_verify(source_text, item["claim"])
        time.sleep(SECONDS_BETWEEN_LLM_CALLS)

        whole_doc_rows.append({
            "claim": item["claim"],
            "source_doc": item["source_doc"],
            "ground_truth": item["label"],
            "nli_pred": nli_pred,
            "llm_pred": llm_pred,
            "nli_correct": nli_pred == item["label"],
            "llm_correct": llm_pred == item["label"],
            "methods_agree": nli_pred == llm_pred,
        })

    df = pd.DataFrame(whole_doc_rows)
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/whole_doc_results.csv", index=False)

    print(f"\nNLI accuracy (whole-doc):  {df.nli_correct.mean() * 100:.1f}%")
    print(f"LLM-judge accuracy:        {df.llm_correct.mean() * 100:.1f}%")

    sentence_rows = []
    for i, item in enumerate(eval_claims):
        nli_pred_v2 = verifier.verify_with_retrieval(item["source_doc"], item["claim"])
        sentence_rows.append({
            "claim": item["claim"],
            "ground_truth": item["label"],
            "nli_pred_sentence_level": nli_pred_v2,
            "correct_sentence_level": nli_pred_v2 == item["label"],
        })

    df_v2 = pd.DataFrame(sentence_rows)
    df_v2.to_csv("results/sentence_level_results.csv", index=False)

    print(f"\nNLI accuracy (sentence-level retrieval): {df_v2.correct_sentence_level.mean() * 100:.1f}%")
    for label in ["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"]:
        subset_v1 = df[df.ground_truth == label]
        subset_v2 = df_v2[df_v2.ground_truth == label]
        print(f"\n{label} (n={len(subset_v1)}):")
        print(f"  whole-doc:      {subset_v1.nli_correct.mean() * 100:.1f}%")
        print(f"  sentence-level: {subset_v2.correct_sentence_level.mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
