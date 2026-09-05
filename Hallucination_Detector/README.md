# Claim-Level Hallucination Detector

Splits an LLM's answer into individual factual claims and checks each one against its source document, two different ways: a small NLI model that runs locally for free, and an LLM acting as a judge. Built to answer one specific question - can the free option replace the paid one - and it turned out the answer is no, but not for the reason you'd guess.

Third in a set of portfolio projects (alongside a grounded RAG chatbot and a VLM fine-tuning pipeline), built after a conversation about what the Berlin ML job market actually screens for: shipped, checkable work rather than research depth.

## What's here

- `policy_docs/` - a small internal-policy corpus (leave, expenses, remote work, IT equipment) used as the source of truth
- `eval_claims.json` - 28 claims with ground truth: 18 true, 6 deliberately wrong, 4 not covered by the docs at all
- `nli_verifier.py` - the NLI-based checker, both the whole-document version and the sentence-level retrieval version
- `llm_verifier.py` - the LLM-as-judge checker, with retry handling for free-tier rate limits
- `run_eval.py` - runs the full comparison and writes results to `results/`
- `results/` - the actual numbers, with an honest account of what the data does and doesn't cover (see `results/README.md`)

## The findings

The NLI model caught 100% of wrong and unverifiable claims. On claims that were actually true, it only got it right 28% of the time - it kept flagging correct statements as unsupported. That turned out to be a premise-length problem: this class of NLI model is trained on single-sentence pairs, not paragraphs, and the whole policy document as premise was well outside what it had ever seen. Retrieving just the one relevant sentence instead roughly doubled accuracy on true claims, to 56%, but still fell well short of the LLM judge's 100%.

Full breakdown, including what's not in this repo and why, is in `results/README.md`.

## Setup

```bash
pip install -r requirements.txt
python run_eval.py
```

Needs a free Gemini API key (prompted at runtime, or set `GEMINI_API_KEY`). The NLI model (`cross-encoder/nli-deberta-v3-base`) and the sentence embedder (`all-MiniLM-L6-v2`) both download automatically on first run and need no API key.

## Why this matters beyond the numbers

A cheap, local model that's perfect at catching problems but unreliable at confirming correct answers is still a real, useful piece of infrastructure - it's a strong first-pass filter that only needs the expensive LLM call for the cases it can't confidently clear on its own. That hybrid pipeline is sketched out but not built in this repo; it's the natural next step if this gets picked back up.
