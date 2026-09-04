# Grounded LLM Router

A support chatbot that answers questions from a document set and refuses to guess when the answer isn't in there. It also picks between a cheap and a strong model depending on how hard the question looks, and logs cost and latency for every call.

I built this after a conversation about job hunting in the German ML market. The feedback was blunt: nobody cares if you can train a model, they want to know if you can ship something and keep it running. So the point here isn't model quality - it's the two things that actually break LLM apps once real users show up.

## Why this exists

1. A support bot doesn't know the answer, so it makes one up. Fine for a demo, a real liability once it's customer-facing.
2. Every query goes to the same expensive model whether it needs to or not.

This project tackles both with something small enough to actually read end to end.

## How it works

```
query
  |
  v
retrieve top-k chunks from Chroma, get a distance score
  |
  v
distance too high? -> refuse, log it, zero cost
  |
  v (confident enough)
long or borderline query? -> strong model
otherwise -> cheap model
  |
  v
generate answer, constrained to the retrieved context only
  |
  v
check the answer itself for "I don't know" language
(this catches cases retrieval missed - see results below)
  |
  v
log everything to sqlite
```

## The corpus

`docs/` has four markdown files for a made-up SaaS called NimbusStack - pricing, security, SLA, features. If you want to actually use this as a portfolio piece, swap these for a real company's public docs or your own papers. A fictional corpus is fine for testing the pipeline, less fine for a recruiter to verify.

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env   # add a free Gemini key from aistudio.google.com
python core.py          # ingest docs/ into Chroma
streamlit run app.py    # launch the UI
```

## Results

Ran the 18-question eval set (10 answerable, 8 deliberately not) against `gemini-3.1-flash-lite` and `gemini-3.5-flash`. Full output in `results/eval_results.csv`.

| Metric | Result |
|---|---|
| Grounded-answer rate | 100% (10/10) |
| Correct-refusal rate | 100% (8/8) |
| Overall accuracy | 100% (18/18) |
| Avg latency | ~6.4s/query |

Eighteen questions is a small set - I'm not going to pretend 100% here means the system never hallucinates. What actually surprised me was how it got there. A distance threshold on its own wasn't enough. Questions like "does this support blockchain wallets" or "quantum computing workflows" came back just as close to the docs as genuinely answerable ones, because they share words like "integrate" and "workflows" with the corpus even though the fact itself isn't in there anywhere. Distance alone can't tell "sounds related" apart from "actually answered" - it took a second check, scanning the generated answer for refusal language, to catch what the threshold missed.

## A note on getting this running

Getting a working model name out of the free tier was its own small project. Over one session I hit rate limits, a deprecated model silently aliasing to a paid-only one, 503s, and models that `list_models()` still listed but the API had already dropped. `list_models.py` is the tool I ended up writing to stop guessing - it checks what's actually reachable with your key before you burn an eval run finding out the hard way. If you're running this fresh, run that first.

## Deploying somewhere with a live link

1. Create a Space at huggingface.co/new-space, SDK = Streamlit
2. Push this repo's contents
3. Add `GEMINI_API_KEY` under Space Settings -> Repository secrets
4. You get a public URL - the thing that matters if you're showing this to someone in HR who isn't going to clone a repo
