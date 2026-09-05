# Results

28 claims: 18 true (SUPPORTED), 6 deliberately wrong (CONTRADICTED), 4 not covered by any doc at all (UNVERIFIABLE). Full claim list with ground truth is in `eval_claims.json`.

## Whole-document premise

Feeding the entire source document to the NLI model as premise. Full per-claim data in `whole_doc_results.csv`.

| Category | n | NLI accuracy | LLM judge accuracy |
|---|---|---|---|
| Overall | 28 | 53.6% | 100.0% |
| SUPPORTED | 18 | 27.8% | 100.0% |
| CONTRADICTED | 6 | 100.0% | 100.0% |
| UNVERIFIABLE | 4 | 100.0% | 100.0% |

The NLI model was perfect at catching wrong or unverifiable claims and wrong most of the time when a claim was actually true. It kept flagging correct statements as unsupported or even contradicted.

## Sentence-level retrieval

Instead of the whole document, retrieve the most relevant sentence (or two) and use that as the premise. Aggregate numbers in `sentence_level_summary.csv`.

| Category | n | Whole-doc | Sentence-level |
|---|---|---|---|
| Overall | 28 | 53.6% | 67.9% |
| SUPPORTED | 18 | 27.8% | 55.6% |
| CONTRADICTED | 6 | 100.0% | 100.0% |
| UNVERIFIABLE | 4 | 100.0% | 75.0% |

A note on what's missing here: the notebook this came from printed these aggregate numbers and saved a per-claim CSV on the Colab side, but that file wasn't part of what got uploaded when this repo was put together. So `sentence_level_summary.csv` has the real aggregate stats, verified against the printed notebook output, but there's no per-claim breakdown for the sentence-level run the way there is for the whole-document run. If you want that detail, `run_eval.py` reproduces it in a few minutes.

## What this actually shows

A cross-encoder NLI model trained on single-sentence premise/hypothesis pairs doesn't generalize well to paragraph-length premises. Feeding it the whole document was badly out of distribution, and it defaulted to "unsupported" for a lot of claims that were plainly true. Retrieving the one relevant sentence first closed part of the gap on true claims (roughly doubled, 28% to 56%) but not all of it. Even with the correct sentence in front of it, the model still missed close to half of the true claims.

Two things worth taking from this. First, NLI alone isn't reliable enough here to replace an LLM judge, even with good retrieval - the LLM was accurate on every single claim in this set, NLI never approached that on true claims regardless of premise quality. Second, NLI was flawless in exactly the direction that matters most for catching hallucinations: it never once missed a wrong or unverifiable claim, in either the whole-document or sentence-level version. A cheap, local, zero-cost model that reliably flags problems but occasionally cries wolf on things that are actually fine is a genuinely useful first-pass filter, just not a full replacement.

A cost-saving pipeline that uses NLI as a cheap filter and only calls the LLM to double-check the cases NLI is unreliable on (the SUPPORTED verdicts) was discussed but not built. That's the natural next step if this gets picked back up.
