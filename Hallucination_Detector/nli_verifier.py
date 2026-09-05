"""
NLI-based claim verifier.

Two modes:
  - verify(): feeds the whole source document as premise. Simple, but see
    results/README.md - this was badly unreliable on true claims (28%
    accuracy) because the underlying cross-encoder was trained on
    single-sentence premise/hypothesis pairs, not paragraphs.
  - verify_with_retrieval(): retrieves the most relevant sentence(s) from
    the doc first, and uses that as the premise instead. Roughly doubled
    accuracy on true claims (to 56%), still short of an LLM judge (100%).

Label order for cross-encoder/nli-deberta-v3-base is
[contradiction, entailment, neutral] per the model card - confirmed, not
assumed, since a wrong label mapping would produce exactly this kind of
lopsided accuracy pattern and was worth ruling out explicitly.
"""

import re

from sentence_transformers import CrossEncoder, SentenceTransformer, util

NLI_LABELS = ["CONTRADICTED", "SUPPORTED", "UNVERIFIABLE"]


class NLIVerifier:
    def __init__(self):
        self.nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self._doc_sentences = {}
        self._doc_embeddings = {}

    def verify(self, premise: str, claim: str) -> str:
        scores = self.nli_model.predict([(premise, claim)])[0]
        return NLI_LABELS[scores.argmax()]

    def split_sentences(self, doc_text: str) -> list[str]:
        """These policy docs are short and mostly one-fact-per-line, so a
        simple split works better here than a general-purpose sentence
        tokenizer would."""
        lines = [l.strip() for l in doc_text.split("\n") if l.strip()]
        lines = [l for l in lines if not l.startswith("#")]
        sentences = []
        for line in lines:
            parts = re.split(r'(?<=[.!?])\s+', line)
            sentences.extend([p.strip() for p in parts if p.strip()])
        return sentences

    def index_doc(self, doc_name: str, doc_text: str):
        sentences = self.split_sentences(doc_text)
        self._doc_sentences[doc_name] = sentences
        self._doc_embeddings[doc_name] = self.embedder.encode(sentences, convert_to_tensor=True)

    def get_relevant_premise(self, doc_name: str, claim: str, top_k: int = 2) -> str:
        claim_emb = self.embedder.encode(claim, convert_to_tensor=True)
        doc_embs = self._doc_embeddings[doc_name]
        sentences = self._doc_sentences[doc_name]
        scores = util.cos_sim(claim_emb, doc_embs)[0]
        top_indices = scores.argsort(descending=True)[:top_k]
        return " ".join(sentences[i] for i in top_indices)

    def verify_with_retrieval(self, doc_name: str, claim: str, top_k: int = 2) -> str:
        premise = self.get_relevant_premise(doc_name, claim, top_k=top_k)
        return self.verify(premise, claim)
