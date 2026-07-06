"""TF-IDF retriever over the local knowledge base.

Indexes every ``*.md`` file under ``docs/knowledge_base/`` at runtime, one chunk
per markdown section (``##`` heading and its body, plus any pre-heading intro).
No filenames or content are hardcoded: whatever markdown exists when the
retriever is constructed is what gets indexed.

Retrieval is scikit-learn ``TfidfVectorizer`` + cosine similarity — deterministic
and dependency-light. ``retrieve(query, k)`` returns the top-k chunks as
``{source_file, section, text, score}`` dicts.
"""

from __future__ import annotations

import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Default KB location (kept independent of the diagnostics module).
KB_DIR = Path(__file__).resolve().parents[2] / "docs" / "knowledge_base"

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def chunk_markdown(path: Path) -> list[dict]:
    """Split one markdown file into section chunks.

    Each chunk is ``{source_file, section, text}`` where ``text`` includes the
    heading line so heading terms contribute to the TF-IDF vocabulary. Content
    before the first heading becomes an ``(intro)`` chunk.
    """
    source_file = path.name
    chunks: list[dict] = []
    section = "(intro)"
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            chunks.append(
                {"source_file": source_file, "section": section, "text": body}
            )

    for line in path.read_text(encoding="utf-8").splitlines():
        m = _HEADING_RE.match(line)
        if m:
            flush()
            section = m.group(2).strip()
            buf = [line]
        else:
            buf.append(line)
    flush()
    return chunks


class Retriever:
    """TF-IDF retriever over knowledge-base markdown chunks."""

    def __init__(self, kb_dir: Path | str = KB_DIR):
        self.kb_dir = Path(kb_dir)
        self.chunks: list[dict] = []
        for md in sorted(self.kb_dir.glob("*.md")):
            self.chunks.extend(chunk_markdown(md))

        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        if self.chunks:
            self._vectorizer = TfidfVectorizer(
                stop_words="english", ngram_range=(1, 2), min_df=1
            )
            self._matrix = self._vectorizer.fit_transform(
                c["text"] for c in self.chunks
            )

    def __len__(self) -> int:
        return len(self.chunks)

    def retrieve(self, query: str, k: int = 4) -> list[dict]:
        """Return the top-``k`` chunks for ``query``, most relevant first.

        Chunks with zero similarity are dropped, so a query with no lexical
        overlap yields an empty list (the assistant treats that as "nothing
        relevant retrieved" rather than inventing content).
        """
        if not self.chunks or self._vectorizer is None or not query.strip():
            return []
        q_vec = self._vectorizer.transform([query])
        scores = linear_kernel(q_vec, self._matrix).ravel()
        order = scores.argsort()[::-1][:k]
        results: list[dict] = []
        for i in order:
            score = float(scores[i])
            if score <= 0.0:
                continue
            c = self.chunks[i]
            results.append(
                {
                    "source_file": c["source_file"],
                    "section": c["section"],
                    "text": c["text"],
                    "score": round(score, 4),
                }
            )
        return results
