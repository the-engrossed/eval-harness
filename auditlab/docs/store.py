"""Document access by item id. Opens known file names; never lists the docs folder."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from auditlab.docs.loader import Page, ParsedDocument, load_html_file

BANK_DOC_ID = "bank-statement"
ITEM_DOC_SUFFIXES = ("invoice", "bol", "hold-letter", "warehouse-note")
_SAFE_ID = re.compile(r"[A-Za-z0-9_-]+")


class UnknownItemError(KeyError):
    pass


class DocStore:
    def __init__(self, case_dir: Path) -> None:
        self.docs_dir = case_dir / "docs"
        self._cache: dict[str, ParsedDocument | None] = {}

    def _load(self, doc_id: str) -> ParsedDocument | None:
        if doc_id not in self._cache:
            path = self.docs_dir / f"{doc_id}.html"
            self._cache[doc_id] = load_html_file(path) if path.is_file() else None
        return self._cache[doc_id]

    def item_docs(self, item_id: str) -> dict[str, ParsedDocument]:
        """Every document that exists for a real item. Raises only if it has no invoice."""
        if not _SAFE_ID.fullmatch(item_id):
            raise UnknownItemError(item_id)
        docs: dict[str, ParsedDocument] = {}
        for suffix in ITEM_DOC_SUFFIXES:
            doc = self._load(f"{item_id}-{suffix}")
            if doc is not None:
                docs[f"{item_id}-{suffix}"] = doc
        if f"{item_id}-invoice" not in docs:
            raise UnknownItemError(item_id)
        return docs

    def bank_pages(self, months: list[tuple[int, int]]) -> dict[int, Page]:
        """Statement pages (by page number) for the given months; absent months are skipped."""
        statement = self._load(BANK_DOC_ID)
        if statement is None:
            return {}
        wanted = {f"Period: {year}-{month:02d}-01 to" for year, month in months}
        return {p.page_no: p for p in statement.pages if any(w in p.text for w in wanted)}

    def pages(self, doc_ids: Iterable[str]) -> dict[str, dict[int, str]]:
        """Page texts by doc id, then page number, for the docs that exist."""
        out: dict[str, dict[int, str]] = {}
        for doc_id in sorted(set(doc_ids)):
            doc = self._load(doc_id) if _SAFE_ID.fullmatch(doc_id) else None
            if doc is not None:
                out[doc_id] = {p.page_no: p.text for p in doc.pages}
        return out

    def page_text(self, doc_id: str, page_no: int) -> str | None:
        doc = self._load(doc_id) if _SAFE_ID.fullmatch(doc_id) else None
        if doc is None:
            return None
        return next((p.text for p in doc.pages if p.page_no == page_no), None)
