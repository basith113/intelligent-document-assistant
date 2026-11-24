"""Split extracted text into overlapping, Unicode-safe token windows."""

import re
from bisect import bisect_right
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import accumulate

import tiktoken

from app.rag.loader import Page

BOUNDARY = re.compile(rb"\n\n|[.!?][ \n]")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    page: int | None
    index: int


class Chunker:
    def __init__(self, size: int, overlap: int, embedding_model: str) -> None:
        if size < 4 or not 0 <= overlap < size:
            raise ValueError("Use at least four tokens per chunk and a smaller nonnegative overlap.")
        self.size = size
        self.overlap = overlap
        try:
            self.encoding = tiktoken.encoding_for_model(embedding_model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def split(self, document_id: str, pages: list[Page]) -> list[Chunk]:
        """Chunk one page at a time so each citation has an unambiguous page."""
        chunks = []
        for page in pages:
            for text in self._windows(page.text):
                index = len(chunks)
                chunks.append(Chunk(f"{document_id}:{index}", text, page.number, index))
        return chunks

    def _windows(self, text: str) -> Iterator[str]:
        data = text.encode("utf-8")
        tokens = self.encoding.encode_ordinary(text)
        offsets = [0, *accumulate(len(self.encoding.decode_single_token_bytes(t)) for t in tokens)]
        start = 0
        while start < len(tokens):
            end = min(start + self.size, len(tokens))
            if end < len(tokens):
                soft_start = min(end, start + max(self.overlap + 1, self.size * 4 // 5))
                boundaries = list(BOUNDARY.finditer(data, offsets[soft_start], offsets[end]))
                if boundaries:
                    preferred = bisect_right(offsets, boundaries[-1].end()) - 1
                    end = max(soft_start, preferred)
            while True:
                while end < len(tokens) and data[offsets[end]] & 0xC0 == 0x80:
                    end -= 1
                excerpt = data[offsets[start]:offsets[end]].decode("utf-8").strip()
                if len(self.encoding.encode_ordinary(excerpt)) <= self.size:
                    break
                end -= 1
            if excerpt:
                yield excerpt
            if end == len(tokens):
                break
            start = max(start + 1, end - self.overlap)
            while start < end and data[offsets[start]] & 0xC0 == 0x80:
                start += 1
