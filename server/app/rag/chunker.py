import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    index: int
    content: str
    content_hash: str
    char_count: int


def chunk_text(text: str, max_chars: int = 700, overlap: int = 90) -> list[TextChunk]:
    normalized = _normalize_text(text)
    if not normalized:
        return []
    if max_chars <= overlap:
        raise ValueError("max_chars must be greater than overlap")

    chunks: list[TextChunk] = []
    start = 0
    index = 0
    while start < len(normalized):
        end = min(start + max_chars, len(normalized))
        end = _adjust_end(normalized, start, end)
        content = normalized[start:end].strip()
        if content:
            chunks.append(
                TextChunk(
                    index=index,
                    content=content,
                    content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    char_count=len(content),
                )
            )
            index += 1
        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(line for line in lines if line)


def _adjust_end(text: str, start: int, end: int) -> int:
    if end >= len(text):
        return end
    window = text[start:end]
    for separator in ("\n\n", "\n", "。", "！", "？", ".", "!", "?"):
        offset = window.rfind(separator)
        if offset >= max(80, len(window) // 2):
            return start + offset + len(separator)
    return end
