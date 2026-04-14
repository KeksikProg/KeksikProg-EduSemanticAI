import re
from typing import Any, Dict, List


BAD_TITLES = (
    "СПИСОК ЛИТЕРАТУРЫ",
    "ЛИТЕРАТУР",
    "ПРИЛОЖЕНИЕ",
)


BAD_PHRASES = [
    r"(?i)\s*,?\s*(на|см\.?|смотри)\s+рисунк(е|ах|и)\s*[\d,\-\–\sи]+",
]


def _normalize_line(line: str) -> str:
    for pattern in BAD_PHRASES:
        line = re.sub(pattern, " ", line)
    return re.sub(r"\s{2,}", " ", line).strip()


def clean_text(text: str) -> str:
    if not text:
        return ""

    lines = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.upper().startswith("РИСУНОК"):
            continue

        line = _normalize_line(line)
        lines.append(line)

    text = " ".join(lines)
    text = re.sub(r"\s+([,.:;])", r"\1", text)
    text = re.sub(r"([,.:;])(?=\S)", r"\1 ", text)

    return text


def skip_section(title: str) -> bool:
    title = title.upper()
    return any(bad in title for bad in BAD_TITLES)


def get_role(title: str, number: str | None) -> str:
    t = title.upper()

    if "ВВЕДЕНИЕ" in t:
        return "intro"
    if "ЗАКЛЮЧЕНИЕ" in t or "ВЫВОД" in t:
        return "conclusion"
    if number:
        return "main"

    return "other"


def _build_segment(
    *,
    seg_id: str,
    parent: Any,
    path: str,
    role: str,
    text: str,
) -> Dict[str, Any]:
    return {
        "id": seg_id.lower(),
        "parent": parent,
        "path": path,
        "role": role,
        "text": text,
    }


def flatten(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    result = []

    for section in doc.get("sections", []):
        title = section.get("title", "")
        number = section.get("number")

        if skip_section(title):
            continue

        role = get_role(title, number)
        subs = section.get("subsections") or []

        if subs:
            for sub in subs:
                text = clean_text(sub.get("text", ""))

                if not text:
                    continue

                result.append(
                    _build_segment(
                        seg_id=sub.get("number") or sub.get("title"),
                        parent=section.get("id"),
                        path=f"{title} -> {sub.get('title')}",
                        role=role,
                        text=text,
                    )
                )
        else:
            text = clean_text(section.get("text", ""))

            if not text:
                continue

            result.append(
                _build_segment(
                    seg_id=number or title,
                    parent=section.get("id"),
                    path=title,
                    role=role,
                    text=text,
                )
            )

    return result


def prepare_for_embeddings(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": doc.get("title"),
        "author": doc.get("author"),
        "type": doc.get("type"),
        "segments": flatten(doc),
    }
