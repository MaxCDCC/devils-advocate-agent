"""Validate data/corpus_*.json, then (re)build the search indexes."""
import json
import sys
from pathlib import Path

import rag

REQUIRED = ["id", "jurisdiction", "lang", "code", "number", "title", "text"]
PLACEHOLDERS = ("PASTE", "REPLACE")


def validate(corpus_dir: str = "data") -> list[str]:
    problems, seen = [], set()
    files = sorted(Path(corpus_dir).glob("corpus_*.json"))
    if not files:
        return [f"No data/corpus_*.json found in {corpus_dir}/"]
    for p in files:
        for i, e in enumerate(json.loads(p.read_text(encoding="utf-8"))):
            where = f"{p.name}[{i}] ({e.get('id', '?')})"
            for f in REQUIRED:
                if not str(e.get(f, "")).strip():
                    problems.append(f"{where}: missing '{f}'")
            blob = f"{e.get('title', '')} {e.get('text', '')}"
            if any(m in blob for m in PLACEHOLDERS):
                problems.append(f"{where}: still contains placeholder text")
            if e.get("lang") not in ("en", "fr"):
                problems.append(f"{where}: lang must be 'en' or 'fr'")
            if e.get("id") in seen:
                problems.append(f"{where}: duplicate id")
            seen.add(e.get("id"))
    return problems


if __name__ == "__main__":
    problems = validate()
    if problems:
        print("Fix these before indexing:")
        print("\n".join(f" - {p}" for p in problems))
        sys.exit(1)
    print(f"{rag.build()} articles indexed.")