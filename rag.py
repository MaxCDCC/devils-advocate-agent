import json
import re
import sqlite3
from pathlib import Path
import hashlib
import pickle
import numpy as np

DB = Path("data/law.db")
VEC = Path("data/vectors.npy")
VEC_META = Path("data/vectors_meta.json")
CACHE = Path("data/embed_cache.pkl")
EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_model = None


def embed(texts: list[str], batch: int = 256) -> np.ndarray:
    """Texts -> unit-length vectors (so dot product == cosine similarity)."""
    global _model
    if _model is None:
        from fastembed import TextEmbedding  # downloads the model on first use
        _model = TextEmbedding(EMB_MODEL, cache_dir=".cache/fastembed")
    out = []
    for i in range(0, len(texts), batch):
        out.append(np.array(list(_model.embed(texts[i:i + batch])), dtype=np.float32))
        if len(texts) > batch:
            print(f"  embedded {min(i + batch, len(texts))}/{len(texts)}", flush=True)
    v = np.concatenate(out)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def embed_cached(texts: list[str]) -> np.ndarray:
    """Like embed(), but remembers vectors on disk: only NEW texts are embedded."""
    cache = pickle.loads(CACHE.read_bytes()) if CACHE.exists() else {}
    keys = [hashlib.sha1((EMB_MODEL + t).encode()).hexdigest() for t in texts]
    todo = sorted({i for i, k in enumerate(keys) if k not in cache})
    if todo:
        print(f"Embedding {len(todo)} new chunks ({len(texts) - len(todo)} cached)...")
        for i, v in zip(todo, embed([texts[i] for i in todo])):
            cache[keys[i]] = v
        CACHE.write_bytes(pickle.dumps(cache))
    return np.stack([cache[k] for k in keys])

def chunk(text: str, max_words: int = 80) -> list[str]:
    """Split long articles on sentence boundaries; short ones stay whole."""
    sentences = re.split(r"(?<=[.;:])\s+", text)
    chunks, cur, n = [], [], 0
    for s in sentences:
        w = len(s.split())
        if cur and n + w > max_words:
            chunks.append(" ".join(cur))
            cur, n = [], 0
        cur.append(s)
        n += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def build(corpus_dir: str = "data") -> int:
    """(Re)build the keyword index AND the vector index from data/corpus_*.json."""
    entries = []
    for p in sorted(Path(corpus_dir).glob("corpus_*.json")):
        entries += json.loads(p.read_text(encoding="utf-8"))
    DB.unlink(missing_ok=True)
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE articles(
        id TEXT PRIMARY KEY, jurisdiction TEXT, lang TEXT, code TEXT,
        number TEXT, title TEXT, text TEXT, url TEXT)""")
    con.execute("""CREATE VIRTUAL TABLE fts USING fts5(
        id UNINDEXED, title, text, keywords,
        tokenize='unicode61 remove_diacritics 2')""")
    chunk_texts, chunk_ids = [], []
    for e in entries:
        con.execute("INSERT INTO articles VALUES (?,?,?,?,?,?,?,?)",
                    (e["id"], e["jurisdiction"], e["lang"], e["code"],
                     str(e["number"]), e["title"], e["text"], e.get("url", "")))
        con.execute("INSERT INTO fts VALUES (?,?,?,?)",
                    (e["id"], e["title"], e["text"], e.get("keywords", "")))
        for c in chunk(e["text"]):
            chunk_texts.append(f'{e["title"]}. {c}')
            chunk_ids.append(e["id"])
    con.commit()
    con.close()
    np.save(VEC, embed_cached(chunk_texts))
    VEC_META.write_text(json.dumps(chunk_ids))
    return len(entries)


def _fts_query(q: str) -> str:
    words = [w for w in re.findall(r"\w+", q) if len(w) > 2]
    return " OR ".join(f'"{w}"' for w in words)


def _lang_of(ids: list[str]) -> dict:
    con = sqlite3.connect(DB)
    marks = ",".join("?" * len(ids))
    rows = con.execute(f"SELECT id, lang FROM articles WHERE id IN ({marks})", ids).fetchall()
    con.close()
    return dict(rows)


def _bm25_ids(query: str, lang: str | None, n: int) -> list[str]:
    fq = _fts_query(query)
    if not fq:
        return []
    sql = """SELECT a.id FROM fts JOIN articles a ON a.id = fts.id
             WHERE fts MATCH ? {f} ORDER BY bm25(fts, 0.0, 3.0, 1.0, 2.0) LIMIT ?"""
    params: list = [fq]
    f = ""
    if lang:
        f = "AND a.lang = ?"
        params.append(lang)
    params.append(n)
    con = sqlite3.connect(DB)
    ids = [r[0] for r in con.execute(sql.format(f=f), params)]
    con.close()
    return ids


def _vector_ids(query: str, lang: str | None, n: int) -> list[str]:
    V = np.load(VEC)
    meta = json.loads(VEC_META.read_text())
    scores = V @ embed([query])[0]
    best: dict[str, float] = {}
    for i, s in enumerate(scores):  # an article scores as its best chunk
        if s > best.get(meta[i], -1.0):
            best[meta[i]] = float(s)
    ranked = sorted(best, key=best.get, reverse=True)
    if lang:
        langs = _lang_of(ranked)
        ranked = [a for a in ranked if langs[a] == lang]
    return ranked[:n]


def _rrf(rankings: list[list[str]], c: int = 60) -> list[str]:
    """Reciprocal Rank Fusion: merge ranked lists without comparing raw scores."""
    score: dict[str, float] = {}
    for r in rankings:
        for pos, aid in enumerate(r):
            score[aid] = score.get(aid, 0.0) + 1.0 / (c + pos + 1)
    return sorted(score, key=score.get, reverse=True)


def search(query: str, lang: str | None = None, k: int = 5, mode: str = "hybrid") -> list[dict]:
    pool = max(k * 4, 20)
    if mode == "bm25":
        ids = _bm25_ids(query, lang, pool)
    elif mode == "vector":
        ids = _vector_ids(query, lang, pool)
    else:
        ids = _rrf([_bm25_ids(query, lang, pool), _vector_ids(query, lang, pool)])
    ids = ids[:k]
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    out = []
    for aid in ids:
        r = con.execute("""SELECT code, number, title, lang, jurisdiction,
                           substr(text, 1, 200) AS preview FROM articles WHERE id=?""", (aid,)).fetchone()
        out.append(dict(r))
    con.close()
    return out


def get_article(code: str, number: str) -> dict | None:
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    row = con.execute(
        "SELECT * FROM articles WHERE lower(code)=lower(?) AND number=?",
        (code.strip(), str(number).strip())).fetchone()
    con.close()
    return dict(row) if row else None