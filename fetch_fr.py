"""Download French codes (one Hugging Face dataset per code) -> data/corpus_fr_<code>.json"""
import json
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

CODES = [  # louisbrulenaudet/<slug>
    "code-penal",
    "code-civil",
    "code-route",
    "code-commerce",
    # "code-general-collectivites-territoriales",  # slug NOT verified: try it
]


def to_entries(df: pd.DataFrame, slug: str) -> list[dict]:
    if "etat" in df.columns:
        print("  etat values:", df["etat"].value_counts().to_dict())
        df = df[df["etat"].fillna("").str.startswith("VIGUEUR")]  # in-force only
    df = df[df["texte"].fillna("").str.strip() != ""]
    if "cid" in df.columns and "dateDebut" in df.columns:  # one version per article
        df = df.sort_values("dateDebut").drop_duplicates("cid", keep="last")
    entries = []
    for r in df.itertuples():
        entries.append({
            "id": f"fr-{slug}-{r.id}",
            "jurisdiction": "FR", "lang": "fr",
            "code": r.title_main, "number": str(r.num),
            "title": r.ref, "text": " ".join(str(r.texte).split()),
            "url": f"https://www.legifrance.gouv.fr/codes/article_lc/{r.id}",
            "keywords": "",
        })
    return entries


def load_code(slug: str) -> pd.DataFrame:
    repo = f"louisbrulenaudet/{slug}"
    files = [f for f in list_repo_files(repo, repo_type="dataset")
             if f.startswith("data/") and f.endswith(".parquet")]
    frames = [pd.read_parquet(hf_hub_download(repo, f, repo_type="dataset")) for f in files]
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    for slug in (sys.argv[1:] or CODES):
        try:
            df = load_code(slug)
            print(f"{slug}: columns = {list(df.columns)}")
            entries = to_entries(df, slug)
        except Exception as e:
            print(f"FAILED {slug}: {e}")
            continue
        out = Path(f"data/corpus_fr_{slug}.json")
        out.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
        print(f"  {len(entries)} articles -> {out.name}")