"""Download WHOLE UK Acts from legislation.gov.uk (CLML XML) -> data/corpus_uk_*.json"""
import json
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ACTS = [  # path on legislation.gov.uk. The script prints each Act's real title: CHECK IT.
    "ukpga/1980/66",      # Highways Act 1980
    "ukpga/1986/64",      # Public Order Act 1986
    "ukpga/1998/42",      # Human Rights Act 1998
    "ukpga/1968/60",      # Theft Act 1968
    "ukpga/1971/48",      # Criminal Damage Act 1971
    "ukpga/1988/52",      # Road Traffic Act 1988
    "ukpga/1978/30",      # Interpretation Act 1978
    "ukpga/2014/12",      # Anti-social Behaviour, Crime and Policing Act 2014
    "ukpga/1994/33",      # Criminal Justice and Public Order Act 1994
    "ukpga/Vict/24-25/100",  # Offences against the Person Act 1861
]
DC_TITLE = "{http://purl.org/dc/elements/1.1/}title"


def local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def parse_act(xml_bytes: bytes):
    """Return (act_title, [(section_number, heading, text), ...])."""
    root = ET.fromstring(xml_bytes)
    act_title = next((e.text.strip() for e in root.iter(DC_TITLE) if e.text), "?")
    sections = []

    def walk(el):
        name = local(el.tag)
        if name == "BlockAmendment":  # quoted text of OTHER laws, not provisions of this Act
            return
        if name == "P1group":
            heading = next((clean(" ".join(c.itertext())) for c in el if local(c.tag) == "Title"), "")
            for p1 in el:
                sid = p1.get("id", "") if local(p1.tag) == "P1" else ""
                if sid.startswith("section-"):
                    sections.append((sid[len("section-"):], heading, clean(" ".join(p1.itertext()))))
            return
        for c in el:
            walk(c)

    walk(root)
    return act_title, sections


def to_entries(path: str, act_title: str, sections) -> list[dict]:
    entries = []
    for num, heading, text in sections:
        if len(text) < 60 and "repealed" in text.lower():  # skip empty repealed shells
            continue
        entries.append({
            "id": f"uk-{path.replace('/', '-')}-{num}",
            "jurisdiction": "UK", "lang": "en",
            "code": act_title, "number": num,
            "title": heading or f"Section {num}",
            "text": text,
            "url": f"https://www.legislation.gov.uk/{path}/section/{num}",
            "keywords": "",
            "retrieved_at": time.strftime("%Y-%m-%d"),
        })
    return entries


def fetch(path: str) -> bytes:
    url = f"https://www.legislation.gov.uk/{path}/data.xml"
    req = urllib.request.Request(url, headers={"User-Agent": "devils-advocate-learning-project"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    for path in ACTS:
        try:
            act_title, sections = parse_act(fetch(path))
        except Exception as e:
            print(f"FAILED {path}: {e}")
            continue
        entries = to_entries(path, act_title, sections)
        out = Path(f"data/corpus_uk_{path.replace('/', '_')}.json")
        out.write_text(json.dumps(entries, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{act_title}: {len(entries)} sections -> {out.name}")
        time.sleep(1)  # be polite