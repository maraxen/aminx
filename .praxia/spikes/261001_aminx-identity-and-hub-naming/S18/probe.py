"""S18: (1) what GitHub's Pages docs say about custom domains and domain verification; (2) what DNS holds today
for praxia.science and hub.praxia.science (DNS-over-HTTPS read of public records; no change of any kind).
"""
import html
import json
import re
import urllib.error
import urllib.request

BASE = "https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/"
PAGES = {
    "managing": BASE + "managing-a-custom-domain-for-your-github-pages-site",
    "verifying": "https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/verifying-your-custom-domain-for-github-pages",
}


def text_of(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "spike-probe/1"})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return None, f"ERR {type(e).__name__}: {e}"
    raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw))), "ok"


for key, url in PAGES.items():
    t, st = text_of(url)
    print(f"docs[{key}] fetch={st} chars={len(t) if t else 0}")
    if t:
        for phrase in ["CNAME", "github.io", "apex", "takeover", "verif"]:
            idx = t.find(phrase)
            snippet = t[max(0, idx - 80): idx + 140] if idx >= 0 else ""
            print(f"  phrase {phrase!r}: found={idx >= 0} | {snippet}")


def doh(name, rtype):
    url = f"https://dns.google/resolve?name={name}&type={rtype}"
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            d = json.load(r)
    except Exception as e:  # noqa: BLE001
        return f"ERR {type(e).__name__}: {e}"
    answers = [a.get("data") for a in d.get("Answer", [])]
    return f"Status={d.get('Status')} answers={answers}"


for name in ["praxia.science", "hub.praxia.science", "aminx.praxia.science", "www.praxia.science"]:
    for rtype in (["A", "AAAA", "CNAME", "NS", "TXT"] if name == "praxia.science" else ["A", "CNAME"]):
        print(f"dns {name} {rtype}: {doh(name, rtype)}")
