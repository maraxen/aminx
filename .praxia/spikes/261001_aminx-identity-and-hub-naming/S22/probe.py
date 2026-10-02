"""S22: does GitHub's Pages documentation say that an Actions-published site gets no CNAME file (so the custom
domain is a repository setting)? Fetches the public docs page and prints the surrounding sentences."""
import html
import re
import urllib.request

URL = "https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/managing-a-custom-domain-for-your-github-pages-site"
req = urllib.request.Request(URL, headers={"User-Agent": "spike-probe/1"})
with urllib.request.urlopen(req, timeout=30) as r:
    raw = r.read().decode("utf-8", "replace")
raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw)))
for phrase in ["no CNAME file is created", "custom GitHub Actions workflow", "Settings", "HTTPS"]:
    idx = text.find(phrase)
    print(f"phrase {phrase!r}: found={idx >= 0}")
    if idx >= 0 and phrase in ("no CNAME file is created", "custom GitHub Actions workflow"):
        print("   context:", text[max(0, idx - 260): idx + 200])
