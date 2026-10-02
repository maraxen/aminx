"""S19: facts under the release guard (S3-12): what release.yml checks today, how the version is defined,
what PyPI's JSON API answers for a published and an unpublished version, and PyPI's file-name reuse rule.
"""
import html
import re
import urllib.error
import urllib.request
from pathlib import Path

wf = Path(".github/workflows/release.yml").read_text()
for needle in ["pypi.org/pypi", "tag_name", "github.ref", "skip-existing", "workflow_dispatch", "environment:"]:
    print(f"release.yml occurrences of {needle!r}: {wf.count(needle)}")

pp = Path("pyproject.toml").read_text()
print("pyproject [project].version line:", re.findall(r'(?m)^version = "[^"]+"$', pp)[:1])
print("pyproject declares dynamic version:", bool(re.search(r"(?m)^dynamic\s*=.*version", pp)))


def status(url):
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception as e:  # noqa: BLE001
        return f"ERR {type(e).__name__}"


print("pypi aminx/0.2.0a3 json status (published):", status("https://pypi.org/pypi/aminx/0.2.0a3/json"))
print("pypi aminx/0.2.0a4 json status (not published on 261001):", status("https://pypi.org/pypi/aminx/0.2.0a4/json"))

try:
    with urllib.request.urlopen("https://pypi.org/help/", timeout=30) as r:
        t = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", r.read().decode("utf-8", "replace"))))
    for phrase in ["does not allow for a filename to be reused", "previously used"]:
        idx = t.find(phrase)
        print(f"pypi help phrase {phrase!r}: found={idx >= 0} | {t[max(0, idx - 100): idx + 200] if idx >= 0 else ''}")
except Exception as e:  # noqa: BLE001
    print("pypi help fetch ERR", type(e).__name__, e)
