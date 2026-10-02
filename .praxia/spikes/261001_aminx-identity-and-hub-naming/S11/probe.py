"""S11: public-name availability for the hub candidates, and the state of the aminx release line on PyPI.

Read-only HTTP GET/HEAD against pypi.org and github.com. A transport failure is printed as ERR and
makes the probe inconclusive (re-run under a new id), never a pass.
"""
import json
import re
import urllib.error
import urllib.request

PRE = re.compile(r"(a|b|rc)\d+|\.dev\d+")


def pypi(name):
    url = f"https://pypi.org/pypi/{name}/json"
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception as e:  # noqa: BLE001 - probe, report and continue
        return f"ERR {type(e).__name__}", None


def gh(path):
    req = urllib.request.Request(f"https://github.com/{path}", method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception as e:  # noqa: BLE001
        return f"ERR {type(e).__name__}"


status, d = pypi("aminx")
if d:
    versions = sorted(d["releases"])
    print(f"pypi aminx status={status} releases={len(versions)} all_prerelease={all(PRE.search(v) for v in versions)} latest_info_version={d['info']['version']} versions={versions}")
else:
    print(f"pypi aminx status={status}")

for name in ["aminx-hub", "praxia-science", "praxia-hub", "praxia"]:
    status, d = pypi(name)
    extra = ""
    if d:
        info = d["info"]
        extra = f" latest={info['version']} author={info.get('author')!r} summary={info.get('summary')!r}"
    print(f"pypi {name} status={status}{extra}")

for path in ["maraxen/aminx", "maraxen/praxia", "maraxen/aminx-hub", "maraxen/praxia-science", "maraxen/praxia.science"]:
    print(f"github {path} status={gh(path)}")
