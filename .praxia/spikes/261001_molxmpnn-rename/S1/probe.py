"""S1: PyPI name state for aminx and molxmpnn (read-only GETs, retried: the egress proxy is flaky)."""
import json
import re
import sys
import time
import urllib.error
import urllib.request

PRE = re.compile(r"(a|b|rc|dev)\d+")


def fetch(name: str):
    url = f"https://pypi.org/pypi/{name}/json"
    last = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            return e.code, None  # a definitive HTTP answer (404 = unregistered)
        except Exception as e:  # noqa: BLE001 - transport flake, retry
            last = repr(e)
            time.sleep(2 * (attempt + 1))
    return f"transport-failure:{last}", None


status_m, _ = fetch("molxmpnn")
print("molxmpnn_status", status_m)
status_a, data = fetch("aminx")
print("aminx_status", status_a)
if data:
    versions = sorted(data["releases"])
    print("aminx_versions", versions)
    print("aminx_all_prerelease", all(PRE.search(v) for v in versions))
    print("aminx_latest", data["info"]["version"])
sys.exit(0)
