"""Throwaway read-only probe #3 for S6.

(1) Download the rete / rete-connection-plugin / rete-area-plugin npm tarballs (read only, extracted in
    memory) and grep their shipped source for the socket-compatibility and connection-veto hooks (A13).
(2) Compare how ECMAScript (bun) and Python format the number forms that appear in params, to show why a
    real JCS (RFC 8785) implementation is required on the Python side (A25).
(3) Record the praxis production zone configuration evidence (A8): angular.json polyfills, zone.js usage.
Writes argv[1]. No installs, no builds.
"""

import io
import json
import re
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

out = []


def tarball(pkg):
    meta = json.load(urllib.request.urlopen(f"https://registry.npmjs.org/{pkg}/latest", timeout=30))
    ver = meta["version"]
    data = urllib.request.urlopen(f"https://registry.npmjs.org/{pkg}/-/{pkg}-{ver}.tgz", timeout=60).read()
    return ver, tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")


PATTERNS = {
    "rete": [r"isCompatibleWith", r"class Socket", r"canMakeConnection", r"connectioncreate\b"],
    "rete-connection-plugin": [r"canMakeConnection", r"connectionpick", r"connectiondrop", r"connectioncreate"],
    "rete-area-plugin": [r"nodepicked", r"translated"],
}
for pkg, pats in PATTERNS.items():
    try:
        ver, tf = tarball(pkg)
    except Exception as e:  # noqa: BLE001
        out.append(f"rete\t{pkg}\tERR\t{e!r}"[:200])
        continue
    for m in tf.getmembers():
        if not m.isfile() or not re.search(r"\.(d\.ts|ts)$", m.name):
            continue
        text = tf.extractfile(m).read().decode("utf8", "replace")
        for pat in pats:
            for i, line in enumerate(text.splitlines(), 1):
                if re.search(pat, line):
                    out.append(f"rete\t{pkg}@{ver}\t{m.name}:{i}\t{line.strip()[:140]}")
                    break

vectors = ["0.1", "1e-5", "1e-7", "1e21", "1e16", "100", "1.0", "123456789012345680000", "0.000001", "5e-324"]
js_code = "const v=" + json.dumps(vectors) + ";console.log(JSON.stringify(v.map(x=>JSON.stringify(Number(x)))))"
try:
    js = json.loads(subprocess.run(["bun", "-e", js_code], capture_output=True, text=True, check=True, timeout=60).stdout)
    for raw, j in zip(vectors, js):
        py = json.dumps(float(raw))
        out.append(f"numfmt\t{raw}\tjs={j}\tpy={py}\t{'SAME' if j == py else 'DIFF'}")
except Exception as e:  # noqa: BLE001
    out.append(f"numfmt\tERR\t{e!r}"[:200])

wc = Path.home() / "projects/praxis/praxis/web-client"
aj = json.loads((wc / "angular.json").read_text())
n_poly = len(re.findall(r'"polyfills"', (wc / "angular.json").read_text()))
out.append(f"praxis-zone\tangular.json polyfills occurrences={n_poly}")
hits = subprocess.run(["grep", "-rln", "-E", "zone.js|provideZone|provideZoneless", str(wc / "src"), "--include=*.ts"],
                      capture_output=True, text=True).stdout.split()
out.append(f"praxis-zone\tsrc files mentioning zone.js/provideZone*: {[h.replace(str(wc) + '/', '') for h in hits]}")
Path(sys.argv[1]).write_text("\n".join(out) + "\n")
print("\n".join(out))
