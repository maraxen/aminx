"""Importing aminx must leave Equinox's Module.__hash__ alone."""

from __future__ import annotations

import subprocess
import sys


def test_import_does_not_replace_equinox_module_hash() -> None:
    code = (
        "import equinox as eqx\n"
        "before = eqx.Module.__hash__\n"
        "import aminx\n"
        "import aminx.sampling.conditional_logits\n"
        "after = eqx.Module.__hash__\n"
        "assert after is before\n"
        "assert not (after.__module__ or '').startswith('aminx')\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)  # noqa: S603
    assert out.stdout.strip() == "ok"
