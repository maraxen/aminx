"""ProtonPottsMPNN family (v6, 30-token vocabulary).

Spec: ``.praxia/docs/specs/261007_protonpottsmpnn-support.md``. Importing this package registers
``ProtonPottsDriver`` under ``model_family="protonpottsmpnn"``; the host imports it on first use of that
family (``host/runner.py``).
"""

from aminx.families.protonpotts_mpnn.driver import ProtonPottsDriver

__all__ = ["ProtonPottsDriver"]
