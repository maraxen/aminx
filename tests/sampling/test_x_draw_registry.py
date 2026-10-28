"""Every residue draw in aminx is registered, and every registered draw either excludes X or says why not.

Debt #2660: the generic autoregressive sampler drew over all 21 columns, so it could emit the
unknown token X, which upstream LigandMPNN never draws. Fixing that one site does not stop the
next draw site from repeating it, so this test enumerates the draw sites by AST rather than by
name, and fails on any site it has not been told about.

Classification of each draw site (``REGISTRY`` below):

* ``x_excluded``: the innermost function that performs the draw calls one of the shared
  masks (``exclude_x_logits``, ``exclude_x_probs``, or refine's ``mask_refine_x``). The AST
  check below proves the mask is in the same function body.
* ``family_excluded``: a model family excludes X by its own mechanism (a valid-token mask,
  a disabled-residue floor). ``FAMILY_EVIDENCE`` names the function and the exact source
  text that does it. The test fails if that text disappears.
* ``known_gap``: the draw can still select X. The reason says why it is not fixed in the
  current change. The test asserts the site is still unmasked, so a fix flips the entry
  and cannot be forgotten.
* ``exempt``: the draw does not choose a residue (for example, it chooses ensemble members
  or a rotamer bin), or it is a shared draw primitive whose callers are registered.

Draw sites are found as calls (``random.categorical``/``choice``/``multinomial``,
``searchsorted``, a Gumbel argmax, ``gumbel_softmax``, ``categorical_draw``) and as
inverse-CDF assignments (``cdf = ...cumsum(...)``, where the target name says it is a CDF).
"""

from __future__ import annotations

import ast
import pathlib

import pytest

SRC_ROOT = pathlib.Path(__file__).resolve().parents[2] / "src" / "aminx"

# (module, attribute) pairs that are token draws when called as ``module.attribute(...)``.
_RANDOM_DRAWS = {"categorical", "choice", "multinomial"}
_SEARCHSORTED = {"searchsorted"}
#: ``mask_refine_x`` is PottsMPNN refine's own X mask (-1e8 on the X logit before the softmax).
_MASK_NAMES = {"exclude_x_logits", "exclude_x_probs", "mask_refine_x"}


def _call_name(node: ast.Call) -> str | None:
  func = node.func
  if isinstance(func, ast.Name):
    return func.id
  if isinstance(func, ast.Attribute):
    return func.attr
  return None


def _draw_kind(node: ast.Call) -> str | None:
  """Return the draw kind for a call node, or None when the call is not a residue draw."""
  name = _call_name(node)
  if name is None:
    return None
  func = node.func
  base = func.value if isinstance(func, ast.Attribute) else None
  base_name = base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", None)
  if name in _RANDOM_DRAWS and base_name == "random":
    return f"random.{name}"
  if name in _SEARCHSORTED:
    return "searchsorted"
  if name == "argmax" and any(
    "gumbel" in ast.unparse(arg) for arg in [*node.args, *(kw.value for kw in node.keywords)]
  ):
    return "gumbel-argmax"
  if name == "gumbel_softmax":
    return "gumbel_softmax"
  if name == "categorical_draw":
    return "inverse-cdf-call"
  return None


def _is_cdf_assign(node: ast.AST) -> bool:
  """True for ``cdf = <...>.cumsum(...)``: the CDF of an inverse-CDF draw.

  Other ``cumsum`` uses (offsets, running indices) are not draws, so the target name has to
  say it is a CDF.
  """
  if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
    return False
  if _call_name(node.value) != "cumsum":
    return False
  return any(
    isinstance(t, ast.Name) and ("cdf" in t.id.lower() or "cumul" in t.id.lower())
    for t in node.targets
  )


def find_draw_sites(root: pathlib.Path = SRC_ROOT) -> dict[str, dict[str, object]]:
  """Map ``relpath::qualname`` to the draw kinds and whether the enclosing function masks X.

  ``qualname`` is the dotted chain of enclosing ``def`` names, so two ``step`` functions in
  one module are distinct sites. A draw in a lambda is attributed to the enclosing ``def``.
  """
  sites: dict[str, dict[str, object]] = {}
  for path in sorted(root.rglob("*.py")):
    rel = path.relative_to(root).as_posix()
    tree = ast.parse(path.read_text(), filename=str(path))
    _collect(tree, rel, [], None, sites)
  return sites


def _collect(
  node: ast.AST,
  rel: str,
  names: list[str],
  innermost: ast.AST | None,
  sites: dict[str, dict[str, object]],
) -> None:
  for child in ast.iter_child_nodes(node):
    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
      _collect(child, rel, [*names, child.name], child, sites)
      continue
    kind = None
    if isinstance(child, ast.Call):
      kind = _draw_kind(child)
    elif _is_cdf_assign(child):
      kind = "inverse-cdf"
    if kind is not None:
      qual = ".".join(names) if names else "<module>"
      key = f"{rel}::{qual}"
      entry = sites.setdefault(key, {"kinds": set(), "masked": True, "lines": []})
      entry["kinds"].add(kind)
      entry["lines"].append(child.lineno)
      entry["masked"] = bool(entry["masked"]) and _masks_x(innermost)
    _collect(child, rel, names, innermost, sites)


def _masks_x(func: ast.AST | None) -> bool:
  if func is None:
    return False
  return any(isinstance(n, ast.Call) and _call_name(n) in _MASK_NAMES for n in ast.walk(func))


#: Every draw site in src/aminx, classified. Keys are ``relpath::qualname``.
REGISTRY: dict[str, tuple[str, str]] = {
  "ebm/structure_prediction.py::resample_ensemble": (
    "exempt",
    "chooses ensemble members, not residues",
  ),
  "inference/decode/autoregressive.py::__call__": (
    "exempt",
    "searchsorted over chromatic wave boundaries; selects a schedule index, not a residue",
  ),
  "utils/aa_convert.py::string_key_to_index": (
    "exempt",
    "searchsorted over sorted string keys; a lookup, not a draw",
  ),
  "types/bundles.py::from_colors": (
    "exempt",
    "maps tie-group IDs to contiguous group indices (0..G-1), not amino-acid token draws",
  ),
  "host/averaging.py::_make_encoding_sampling_split_fn_legacy.sample_fn.sample_group_step": (
    "x_excluded",
    "generic host sampler, grouped positions",
  ),
  "host/averaging.py::_make_encoding_sampling_split_fn_legacy.sample_fn.sample_step": (
    "x_excluded",
    "generic host sampler, decoding order",
  ),
  "inference/decode/autoregressive.py::_fuse_and_sample": (
    "x_excluded",
    "generic autoregressive decode (debt #2660, LigandMPNN hard-omits X)",
  ),
  "potts/sampling.py::gibbs_sweep.step": (
    "x_excluded",
    "Gibbs draw; matches PottsMPNN's energy-design convention (slice to 20 letters)",
  ),
  "potts/sampling.py::dlmc_sweep.one_sweep.step": (
    "x_excluded",
    "Zanella MH proposal: X weight zeroed in both forward and reverse proposals; the entry "
    "guard moves active X sites to alanine (burn-in).",
  ),
  "families/protonpotts_mpnn/ph_methods.py::pick_probs": (
    "known_gap",
    "ProtonPotts sampling path: inverse-CDF draw. The pH design path already zeroes X "
    "(ph_plan.valid_token_mask); this path mirrors upstream, which does not mask UNK when "
    "sampling. Left unchanged because it is graded against sealed upstream dumps. Filed as debt.",
  ),
  "inference/decode/ste.py::__call__.loss_fn": (
    "exempt",
    "relaxed gumbel_softmax sample used inside the loss for gradients; ASSUMED not to be "
    "the emitted sequence (emitted tokens come from the decode path). Verify before relying on it.",
  ),
  "inference/optimize_ste.py::make_optimize_sequence_fn.optimize_sequence.update_step.loss_fn": (
    "exempt",
    "relaxed gumbel_softmax sample used inside the loss for gradients; ASSUMED not emitted. "
    "Verify before relying on it.",
  ),
  # --- Inverse-CDF draws in the ported families -----------------------------------------
  "families/potts_mpnn/decode.py::categorical_draw": (
    "exempt",
    "shared inverse-CDF primitive; X exclusion is each caller's job, and each caller is registered",
  ),
  "model/laser/joint_decode.py::categorical_draw": (
    "exempt",
    "shared inverse-CDF primitive; X exclusion is each caller's job, and each caller is registered",
  ),
  "families/potts_mpnn/decode.py::_decode_group.drawn_token": (
    "known_gap",
    "PottsMPNN autoregressive decode. X is excluded only when the caller lists it in omit_aa "
    "(default empty), which mirrors upstream's decoder (omit_AAs, empty in the shipped "
    "config) and is graded against sealed upstream dumps. A default-off flag is debt #2670.",
  ),
  "families/potts_mpnn/refine.py::_sample_logits": (
    "x_excluded",
    "PottsMPNN refine; mask_refine_x makes X undrawable (debt #2260)",
  ),
  "families/potts_mpnn/refine.py::_untied_nodes_sweep.step.run": (
    "x_excluded",
    "PottsMPNN refine, untied sweep; mask_refine_x",
  ),
  "families/potts_mpnn/refine.py::_tied_nodes_group.from_draw": (
    "x_excluded",
    "PottsMPNN refine, tied group; mask_refine_x",
  ),
  "families/protonpotts_mpnn/decode.py::__call__.step": (
    "family_excluded",
    "ProtonPotts AR decode: probabilities are multiplied by (1 - unknown), unknown = 1 at X",
  ),
  "families/protonpotts_mpnn/ph_methods.py::_gibbs_sweep.step": (
    "family_excluded",
    "ProtonPotts Gibbs: energies of tokens outside ctx.valid (valid_token_mask, which drops X) "
    "are +inf, so their probability is exactly 0",
  ),
  "families/protonpotts_mpnn/ph_descent.py::select_joint": (
    "family_excluded",
    "ProtonPotts pH design: block_objective gives +inf to tokens outside valid_token_mask "
    "(which drops X), so exp(-inf) puts zero mass on them",
  ),
  "families/protonpotts_mpnn/ph_export.py::make_visit.visit": (
    "family_excluded",
    "exported pH design step: same block_objective and valid_tokens as select_joint",
  ),
  "model/laser/joint_decode.py::_draw": (
    "family_excluded",
    "LASEr joint decode: disabled residues are floored to finfo.min before the softmax, and "
    "disabled_residues defaults to ('X',). A caller that passes a tuple without X re-enables "
    "it (debt #2670 covers making that one flag). Also draws chi bins, which are not residues.",
  ),
  "model/laser/tied.py::_scan_tied.body": (
    "family_excluded",
    "LASEr tied decode: same disabled-residue floor on both structures' logits; "
    "disabled_residues defaults to ('X',)",
  ),
  "model/laser/tied.py::_one_chi": (
    "exempt",
    "draws a chi rotamer bin, not a residue",
  ),
  # --- Driver-backed families added by the caliby and fampnn stack ------------------------
  "families/caliby/driver.py::_run_potts_sample": (
    "known_gap",
    "Caliby Potts sampler: each sweep is one parallel categorical over all sites (not Gibbs), "
    "from the field of the previous state. X (index 20 in ACDEFGHIKLMNPQRSTVWYX) is not masked; "
    "it is only penalised by options.omit_gap_penalty (run/options.py:172, default 1e6, named "
    "'gap' but applied to index 20). A configured zero re-enables X. Seeds are "
    "fold_in(seed + idx*100 + t_idx), which collides once there are 100 or more temperatures "
    "(debt #3121/#3122). Vectorisation: the item, temperature and sample loops (driver.py "
    "~327-345) are Python loops, which the vectorize rule treats as a defect. Not fixed here.",
  ),
  "families/fampnn/driver.py::mar_sample._mar_step": (
    "family_excluded",
    "FAMPNN MAR unmasking: categorical over the first 20 logits only, which drops X at index 20 "
    "of FAMPNN_ALPHABET ARNDCQEGHILKMFPSTWYVX. Runs inside a fori_loop (allowlisted). Caller-side "
    "defects, not fixed here: the per-sample key comes from fold_in(seed + idx*100 + t_idx) "
    "(driver.py:528), which collides at 100+ temperatures (same as caliby, debt #3121/#3122), and "
    "the samples and temperatures are a Python loop calling mar_sample (driver.py:526-535).",
  ),
}

#: For each ``family_excluded`` site: (file, qualname, exact source text) triples. Every
#: triple must still be present, or the site is no longer known to exclude X.
FAMILY_EVIDENCE: dict[str, tuple[tuple[str, str, str], ...]] = {
  "families/protonpotts_mpnn/decode.py::__call__.step": (
    ("families/protonpotts_mpnn/decode.py", "__call__.step", "kept = probs * (1.0 - unknown)"),
    ("families/protonpotts_mpnn/decode.py", "__call__", "unknown = _unknown_mask(vocab, dtype)"),
    ("families/protonpotts_mpnn/decode.py", "_unknown_mask", ".at[PROTONPOTTS_V6.x_index].set(1.0)"),
  ),
  "families/protonpotts_mpnn/ph_methods.py::_gibbs_sweep.step": (
    ("families/protonpotts_mpnn/ph_methods.py", "_gibbs_sweep.step", "row = jnp.where(valid, row, jnp.inf)"),
    ("families/protonpotts_mpnn/ph_methods.py", "design_methods", "valid_token_mask(config.forbidden_tokens)"),
    ("families/protonpotts_mpnn/ph_plan.py", "valid_token_mask", "mask[_INDEX[_UNK_SYMBOL]] = False"),
  ),
  "families/protonpotts_mpnn/ph_descent.py::select_joint": (
    ("families/protonpotts_mpnn/ph_descent.py", "_allowed_tokens", "member & valid_tokens[None, :]"),
    ("families/protonpotts_mpnn/ph_descent.py", "_joint", "jnp.where(allowed[b], 0.0, jnp.inf)"),
    ("families/protonpotts_mpnn/ph_plan.py", "valid_token_mask", "mask[_INDEX[_UNK_SYMBOL]] = False"),
  ),
  "families/protonpotts_mpnn/ph_export.py::make_visit.visit": (
    ("families/protonpotts_mpnn/ph_export.py", "make_visit.visit", "ph_descent.block_objective("),
    ("families/protonpotts_mpnn/ph_descent.py", "_allowed_tokens", "member & valid_tokens[None, :]"),
  ),
  "families/fampnn/driver.py::mar_sample._mar_step": (
    ("families/fampnn/driver.py", "mar_sample._mar_step", "logits_20 = fwd_out.logits_fampnn[:, :20] / t_eff"),
    (
      "families/fampnn/driver.py",
      "mar_sample._mar_step",
      "sampled_aa = jax.random.categorical(k_cat, logits_20, axis=-1)",
    ),
  ),
  "model/laser/joint_decode.py::_draw": (
    ("model/laser/joint_decode.py", "_run_joint_step", "jnp.where(sampling & inputs.disabled, floor, logits)"),
    ("model/laser/joint_decode.py", "decode_residue", 'disabled_residues: tuple[str, ...] = ("X",)'),
    ("model/laser/joint_decode.py", "decode_order", 'disabled_residues: tuple[str, ...] = ("X",)'),
  ),
  "model/laser/tied.py::_scan_tied.body": (
    ("model/laser/tied.py", "_scan_tied.body", "raw1 = jnp.where(sampling & disabled, floor, raw1)"),
    ("model/laser/tied.py", "_scan_tied.body", "raw2 = jnp.where(sampling & disabled, floor, raw2)"),
    ("model/laser/tied.py", "tied_decode", 'disabled_residues: tuple[str, ...] = ("X",)'),
  ),
}

_STATUSES = {"x_excluded", "family_excluded", "known_gap", "exempt"}


def _function_source(rel: str, qualname: str) -> str | None:
  """Source text of the ``def`` at ``qualname`` in ``src/aminx/<rel>``, or None if absent."""
  source = (SRC_ROOT / rel).read_text()
  node: ast.AST = ast.parse(source)
  for name in qualname.split("."):
    node = next(
      (
        child
        for child in ast.walk(node)
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == name
      ),
      None,
    )
    if node is None:
      return None
  return ast.get_source_segment(source, node)


# --- Negative and positive controls for the scanner itself -------------------------------


def _scan_source(source: str) -> dict[str, dict[str, object]]:
  tree = ast.parse(source)
  sites: dict[str, dict[str, object]] = {}
  _collect(tree, "synthetic.py", [], None, sites)
  return sites


def test_scanner_sees_an_unmasked_categorical_draw() -> None:
  """Negative control: an unmasked draw must be reported as unmasked (the instrument can fail)."""
  sites = _scan_source(
    "import jax\ndef sample(key, logits):\n  return jax.random.categorical(key, logits)\n",
  )
  assert list(sites) == ["synthetic.py::sample"]
  assert sites["synthetic.py::sample"]["masked"] is False


def test_scanner_accepts_a_masked_draw() -> None:
  """Positive control: the same draw after exclude_x_logits is reported as masked."""
  sites = _scan_source(
    "import jax\n"
    "def sample(key, logits):\n"
    "  return jax.random.categorical(key, exclude_x_logits(logits))\n",
  )
  assert sites["synthetic.py::sample"]["masked"] is True


def test_scanner_qualifies_nested_defs() -> None:
  """Two ``step`` functions in one module are two sites, not one."""
  sites = _scan_source(
    "import jax\n"
    "def a():\n"
    "  def step(k, l):\n"
    "    return jax.random.categorical(k, exclude_x_logits(l))\n"
    "def b():\n"
    "  def step(k, l):\n"
    "    return jax.random.categorical(k, l)\n",
  )
  assert sites["synthetic.py::a.step"]["masked"] is True
  assert sites["synthetic.py::b.step"]["masked"] is False


def test_scanner_finds_gumbel_argmax_and_searchsorted() -> None:
  sites = _scan_source(
    "import jax.numpy as jnp\n"
    "import numpy as np\n"
    "def g(x, gumbel):\n"
    "  return jnp.argmax(x + gumbel, axis=-1)\n"
    "def inv(c, u):\n"
    "  return np.searchsorted(c, u)\n",
  )
  assert sites["synthetic.py::g"]["kinds"] == {"gumbel-argmax"}
  assert sites["synthetic.py::inv"]["kinds"] == {"searchsorted"}


def test_scanner_finds_inverse_cdf_draws_but_not_other_cumsums() -> None:
  """An inverse-CDF draw is a site; a cumsum of offsets is not (negative control)."""
  sites = _scan_source(
    "import jax.numpy as jnp\n"
    "def draw(p, u):\n"
    "  cdf = jnp.cumsum(p)\n"
    "  return jnp.sum(cdf <= u * cdf[-1])\n"
    "def caller(p, u):\n"
    "  return categorical_draw(p, u)\n"
    "def offsets(sizes):\n"
    "  starts = jnp.cumsum(sizes) - sizes\n"
    "  return starts\n",
  )
  assert sites["synthetic.py::draw"]["kinds"] == {"inverse-cdf"}
  assert sites["synthetic.py::caller"]["kinds"] == {"inverse-cdf-call"}
  assert "synthetic.py::offsets" not in sites


# --- The registry itself ----------------------------------------------------------------


def test_every_draw_site_is_registered() -> None:
  """A new residue draw cannot appear in src/aminx without a classification here."""
  found = set(find_draw_sites())
  registered = set(REGISTRY)
  assert found - registered == set(), f"unregistered draw sites: {sorted(found - registered)}"
  assert registered - found == set(), f"stale registry entries: {sorted(registered - found)}"


def test_x_excluded_sites_mask_x_in_their_own_function() -> None:
  sites = find_draw_sites()
  for key, (status, _reason) in REGISTRY.items():
    if status == "x_excluded":
      assert sites[key]["masked"] is True, f"{key} is registered x_excluded but does not mask X"


def test_family_excluded_sites_still_carry_their_mask() -> None:
  """Each family mechanism that keeps X out is still in the source, word for word."""
  family = {key for key, (status, _r) in REGISTRY.items() if status == "family_excluded"}
  assert family == set(FAMILY_EVIDENCE), "every family_excluded site needs evidence, and only those"
  for key, triples in FAMILY_EVIDENCE.items():
    assert triples, f"{key} has no evidence"
    for rel, qualname, text in triples:
      body = _function_source(rel, qualname)
      assert body is not None, f"{key}: {rel}::{qualname} no longer exists"
      assert text in body, f"{key}: {rel}::{qualname} no longer contains {text!r}"


def test_evidence_check_fails_on_missing_text() -> None:
  """Negative control: the evidence lookup does report text that is not there."""
  body = _function_source("model/laser/tied.py", "_scan_tied.body")
  assert body is not None
  assert "this text is not in the function" not in body
  assert _function_source("model/laser/tied.py", "no_such_function") is None


def test_known_gaps_are_still_unmasked() -> None:
  """A known gap that gets fixed must be re-classified, so the register never goes stale."""
  sites = find_draw_sites()
  for key, (status, reason) in REGISTRY.items():
    if status == "known_gap":
      assert reason, f"{key} has a known gap with no reason"
      assert sites[key]["masked"] is False, f"{key} is now masked; reclassify it as x_excluded"


def test_registry_statuses_are_valid() -> None:
  for key, (status, reason) in REGISTRY.items():
    assert status in _STATUSES, key
    assert reason, key


@pytest.mark.parametrize("status", sorted(_STATUSES))
def test_each_status_is_used(status: str) -> None:
  """Keeps the three categories honest: an unused category is a sign the rule has drifted."""
  assert any(s == status for s, _ in REGISTRY.values())
