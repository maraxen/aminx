#!/usr/bin/env python3
"""Pure-AST extractor for PottsMPNN and LASErMPNN knob surfaces.

Never imports torch or the upstream packages. Reads ``*.py`` as text, parses
with :mod:`ast`, and reads ``VENDOR_PIN.toml`` / example YAML as data.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import keyword
import re
import subprocess
import sys
import tomllib
import warnings
from collections.abc import Mapping, Sequence
from typing import cast
from dataclasses import dataclass, field
from pathlib import Path

_MANUAL_BEGIN = "# MANUAL (not AST-derivable)"
_MANUAL_END = "# END MANUAL"
_SAMPLING_FUNCS = frozenset({"sample_model", "sample", "tied_sample"})
_BUILTIN_NAMES = frozenset({"int", "float", "str", "bool", "bytes", "object"})


@dataclass(frozen=True)
class Knob:
  """One upstream knob, already namespaced into a reference field."""

  repo: str
  slug: str
  class_name: str
  dest: str
  default_repr: str | None
  annotation: str
  type_comment: str
  help_text: str
  source: str
  line: int
  resolved: bool
  unresolved_reason: str

  @property
  def ref(self) -> str:
    return f"{self.slug}__{self.dest}"


@dataclass
class Entry:
  """One generated frozen dataclass."""

  repo: str
  slug: str
  class_name: str
  origin: str
  knobs: list[Knob] = field(default_factory=list)


@dataclass(frozen=True)
class AliasRow:
  """One ``alias_map.toml`` row."""

  ref: str
  targets: tuple[str, ...]
  equivalence: str
  reason: str
  note: str


@dataclass
class Extract:
  """Full extraction plus notes for the structured report."""

  entries: list[Entry]
  unresolved: list[str]


def _pascal(stem: str) -> str:
  parts = re.split(r"[^0-9A-Za-z]+", stem)
  return "".join(part[:1].upper() + part[1:] for part in parts if part)


def _repo_prefix(repo: str) -> str:
  if repo == "laser":
    return "Laser"
  if repo == "protonpotts":
    return "ProtonPotts"
  return "Potts"


def _ident(name: str) -> str:
  cleaned = re.sub(r"[^0-9A-Za-z_]", "_", name)
  if not cleaned or cleaned[0].isdigit():
    cleaned = f"k_{cleaned}"
  if keyword.iskeyword(cleaned):
    cleaned = f"{cleaned}_"
  return cleaned


def _one_line(text: str, limit: int = 240) -> str:
  collapsed = " ".join(text.split())
  if len(collapsed) > limit:
    return collapsed[: limit - 3] + "..."
  return collapsed


def _rel(root: Path, path: Path) -> str:
  try:
    return path.relative_to(root).as_posix()
  except ValueError:
    return path.as_posix()


def _read_commit(root: Path) -> str:
  pin = root / "VENDOR_PIN.toml"
  if not pin.is_file():
    return "MISSING"
  loaded = tomllib.loads(pin.read_text(encoding="utf-8"))
  commit = loaded.get("commit")
  if isinstance(commit, str) and commit:
    return commit
  return "MISSING"


def _protonpotts_commit(root: Path, override: str | None) -> str:
  """The recorded ProtonPottsMPNN commit: the override, else the checkout's git HEAD (no VENDOR_PIN.toml there)."""
  if override:
    return override
  head = root / ".git"
  if not head.exists():
    return "MISSING"
  completed = subprocess.run(  # noqa: S603
    ["git", "-C", str(root), "rev-parse", "HEAD"],  # noqa: S607
    check=False,
    capture_output=True,
    text=True,
  )
  return completed.stdout.strip() if completed.returncode == 0 and completed.stdout.strip() else "MISSING"


def _extractor_sha256() -> str:
  return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _literal(node: ast.AST) -> tuple[bool, object]:
  if isinstance(node, ast.Constant):
    return True, node.value
  if (
    isinstance(node, ast.UnaryOp)
    and isinstance(node.op, (ast.UAdd, ast.USub))
    and isinstance(node.operand, ast.Constant)
    and isinstance(node.operand.value, (int, float))
    and not isinstance(node.operand.value, bool)
  ):
    value = node.operand.value
    if isinstance(node.op, ast.USub):
      return True, -value
    return True, value
  if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
    items: list[object] = []
    for elt in node.elts:
      ok, value = _literal(elt)
      if not ok:
        return False, None
      items.append(value)
    if isinstance(node, ast.List):
      return True, items
    if isinstance(node, ast.Set):
      return True, items
    return True, tuple(items)
  return False, None


def _freeze(value: object) -> object:
  """Replace mutable literals with tuples so dataclass defaults stay legal."""
  if isinstance(value, list):
    return tuple(_freeze(item) for item in value)
  if isinstance(value, tuple):
    return tuple(_freeze(item) for item in value)
  return value


def _annotation_for(value: object, declared: str | None, resolved: bool) -> str:
  if not resolved:
    if declared in _BUILTIN_NAMES:
      return f"{declared} | None"
    return "object | None"
  if declared in _BUILTIN_NAMES and not isinstance(value, tuple):
    if value is None:
      return f"{declared} | None"
    return declared
  if isinstance(value, bool):
    return "bool"
  if isinstance(value, int) and not isinstance(value, bool):
    return "int"
  if isinstance(value, float):
    return "float"
  if isinstance(value, str):
    return "str"
  if value is None:
    return "object | None"
  if isinstance(value, tuple):
    return "tuple[object, ...]"
  return "object"


def _declared_type(node: ast.AST | None) -> str | None:
  if node is None:
    return None
  if isinstance(node, ast.Name) and node.id in _BUILTIN_NAMES:
    return node.id
  if isinstance(node, ast.Attribute):
    return node.attr if node.attr in _BUILTIN_NAMES else None
  return None


def _type_comment(node: ast.AST | None) -> str:
  if node is None:
    return "unknown"
  try:
    return _one_line(ast.unparse(node), 120)
  except (ValueError, TypeError):
    return "unknown"


def _kw_map(call: ast.Call) -> dict[str, ast.AST]:
  mapped: dict[str, ast.AST] = {}
  for kw in call.keywords:
    if kw.arg is not None:
      mapped[kw.arg] = kw.value
  return mapped


def _option_dest(call: ast.Call, keywords: Mapping[str, ast.AST]) -> tuple[str | None, str]:
  if "dest" in keywords:
    ok, value = _literal(keywords["dest"])
    if ok and isinstance(value, str) and value:
      return value, ""
    rendered = _one_line(ast.unparse(keywords["dest"]), 160)
    return None, f"dest is not a string literal ({rendered})"
  strings: list[str] = []
  for arg in call.args:
    ok, value = _literal(arg)
    if ok and isinstance(value, str):
      strings.append(value)
    elif not strings:
      return None, "add_argument option string is not a literal"
  if not strings:
    return None, "add_argument has no option string"
  long_opts = [item for item in strings if item.startswith("--")]
  chosen = long_opts[0] if long_opts else strings[0]
  if chosen.startswith("--"):
    return chosen[2:].replace("-", "_"), ""
  if chosen.startswith("-"):
    return chosen[1:].replace("-", "_"), ""
  return chosen.replace("-", "_"), ""


def _py_files(root: Path) -> list[Path]:
  files = [
    path
    for path in root.rglob("*.py")
    if "__pycache__" not in path.parts and ".git" not in path.parts
  ]
  return sorted(files, key=lambda path: path.as_posix())


def _imports_argparse(tree: ast.AST) -> bool:
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      if any(alias.name == "argparse" or alias.name.startswith("argparse.") for alias in node.names):
        return True
    elif isinstance(node, ast.ImportFrom) and node.module is not None:
      if node.module == "argparse" or node.module.startswith("argparse."):
        return True
  return False


def _has_add_argument(tree: ast.AST) -> bool:
  for node in ast.walk(tree):
    if (
      isinstance(node, ast.Call)
      and isinstance(node.func, ast.Attribute)
      and node.func.attr == "add_argument"
    ):
      return True
  return False


def _enclosing_class(tree: ast.AST, lineno: int) -> str | None:
  best: tuple[int, str] | None = None

  def visit(node: ast.AST, current: str | None) -> None:
    nonlocal best
    if isinstance(node, ast.ClassDef):
      current = node.name
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.lineno == lineno:
      if current is not None and (best is None or node.lineno >= best[0]):
        best = (node.lineno, current)
    for child in ast.iter_child_nodes(node):
      visit(child, current)

  visit(tree, None)
  return None if best is None else best[1]


def _extract_argparse(repo: str, root: Path, path: Path, unresolved: list[str]) -> Entry | None:
  try:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
  except SyntaxError as exc:
    unresolved.append(f"{_rel(root, path)}:{exc.lineno or 0}: syntax error: {exc.msg}")
    return None
  except OSError as exc:
    unresolved.append(f"{_rel(root, path)}:0: unreadable: {exc}")
    return None
  if not _imports_argparse(tree) and not _has_add_argument(tree):
    return None
  stem = path.stem
  slug = _ident(f"{repo}_{stem}")
  class_name = f"{_repo_prefix(repo)}{_pascal(stem)}Knobs"
  entry = Entry(repo=repo, slug=slug, class_name=class_name, origin=_rel(root, path))
  seen: set[str] = set()
  rel = _rel(root, path)
  for node in ast.walk(tree):
    if not (
      isinstance(node, ast.Call)
      and isinstance(node.func, ast.Attribute)
      and node.func.attr == "add_argument"
    ):
      continue
    keywords = _kw_map(node)
    if any(kw.arg is None for kw in node.keywords):
      unresolved.append(f"{rel}:{node.lineno}: add_argument has **kwargs; dest not fully static")
    option_strings: list[str] = []
    for arg in node.args:
      literal_ok, literal_value = _literal(arg)
      if literal_ok and isinstance(literal_value, str):
        option_strings.append(literal_value)
    optional = any(item.startswith("-") for item in option_strings)
    dest, dest_problem = _option_dest(node, keywords)
    if dest is None:
      unresolved.append(f"{rel}:{node.lineno}: {dest_problem}")
      continue
    dest = _ident(dest)
    if dest in seen:
      unresolved.append(f"{rel}:{node.lineno}: duplicate dest {dest}; kept the earlier definition")
      continue
    seen.add(dest)
    action = ""
    if "action" in keywords:
      ok, value = _literal(keywords["action"])
      if ok and isinstance(value, str):
        action = value
      else:
        unresolved.append(
          f"{rel}:{node.lineno}: action for {dest} is not a string literal; default left unresolved",
        )
    if action in {"help", "version"}:
      continue
    declared = _declared_type(keywords.get("type"))
    type_comment = _type_comment(keywords.get("type"))
    help_text = ""
    if "help" in keywords:
      ok, value = _literal(keywords["help"])
      if ok and isinstance(value, str):
        help_text = value
      else:
        try:
          help_text = ast.unparse(keywords["help"])
        except (ValueError, TypeError):
          help_text = ""
        unresolved.append(f"{rel}:{node.lineno}: help for {dest} is not a string literal")
    resolved = True
    unresolved_reason = ""
    default_value: object = None
    if action == "store_true":
      declared = "bool"
      type_comment = "bool"
      if "default" in keywords:
        ok, default_value = _literal(keywords["default"])
        resolved = ok
        if not ok:
          unresolved_reason = "store_true default is not a literal"
      else:
        default_value = False
    elif action == "store_false":
      declared = "bool"
      type_comment = "bool"
      if "default" in keywords:
        ok, default_value = _literal(keywords["default"])
        resolved = ok
        if not ok:
          unresolved_reason = "store_false default is not a literal"
      else:
        default_value = True
    elif action in {"append", "append_const", "extend"} and "default" not in keywords:
      default_value = ()
      type_comment = f"{type_comment}; action={action}"
    elif "default" in keywords:
      ok, default_value = _literal(keywords["default"])
      resolved = ok
      if not ok:
        unresolved_reason = "default is not a literal (" + _one_line(
          ast.unparse(keywords["default"]),
          160,
        ) + ")"
    else:
      required_flag = False
      if "required" in keywords:
        ok, required_value = _literal(keywords["required"])
        required_flag = bool(ok and required_value is True)
      if optional and not required_flag and action in {"", "store"}:
        # argparse's implicit default for an optional argument is None.
        resolved = True
        default_value = None
      elif required_flag:
        resolved = False
        unresolved_reason = "required; no upstream default"
      else:
        resolved = False
        unresolved_reason = (
          "no default literal" if not action else f"action={action} has no literal default"
        )
    if not resolved:
      unresolved.append(f"{rel}:{node.lineno}: {dest}: {unresolved_reason}")
      default_repr = "None"
      default_value = None
    else:
      frozen = _freeze(default_value)
      default_value = frozen
      default_repr = repr(frozen)
    if action and action not in {"store", "store_true", "store_false"}:
      type_comment = f"{type_comment}; action={action}"
    choices = ""
    if "choices" in keywords:
      ok, value = _literal(keywords["choices"])
      if ok:
        choices = repr(_freeze(value))
      else:
        unresolved.append(f"{rel}:{node.lineno}: choices for {dest} is not a literal")
        choices = "UNRESOLVED"
    if choices:
      type_comment = f"{type_comment}; choices={choices}"
    entry.knobs.append(
      Knob(
        repo=repo,
        slug=slug,
        class_name=class_name,
        dest=dest,
        default_repr=default_repr,
        annotation=_annotation_for(default_value, declared, resolved),
        type_comment=type_comment,
        help_text=help_text,
        source=rel,
        line=node.lineno,
        resolved=resolved,
        unresolved_reason=unresolved_reason,
      ),
    )
  entry.knobs.sort(key=lambda knob: knob.dest)
  return entry


def _param_knob(
  repo: str,
  slug: str,
  class_name: str,
  source: str,
  func: ast.FunctionDef | ast.AsyncFunctionDef,
  arg: ast.arg,
  default: ast.AST | None,
  unresolved: list[str],
) -> Knob:
  dest = _ident(arg.arg)
  declared = _declared_type(arg.annotation)
  type_comment = _type_comment(arg.annotation)
  if default is None:
    resolved = False
    reason = "required parameter; no upstream default"
    unresolved.append(f"{source}:{func.lineno}: {func.name}.{dest}: {reason}")
    return Knob(
      repo=repo,
      slug=slug,
      class_name=class_name,
      dest=dest,
      default_repr="None",
      annotation=_annotation_for(None, declared, False),
      type_comment=type_comment,
      help_text=f"parameter of {func.name}",
      source=source,
      line=arg.lineno,
      resolved=False,
      unresolved_reason=reason,
    )
  ok, value = _literal(default)
  if not ok:
    reason = "default is not a literal (" + _one_line(ast.unparse(default), 160) + ")"
    unresolved.append(f"{source}:{arg.lineno}: {func.name}.{dest}: {reason}")
    return Knob(
      repo=repo,
      slug=slug,
      class_name=class_name,
      dest=dest,
      default_repr="None",
      annotation=_annotation_for(None, declared, False),
      type_comment=type_comment,
      help_text=f"parameter of {func.name}",
      source=source,
      line=arg.lineno,
      resolved=False,
      unresolved_reason=reason,
    )
  frozen = _freeze(value)
  return Knob(
    repo=repo,
    slug=slug,
    class_name=class_name,
    dest=dest,
    default_repr=repr(frozen),
    annotation=_annotation_for(frozen, declared, True),
    type_comment=type_comment,
    help_text=f"parameter of {func.name}",
    source=source,
    line=arg.lineno,
    resolved=True,
    unresolved_reason="",
  )


def _function_entries(
  repo: str,
  root: Path,
  path: Path,
  unresolved: list[str],
  names: frozenset[str] = _SAMPLING_FUNCS,
) -> list[Entry]:
  try:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
  except (SyntaxError, OSError) as exc:
    if isinstance(exc, SyntaxError):
      unresolved.append(f"{_rel(root, path)}:{exc.lineno or 0}: syntax error: {exc.msg}")
    return []
  rel = _rel(root, path)
  stem = path.stem
  found: list[Entry] = []
  for node in ast.walk(tree):
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
      continue
    if node.name not in names:
      continue
    owner = _enclosing_class(tree, node.lineno)
    slug = _ident(f"{repo}_{stem}_{node.name}")
    class_name = f"{_repo_prefix(repo)}{_pascal(stem)}{_pascal(node.name)}Knobs"
    if owner is not None:
      slug = _ident(f"{repo}_{stem}_{owner}_{node.name}")
      class_name = f"{_repo_prefix(repo)}{_pascal(stem)}{_pascal(owner)}{_pascal(node.name)}Knobs"
    args = node.args
    positional = list(args.posonlyargs) + list(args.args)
    defaults = list(args.defaults)
    pad = len(positional) - len(defaults)
    paired: list[tuple[ast.arg, ast.AST | None]] = []
    for index, arg in enumerate(positional):
      if arg.arg in {"self", "cls"}:
        continue
      default = defaults[index - pad] if index >= pad else None
      paired.append((arg, default))
    kw_defaults = list(args.kw_defaults)
    for arg, default in zip(args.kwonlyargs, kw_defaults, strict=True):
      paired.append((arg, default))
    entry = Entry(
      repo=repo,
      slug=slug,
      class_name=class_name,
      origin=f"{rel}:{node.lineno}:{node.name}",
    )
    for arg, default in paired:
      entry.knobs.append(
        _param_knob(repo, slug, class_name, rel, node, arg, default, unresolved),
      )
    entry.knobs.sort(key=lambda knob: (knob.dest, knob.line))
    found.append(entry)
  return found


# ProtonPottsMPNN has no argparse CLI on its design path. Its knobs are the fields of one dataclass plus the
# arguments of one engine method, both in the vendored foundry package (spec §50).
_PROTONPOTTS_ENGINE = Path("foundry/models/mpnn/src/mpnn/inference_engines/potts_mpnn_ph.py")
_PROTONPOTTS_DATACLASSES = frozenset({"PHDesignCriteria"})
_PROTONPOTTS_FUNCS = frozenset({"run_ph_redesign"})


def _field_default(value: ast.AST | None) -> tuple[ast.AST | None, str]:
  """The default expression of a dataclass field, looking through ``field(default=/default_factory=)``.

  Returns ``(expression, "")``, or ``(None, reason)`` when the field has no static default.
  """
  if value is None:
    return None, "required field; no upstream default"
  if not (isinstance(value, ast.Call) and _call_name(value) == "field"):
    return value, ""
  keywords = _kw_map(value)
  if "default" in keywords:
    return keywords["default"], ""
  factory = keywords.get("default_factory")
  if isinstance(factory, ast.Lambda):
    return factory.body, ""
  if isinstance(factory, ast.Name) and factory.id == "list":
    return ast.List(elts=[], ctx=ast.Load()), ""
  if factory is not None:
    return None, "default_factory is not a lambda (" + _one_line(ast.unparse(factory), 120) + ")"
  return None, "field() has neither default nor default_factory"


def _call_name(call: ast.Call) -> str:
  if isinstance(call.func, ast.Name):
    return call.func.id
  if isinstance(call.func, ast.Attribute):
    return call.func.attr
  return ""


def _protonpotts_dataclass_entries(root: Path, path: Path, unresolved: list[str]) -> list[Entry]:
  """One Entry per named dataclass in ``path``; each annotated field is one knob."""
  rel = _rel(root, path)
  tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
  found: list[Entry] = []
  for node in ast.walk(tree):
    if not (isinstance(node, ast.ClassDef) and node.name in _PROTONPOTTS_DATACLASSES):
      continue
    slug = _ident(f"protonpotts_{path.stem}_{node.name}")
    class_name = f"ProtonPotts{_pascal(path.stem)}{_pascal(node.name)}Knobs"
    entry = Entry(repo="protonpotts", slug=slug, class_name=class_name, origin=f"{rel}:{node.lineno}:{node.name}")
    for stmt in node.body:
      if not (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)):
        continue
      dest = _ident(stmt.target.id)
      declared = _declared_type(stmt.annotation)
      expression, reason = _field_default(stmt.value)
      resolved = False
      value: object = None
      if expression is not None:
        resolved, value = _literal(expression)
        if not resolved:
          reason = "default is not a literal (" + _one_line(ast.unparse(expression), 160) + ")"
      if not resolved:
        unresolved.append(f"{rel}:{stmt.lineno}: {node.name}.{dest}: {reason}")
      frozen = _freeze(value) if resolved else None
      entry.knobs.append(
        Knob(
          repo="protonpotts",
          slug=slug,
          class_name=class_name,
          dest=dest,
          default_repr=repr(frozen) if resolved else "None",
          annotation=_annotation_for(frozen, declared, resolved),
          type_comment=_type_comment(stmt.annotation),
          help_text=f"field of {node.name}",
          source=rel,
          line=stmt.lineno,
          resolved=resolved,
          unresolved_reason="" if resolved else reason,
        ),
      )
    entry.knobs.sort(key=lambda knob: (knob.dest, knob.line))
    found.append(entry)
  return found


def _extract_protonpotts(root: Path, unresolved: list[str]) -> list[Entry]:
  engine = root / _PROTONPOTTS_ENGINE
  if not engine.is_file():
    unresolved.append(f"{_rel(root, engine)}:0: ProtonPottsMPNN engine file is absent")
    return []
  entries = _protonpotts_dataclass_entries(root, engine, unresolved)
  entries.extend(_function_entries("protonpotts", root, engine, unresolved, names=_PROTONPOTTS_FUNCS))
  found = {entry.origin.rsplit(":", 1)[-1] for entry in entries}
  for name in sorted((_PROTONPOTTS_DATACLASSES | _PROTONPOTTS_FUNCS) - found):
    unresolved.append(f"{_rel(root, engine)}:0: expected {name} was not found")
  return entries


def _cfg_chain(node: ast.AST) -> tuple[str, str] | None:
  if not isinstance(node, ast.Attribute):
    return None
  parent = node.value
  if (
    isinstance(parent, ast.Attribute)
    and isinstance(parent.value, ast.Name)
    and parent.value.id == "cfg"
    and parent.attr in {"inference", "model"}
  ):
    return parent.attr, node.attr
  return None


def _membership_name(node: ast.Compare) -> str | None:
  if len(node.ops) != 1 or not isinstance(node.ops[0], ast.In):
    return None
  if len(node.comparators) != 1:
    return None
  comparator = node.comparators[0]
  if not (
    isinstance(comparator, ast.Attribute)
    and isinstance(comparator.value, ast.Name)
    and comparator.value.id == "cfg"
    and comparator.attr == "inference"
  ):
    return None
  ok, value = _literal(node.left)
  if ok and isinstance(value, str) and value:
    return value
  return None


def _extract_cfg_reads(root: Path, unresolved: list[str]) -> dict[tuple[str, str], tuple[str, int]]:
  found: dict[tuple[str, str], tuple[str, int]] = {}
  for path in _py_files(root):
    try:
      tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, OSError) as exc:
      if isinstance(exc, SyntaxError):
        unresolved.append(f"{_rel(root, path)}:{exc.lineno or 0}: syntax error: {exc.msg}")
      continue
    rel = _rel(root, path)
    for node in ast.walk(tree):
      if isinstance(node, ast.Attribute):
        chain = _cfg_chain(node)
        if chain is not None and chain not in found:
          found[chain] = (rel, node.lineno)
      elif isinstance(node, ast.Compare):
        name = _membership_name(node)
        if name is None:
          if (
            len(node.ops) == 1
            and isinstance(node.ops[0], ast.In)
            and len(node.comparators) == 1
            and isinstance(node.comparators[0], ast.Attribute)
            and isinstance(node.comparators[0].value, ast.Name)
            and node.comparators[0].value.id == "cfg"
            and node.comparators[0].attr == "inference"
          ):
            unresolved.append(
              f"{rel}:{node.lineno}: membership test on cfg.inference is not a string literal",
            )
        elif ("inference", name) not in found:
          found[("inference", name)] = (rel, node.lineno)
  return found


def _parse_scalar(text: str) -> object:
  if text in {"null", "Null", "NULL", "~"}:
    return None
  if text in {"true", "True", "TRUE"}:
    return True
  if text in {"false", "False", "FALSE"}:
    return False
  if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
    return text[1:-1]
  if text.startswith("[") and text.endswith("]"):
    inner = text[1:-1].strip()
    if not inner:
      return []
    parts = [part.strip() for part in inner.split(",")]
    return [_parse_scalar(part) for part in parts if part]
  if re.fullmatch(r"-?\d+", text):
    return int(text)
  if re.fullmatch(r"-?\d+\.\d+", text):
    return float(text)
  return text


def _parse_yaml_documents(text: str) -> dict[str, tuple[object, int]]:
  """Parse the flat example configs. Values remember their 1-based line."""
  raw_lines = text.splitlines()
  items: list[tuple[int, int, str]] = []
  for index, raw in enumerate(raw_lines, start=1):
    if not raw.strip() or raw.lstrip().startswith("#"):
      continue
    indent = len(raw) - len(raw.lstrip(" "))
    items.append((index, indent, raw.strip()))

  def parse_map(pos: int, indent: int) -> tuple[dict[str, tuple[object, int]], int]:
    mapped: dict[str, tuple[object, int]] = {}
    while pos < len(items) and items[pos][1] == indent:
      line_no, _item_indent, content = items[pos]
      key, sep, rest = content.partition(":")
      if not sep:
        pos += 1
        continue
      key = key.strip()
      rest = rest.strip()
      if rest == "":
        if pos + 1 < len(items) and items[pos + 1][1] > indent:
          child, pos = parse_map(pos + 1, items[pos + 1][1])
          mapped[key] = (child, line_no)
        else:
          mapped[key] = (None, line_no)
          pos += 1
      else:
        mapped[key] = (_parse_scalar(rest), line_no)
        pos += 1
    return mapped, pos

  if not items:
    return {}
  parsed, _ = parse_map(0, items[0][1])
  return parsed


def _flatten_yaml(
  node: object,
  line: int,
  prefix: tuple[str, ...] = (),
) -> list[tuple[tuple[str, ...], object, int]]:
  if isinstance(node, dict):
    rows: list[tuple[tuple[str, ...], object, int]] = []
    mapping = cast(dict[str, tuple[object, int]], node)
    for key in sorted(mapping):
      child, child_line = mapping[key]
      rows.extend(_flatten_yaml(child, child_line, prefix + (str(key),)))
    return rows
  return [(prefix, node, line)]


def _extract_potts_cfg(root: Path, unresolved: list[str]) -> Entry:
  slug = "pottsmpnn_cfg"
  class_name = "PottsmpnnCfgKnobs"
  entry = Entry(repo="potts", slug=slug, class_name=class_name, origin="cfg.model/cfg.inference ∪ example YAML")
  values: dict[str, list[tuple[object, str, int]]] = {}
  yaml_paths = sorted(root.glob("inputs/example_config_*.yaml"))
  if not yaml_paths:
    unresolved.append(f"{root}/inputs/example_config_*.yaml:0: no example config YAML found")
  for path in yaml_paths:
    rel = _rel(root, path)
    try:
      parsed = _parse_yaml_documents(path.read_text(encoding="utf-8"))
    except OSError as exc:
      unresolved.append(f"{rel}:0: unreadable YAML: {exc}")
      continue
    for key in sorted(parsed):
      child, line = parsed[key]
      for dotted, value, value_line in _flatten_yaml(child, line, (key,)):
        dest = _ident("__".join(dotted))
        values.setdefault(dest, []).append((value, rel, value_line))
  reads = _extract_cfg_reads(root, unresolved)
  for section, name in sorted(reads):
    dest = _ident(f"{section}__{name}")
    source, line = reads[(section, name)]
    values.setdefault(dest, []).append((None, source, line))
    # AST-only keys carry a None sentinel that must not count as a YAML default.
  knobs: list[Knob] = []
  for dest in sorted(values):
    observations = values[dest]
    yaml_obs = [item for item in observations if not item[1].endswith(".py")]
    ast_obs = [item for item in observations if item[1].endswith(".py")]
    chosen_source, chosen_line = (yaml_obs or ast_obs)[0][1], (yaml_obs or ast_obs)[0][2]
    resolved = True
    reason = ""
    default_value: object = None
    if yaml_obs:
      unique = {repr(item[0]) for item in yaml_obs}
      if len(unique) == 1:
        default_value = _freeze(yaml_obs[0][0])
        chosen_source, chosen_line = yaml_obs[0][1], yaml_obs[0][2]
      else:
        resolved = False
        reason = "YAML defaults disagree: " + "; ".join(
          f"{src}:{line}={value!r}" for value, src, line in yaml_obs
        )
        unresolved.append(f"{yaml_obs[0][1]}:{yaml_obs[0][2]}: {dest}: {reason}")
        chosen_source, chosen_line = yaml_obs[0][1], yaml_obs[0][2]
    else:
      resolved = False
      reason = "cfg attribute is read but has no example-YAML default"
      source, line = ast_obs[0][1], ast_obs[0][2]
      unresolved.append(f"{source}:{line}: {dest}: {reason}")
      chosen_source, chosen_line = source, line
    if ast_obs and yaml_obs:
      ast_source, ast_line = ast_obs[0][1], ast_obs[0][2]
      help_text = f"yaml+ast; first read {ast_source}:{ast_line}"
    elif yaml_obs:
      help_text = "example YAML key"
    else:
      help_text = "cfg attribute read"
    knobs.append(
      Knob(
        repo="potts",
        slug=slug,
        class_name=class_name,
        dest=dest,
        default_repr=repr(default_value) if resolved else "None",
        annotation=_annotation_for(default_value, None, resolved),
        type_comment="yaml" if yaml_obs else "cfg-attr",
        help_text=help_text,
        source=chosen_source,
        line=chosen_line,
        resolved=resolved,
        unresolved_reason=reason,
      ),
    )
  entry.knobs = knobs
  return entry


def _subscript_chain(node: ast.Subscript) -> list[str] | None:
  keys: list[str] = []
  current: ast.AST = node
  while isinstance(current, ast.Subscript):
    slice_node = current.slice
    ok, value = _literal(slice_node)
    if not ok or not isinstance(value, str):
      return None
    keys.append(value)
    current = current.value
  keys.reverse()
  return keys


def _extract_checkpoint_params(root: Path, unresolved: list[str]) -> Entry | None:
  """LASEr ``model_params`` keys ``build_hydrogens`` and ``graph_structure.*``."""
  slug = "laser_checkpoint"
  class_name = "LaserCheckpointParamsKnobs"
  literals: dict[str, list[tuple[object, str, int]]] = {}
  mentions: dict[str, tuple[str, int]] = {}

  def remember_literal(dest: str, value: object, source: str, line: int) -> None:
    literals.setdefault(dest, []).append((value, source, line))

  def remember_mention(dest: str, source: str, line: int) -> None:
    mentions.setdefault(dest, (source, line))

  for path in _py_files(root):
    if "tests" in path.parts:
      continue
    try:
      tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, OSError):
      continue
    rel = _rel(root, path)
    for node in ast.walk(tree):
      if isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values, strict=False):
          if key is None or value is None:
            continue
          ok, key_value = _literal(key)
          if not ok or not isinstance(key_value, str):
            continue
          if key_value == "build_hydrogens":
            lit_ok, lit = _literal(value)
            if lit_ok:
              remember_literal("build_hydrogens", lit, rel, key.lineno)
            else:
              remember_mention("build_hydrogens", rel, key.lineno)
              unresolved.append(
                f"{rel}:{key.lineno}: build_hydrogens default is not a literal",
              )
          if key_value == "graph_structure" and isinstance(value, ast.Dict):
            remember_mention("graph_structure", rel, key.lineno)
            for subkey, subvalue in zip(value.keys, value.values, strict=False):
              if subkey is None or subvalue is None:
                unresolved.append(
                  f"{rel}:{key.lineno}: graph_structure has a non-literal key",
                )
                continue
              sub_ok, sub_name = _literal(subkey)
              if not sub_ok or not isinstance(sub_name, str):
                unresolved.append(
                  f"{rel}:{getattr(subkey, 'lineno', key.lineno)}: graph_structure key is not a string",
                )
                continue
              dest = _ident(f"graph_structure__{sub_name}")
              val_ok, val = _literal(subvalue)
              if val_ok:
                remember_literal(dest, val, rel, subkey.lineno)
              else:
                remember_mention(dest, rel, subkey.lineno)
                unresolved.append(
                  f"{rel}:{subkey.lineno}: {dest} default is not a literal",
                )
      elif isinstance(node, ast.Subscript):
        chain = _subscript_chain(node)
        if chain is None:
          continue
        if "build_hydrogens" in chain and "model_params" in chain:
          remember_mention("build_hydrogens", rel, node.lineno)
        if "graph_structure" in chain:
          remember_mention("graph_structure", rel, node.lineno)
          index = chain.index("graph_structure")
          if index + 1 < len(chain):
            dest = _ident("graph_structure__" + "__".join(chain[index + 1 :]))
            remember_mention(dest, rel, node.lineno)
  dests = sorted((set(literals) | set(mentions)) - {"graph_structure"})
  if not dests:
    return None
  entry = Entry(
    repo="laser",
    slug=slug,
    class_name=class_name,
    origin="model_params build_hydrogens / graph_structure literals and subscript reads",
  )
  for dest in dests:
    obs = literals.get(dest, [])
    unique = {repr(_freeze(item[0])) for item in obs}
    if len(unique) == 1:
      frozen = _freeze(obs[0][0])
      source, line = obs[0][1], obs[0][2]
      entry.knobs.append(
        Knob(
          repo="laser",
          slug=slug,
          class_name=class_name,
          dest=dest,
          default_repr=repr(frozen),
          annotation=_annotation_for(frozen, None, True),
          type_comment="checkpoint literal",
          help_text="checkpoint-derived model_params key",
          source=source,
          line=line,
          resolved=True,
          unresolved_reason="",
        ),
      )
    else:
      if len(unique) > 1:
        reason = "checkpoint literals disagree: " + "; ".join(
          f"{src}:{line}={value!r}" for value, src, line in obs
        )
      else:
        reason = "checkpoint key is read but has no literal default"
      source, line = mentions.get(dest, (obs[0][1], obs[0][2]) if obs else ("", 0))
      unresolved.append(f"{source}:{line}: {dest}: {reason}")
      entry.knobs.append(
        Knob(
          repo="laser",
          slug=slug,
          class_name=class_name,
          dest=dest,
          default_repr="None",
          annotation="object | None",
          type_comment="checkpoint",
          help_text="checkpoint-derived model_params key",
          source=source,
          line=line,
          resolved=False,
          unresolved_reason=reason,
        ),
      )
  return entry


def extract(potts_root: Path, laser_root: Path, protonpotts_root: Path | None = None) -> Extract:
  """Walk the pinned trees and return sorted entry points.

  ``protonpotts_root`` is optional: without it the output is exactly what the two-root extractor produced.
  """
  unresolved: list[str] = []
  entries: list[Entry] = []
  for repo, root in (("potts", potts_root), ("laser", laser_root)):
    if not root.is_dir():
      unresolved.append(f"{root}:0: upstream root is absent")
      continue
    for path in _py_files(root):
      parsed = _extract_argparse(repo, root, path, unresolved)
      if parsed is not None and parsed.knobs:
        entries.append(parsed)
      if repo == "laser":
        entries.extend(_function_entries(repo, root, path, unresolved))
    if repo == "potts":
      entries.append(_extract_potts_cfg(root, unresolved))
    if repo == "laser":
      checkpoint = _extract_checkpoint_params(root, unresolved)
      if checkpoint is not None:
        entries.append(checkpoint)
  # Potts public sampling methods are entry points too (sample / tied_sample).
  if potts_root.is_dir():
    for path in _py_files(potts_root):
      entries.extend(_function_entries("potts", potts_root, path, unresolved))
  if protonpotts_root is not None:
    if protonpotts_root.is_dir():
      entries.extend(_extract_protonpotts(protonpotts_root, unresolved))
    else:
      unresolved.append(f"{protonpotts_root}:0: upstream root is absent")
  deduped: dict[str, Entry] = {}
  for entry in entries:
    if not entry.knobs:
      continue
    if entry.class_name in deduped:
      unresolved.append(
        f"{entry.origin}:0: duplicate class {entry.class_name}; merged into the first",
      )
      existing = deduped[entry.class_name]
      have = {knob.dest for knob in existing.knobs}
      for knob in entry.knobs:
        if knob.dest not in have:
          existing.knobs.append(knob)
          have.add(knob.dest)
      existing.knobs.sort(key=lambda knob: knob.dest)
    else:
      deduped[entry.class_name] = entry
  ordered = sorted(deduped.values(), key=lambda entry: entry.class_name)
  for entry in ordered:
    entry.knobs.sort(key=lambda knob: knob.ref)
  return Extract(entries=ordered, unresolved=sorted(set(unresolved)))


def _manual_block() -> str:
  return "\n".join(
    [
      _MANUAL_BEGIN,
      "# Input-list rows parsed by PottsMPNN sample_seqs.py:81-94.",
      "# Line format: pdb|designed:chains|fixed:chains",
      "@dataclass(frozen=True)",
      "class pottsmpnn_input_list:",
      '  """Manual input-list fields. The extractor --check diff skips this block."""',
      "",
      "  pottsmpnn_input_list__pdb: str | None = None  "
      "# type=str; help=PDB stem (text before the first |); sample_seqs.py:81-94",
      "  pottsmpnn_input_list__designed_chains: str | None = None  "
      "# type=str; help=designed chains (colon-separated); sample_seqs.py:81-94",
      "  pottsmpnn_input_list__fixed_chains: str | None = None  "
      "# type=str; help=fixed chains (colon-separated); sample_seqs.py:81-94",
      _MANUAL_END,
      "",
    ],
  )


def _knob_line(knob: Knob) -> str:
  default = knob.default_repr if knob.default_repr is not None else "None"
  help_text = _one_line(knob.help_text) if knob.help_text else ""
  flag = "" if knob.resolved else f"; UNRESOLVED {knob.unresolved_reason}"
  comment = (
    f"type={knob.type_comment}; help={help_text}; {knob.source}:{knob.line}{flag}"
  )
  return f"  {knob.ref}: {knob.annotation} = {default}  # {_one_line(comment, 500)}"


def render_surfaces(
  extracted: Extract,
  potts_commit: str,
  laser_commit: str,
  sha256: str,
  protonpotts_commit: str | None = None,
) -> str:
  """Render ``reference_surfaces.py`` including the manual block."""
  names = "PottsMPNN and LASErMPNN" if protonpotts_commit is None else "PottsMPNN, LASErMPNN and ProtonPottsMPNN"
  lines: list[str] = [
    f'"""Generated reference surfaces for upstream {names} knobs.',
    "",
    "Produced by scripts/redsox/extract_upstream_knobs.py. Do not edit the",
    "generated dataclasses by hand. The manual block is the exception.",
    '"""',
    "",
    "from __future__ import annotations",
    "",
    "from dataclasses import dataclass",
    "",
    f'POTTSMPNN_COMMIT = "{potts_commit}"',
    f'LASERMPNN_COMMIT = "{laser_commit}"',
    *([] if protonpotts_commit is None else [f'PROTONPOTTSMPNN_COMMIT = "{protonpotts_commit}"']),
    f'EXTRACTOR_SHA256 = "{sha256}"',
    "",
  ]
  for entry in extracted.entries:
    lines.append("@dataclass(frozen=True)")
    lines.append(f"class {entry.class_name}:")
    lines.append(f'  """{entry.origin}."""')
    lines.append("")
    if not entry.knobs:
      lines.append("  pass")
    for knob in entry.knobs:
      lines.append(_knob_line(knob))
    lines.append("")
  lines.append(_manual_block())
  return "\n".join(lines)


def _strip_manual(text: str) -> str:
  kept: list[str] = []
  skipping = False
  for line in text.splitlines(keepends=True):
    stripped = line.strip()
    if stripped == _MANUAL_BEGIN:
      skipping = True
      continue
    if stripped == _MANUAL_END:
      skipping = False
      continue
    if not skipping:
      kept.append(line)
  return "".join(kept)


def _dest_is(dest: str, name: str) -> bool:
  return dest == name or dest.endswith(f"__{name}")


def _stem(slug: str) -> str:
  for prefix in ("laser_", "potts_"):
    if slug.startswith(prefix):
      return slug[len(prefix) :]
  return slug


def classify(knob: Knob) -> AliasRow:
  """Fill rows the spec already decides; every other row stays TODO."""
  dest = knob.dest
  stem = _stem(knob.slug)
  ref = knob.ref

  if stem == "mutation_search" or stem.startswith("mutation_search_"):
    return AliasRow(
      ref=ref,
      targets=(),
      equivalence="exclusion",
      reason="deferred:2070",
      note="mutation_search.* deferred (aminx tech debt 2070)",
    )
  if _dest_is(dest, "entropy_decoder"):
    return AliasRow(
      ref=ref,
      targets=(),
      equivalence="exclusion",
      reason="deferred:2069",
      note="entropy_decoder deferred (aminx tech debt 2069)",
    )
  if _dest_is(dest, "ca_only"):
    return AliasRow(
      ref=ref,
      targets=(),
      equivalence="exclusion",
      reason="deferred:2068",
      note="CA-only deferred (aminx tech debt 2068)",
    )
  if _dest_is(dest, "vocab"):
    return AliasRow(
      ref=ref,
      targets=(),
      equivalence="exclusion",
      reason="deferred:2067",
      note="MSA vocab-22 deferred (aminx tech debt 2067)",
    )
  if _dest_is(dest, "chain_dict_json"):
    return AliasRow(
      ref=ref,
      targets=("fixed_mask", "chain_design_mask_json"),
      equivalence="semantic",
      reason="",
      note="chain dict JSON maps onto fixed_mask and chain_design_mask_json",
    )
  if _dest_is(dest, "fix_decoding_order") or _dest_is(dest, "decoding_order_offset"):
    return AliasRow(
      ref=ref,
      targets=("random_seed",),
      equivalence="semantic",
      reason="",
      note="seeded-order test; order source is the driver, not decoding_order_fn",
    )
  if knob.repo == "potts" and _dest_is(dest, "noise"):
    return AliasRow(
      ref=ref,
      targets=("noise",),
      equivalence="semantic",
      reason="",
      note="eval augment_eps on all backbone atoms; knob test is iid Gaussian N/CA/C/O",
    )
  if knob.repo == "laser" and (_dest_is(dest, "bb_noise") or _dest_is(dest, "backbone_noise")):
    return AliasRow(
      ref=ref,
      targets=("noise",),
      equivalence="semantic",
      reason="",
      note="LASEr bb_noise / CLI dest backbone_noise maps to noise",
    )
  if _dest_is(dest, "repack_all"):
    return AliasRow(
      ref=ref,
      targets=("repack_all",),
      equivalence="identical",
      reason="",
      note="",
    )
  if _dest_is(dest, "repack_only_input_sequence") and stem.startswith("run_batch_inference"):
    return AliasRow(
      ref=ref,
      targets=("repack_only",),
      equivalence="semantic",
      reason="",
      note="repack_only_input_sequence maps to repack_only",
    )
  if _dest_is(dest, "repack_only"):
    return AliasRow(
      ref=ref,
      targets=("repack_only",),
      equivalence="semantic",
      reason="",
      note="",
    )
  if stem == "run_proofreading" and _dest_is(dest, "disable_inference_dropout"):
    return AliasRow(
      ref=ref,
      targets=("proofread_dropout",),
      equivalence="semantic",
      reason="",
      note="inverted relative to proofread_dropout",
    )
  if stem.startswith("run_batch_inference") and _dest_is(dest, "ignore_key_mismatch"):
    return AliasRow(
      ref=ref,
      targets=("strict_load",),
      equivalence="semantic",
      reason="",
      note=(
        "store_false, passed as strict= (run_batch_inference.py:250,379); "
        "dest True ≡ strict_load=True; name/polarity inversion"
      ),
    )
  if stem.startswith("run_inference") and _dest_is(dest, "strict_load"):
    return AliasRow(
      ref=ref,
      targets=("strict_load",),
      equivalence="semantic",
      reason="",
      note=(
        "CLI --ignore_statedict_mismatch is store_false (run_inference.py:783); "
        "dest True ≡ strict_load=True; name/polarity inversion"
      ),
    )
  if knob.repo == "laser" and _dest_is(dest, "sequence_temp"):
    return AliasRow(
      ref=ref,
      targets=("temperature",),
      equivalence="semantic",
      reason="",
      note="family default None; maps to temperature",
    )
  if (
    _dest_is(dest, "designs_per_batch")
    or _dest_is(dest, "max_tokens")
    or _dest_is(dest, "inputs_processed_simultaneously")
  ):
    return AliasRow(
      ref=ref,
      targets=("batch_size",),
      equivalence="semantic",
      reason="",
      note="output-invariance test; maps to batch_size",
    )
  if _dest_is(dest, "model_weights") or _dest_is(dest, "check_path"):
    return AliasRow(
      ref=ref,
      targets=("checkpoint_id", "model_local_path"),
      equivalence="semantic",
      reason="",
      note="weights path maps to checkpoint_id and model_local_path",
    )
  if _dest_is(dest, "device"):
    return AliasRow(
      ref=ref,
      targets=(),
      equivalence="exclusion",
      reason="device",
      note="",
    )
  if _dest_is(dest, "verbose") or _dest_is(dest, "disable_pbar") or _dest_is(dest, "silent"):
    return AliasRow(
      ref=ref,
      targets=(),
      equivalence="exclusion",
      reason="io_only",
      note="",
    )
  if _dest_is(dest, "filter"):
    return AliasRow(
      ref=ref,
      targets=(),
      equivalence="exclusion",
      reason="upstream_noop",
      note="upstream filter drops nothing; it is not a PottsMPNN option",
    )
  if _dest_is(dest, "optimize_fasta"):
    return AliasRow(
      ref=ref,
      targets=("optimize_fasta",),
      equivalence="divergence",
      reason="",
      note="upstream asserts optimize_fasta then reads out_dir/out_name.fasta (§6.5b)",
    )
  if _dest_is(dest, "ddG"):
    return AliasRow(
      ref=ref,
      targets=("output_kind",),
      equivalence="semantic",
      reason="",
      note="True ≡ output_kind ddg; False ≡ energy",
    )
  if knob.slug == "pottsmpnn_cfg" and dest.startswith("model__"):
    return AliasRow(
      ref=ref,
      targets=(),
      equivalence="exclusion",
      reason="checkpoint_derived",
      note="model.* keys are checkpoint-derived",
    )
  if (
    knob.slug == "laser_checkpoint"
    or dest == "graph_structure"
    or dest.startswith("graph_structure__")
    or _dest_is(dest, "build_hydrogens")
  ):
    return AliasRow(
      ref=ref,
      targets=(),
      equivalence="exclusion",
      reason="checkpoint_derived",
      note="model.* / graph_structure.* / build_hydrogens are checkpoint-derived",
    )
  return AliasRow(ref=ref, targets=(), equivalence="TODO", reason="", note="")


def _manual_rows() -> list[AliasRow]:
  return [
    AliasRow(
      ref="pottsmpnn_input_list__pdb",
      targets=(),
      equivalence="TODO",
      reason="",
      note="",
    ),
    AliasRow(
      ref="pottsmpnn_input_list__designed_chains",
      targets=("fixed_mask", "chain_design_mask_json"),
      equivalence="semantic",
      reason="",
      note="input-list designed chains map onto fixed_mask and chain_design_mask_json",
    ),
    AliasRow(
      ref="pottsmpnn_input_list__fixed_chains",
      targets=("fixed_mask", "chain_design_mask_json"),
      equivalence="semantic",
      reason="",
      note="input-list fixed chains map onto fixed_mask and chain_design_mask_json",
    ),
  ]


def _toml_string(text: str) -> str:
  escaped = text.replace("\\", "\\\\").replace('"', '\\"')
  return f'"{escaped}"'


def render_alias(rows: Sequence[AliasRow]) -> str:
  """Render the alias skeleton. One ``[[row]]`` per reference field."""
  blocks: list[str] = [
    "# Alias skeleton for tests/knob_gate/reference_surfaces.py.",
    "# Spec-decided rows (§6.3, §6.4) are filled. Every other row is TODO.",
    "",
  ]
  for row in sorted(rows, key=lambda item: item.ref):
    targets = ", ".join(_toml_string(target) for target in row.targets)
    blocks.append("[[row]]")
    blocks.append(f"ref = {_toml_string(row.ref)}")
    blocks.append(f"targets = [{targets}]")
    blocks.append(f"equivalence = {_toml_string(row.equivalence)}")
    if row.reason:
      blocks.append(f"reason = {_toml_string(row.reason)}")
    blocks.append("parity_test_ids = []")
    blocks.append(f"note = {_toml_string(row.note)}")
    blocks.append("")
  return "\n".join(blocks)


def alias_rows(extracted: Extract) -> list[AliasRow]:
  rows = [classify(knob) for entry in extracted.entries for knob in entry.knobs]
  rows.extend(_manual_rows())
  return rows


def _check(expected_path: Path, fresh: str) -> int:
  if not expected_path.is_file():
    print(f"drift: {expected_path} is missing", file=sys.stderr)
    return 1
  current = _strip_manual(expected_path.read_text(encoding="utf-8"))
  regenerated = _strip_manual(fresh)
  if current == regenerated:
    return 0
  diff = difflib.unified_diff(
    current.splitlines(),
    regenerated.splitlines(),
    fromfile=str(expected_path),
    tofile="regenerated",
    lineterm="",
  )
  print("reference surface drift (manual block ignored):", file=sys.stderr)
  sys.stderr.write("\n".join(diff))
  sys.stderr.write("\n")
  return 1


def _counts(extracted: Extract) -> list[str]:
  return [f"{entry.class_name}={len(entry.knobs)}" for entry in extracted.entries]


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--potts-root", type=Path, required=True)
  parser.add_argument("--laser-root", type=Path, required=True)
  parser.add_argument(
    "--protonpotts-root",
    type=Path,
    default=None,
    help="Optional ProtonPottsMPNN checkout; adds its PHDesignCriteria and run_ph_redesign surfaces.",
  )
  parser.add_argument(
    "--protonpotts-commit",
    default=None,
    help="Commit to record for --protonpotts-root (it ships no VENDOR_PIN.toml); default is its git HEAD.",
  )
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument(
    "--alias-out",
    type=Path,
    default=None,
    help="Alias skeleton path. Defaults to alias_map.toml beside --out.",
  )
  parser.add_argument(
    "--check",
    action="store_true",
    help="Regenerate in memory and diff against --out, skipping the manual block.",
  )
  args = parser.parse_args(argv)
  warnings.filterwarnings("ignore", category=SyntaxWarning)
  potts_root = args.potts_root.expanduser().resolve()
  laser_root = args.laser_root.expanduser().resolve()
  protonpotts_root = args.protonpotts_root.expanduser().resolve() if args.protonpotts_root else None
  extracted = extract(potts_root, laser_root, protonpotts_root)
  sha = _extractor_sha256()
  text = render_surfaces(
    extracted,
    _read_commit(potts_root),
    _read_commit(laser_root),
    sha,
    None if protonpotts_root is None else _protonpotts_commit(protonpotts_root, args.protonpotts_commit),
  )
  if args.check:
    return _check(args.out, text)
  alias_path = args.alias_out
  if alias_path is None:
    alias_path = args.out.parent / "alias_map.toml"
    if alias_path.exists():
      # The skeleton below is TODO rows. Writing it over the curated map once discarded every hand-set mapping.
      print(
        f"REFUSING: {alias_path} exists and is curated; pass --alias-out <scratch path> to write a skeleton",
        file=sys.stderr,
      )
      return 2
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(text, encoding="utf-8")
  alias_path.write_text(render_alias(alias_rows(extracted)), encoding="utf-8")
  print(f"wrote {args.out}")
  print(f"wrote {alias_path}")
  print("counts " + " ".join(_counts(extracted)))
  if extracted.unresolved:
    print(f"unresolved {len(extracted.unresolved)}")
    for item in extracted.unresolved:
      print(f"UNRESOLVED {item}")
  else:
    print("unresolved 0")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
