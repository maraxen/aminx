import importlib.util as u

n = "molxmpnn_absent_probe_pkg"
for label, fn in [
    ("dotted_find_spec_absent_parent", lambda: u.find_spec(n + ".scoring.score")),
    ("parent_find_spec_absent", lambda: u.find_spec(n)),
    ("guard_parent_first", lambda: (u.find_spec(n) is not None) and (u.find_spec(n + ".scoring.score") is not None)),
]:
    try:
        print(label, "->", repr(fn()))
    except Exception as e:
        print(label, "-> RAISES", type(e).__name__, e)
