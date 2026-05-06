"""Pin every error-handling site in vendored modules so a bare ``except:``
or a broad ``except Exception:`` cannot land without a follow-up
``raise`` (or an explicit allow-list comment).

The F-AL-v2 substrate-hardening sweep surfaced that a chain of
silent-swallow ``except`` clauses in the upstream
``SARRA_data_download.get_AgERA5_data`` module hid a CDS-side error
behind a misleading ``FileNotFoundError`` five levels removed from
the real cause. The vendoring playbook accepts deliberate deviations
from upstream-verbatim when an upstream pattern violates the
silent-skip discipline F-AL closed at the prismpy layer; this
structural pin makes that discipline enforceable.

A guard counts as compliant when:

- The ``except`` re-raises directly (``raise`` with no argument), OR
- The ``except`` raises a different exception class (chained or
  not), OR
- The ``except`` body or the line above it carries the literal
  comment ``# vendor-no-bare-except: allow-listed`` plus a short
  rationale.

The third option is for places where the vendored library
deliberately wants to swallow a low-impact error (e.g., a non-
fatal cleanup race). The allow-list discipline matches the existing
``_KNOWN_LOCAL_ONLY_OPTIONAL_IMPORTS`` allow-list in
``test_climate_source_imports_declared.py``: every entry should be
paired with a sprint-review checkpoint.

Anti-mutation drill: introduce a bare ``except: pass`` (or
``except Exception: print(...)`` without raise) inside any
``prismpy/vendor/**/*.py`` module — this test must fail with the
file path + line number + a diagnostic naming the offender.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_VENDOR_DIR = _REPO_ROOT / "src" / "prismpy" / "vendor"

# Allow-list comment marker. Any except-handler whose body's first
# line (or the line directly above the ``except`` keyword) contains
# this marker is exempt from the no-bare-except discipline.
_ALLOW_MARKER = "vendor-no-bare-except: allow-listed"


def _handler_is_broad(handler: ast.ExceptHandler) -> bool:
    """True iff ``handler`` is a bare ``except:`` or a broad
    ``except Exception:`` / ``except BaseException:``. Specific
    exception classes (``except ImportError:``, ``except OSError:``,
    ``except cdsapi.Error:`` etc.) are not flagged."""
    exc = handler.type
    if exc is None:
        return True
    if isinstance(exc, ast.Name) and exc.id in ("Exception", "BaseException"):
        return True
    if isinstance(exc, ast.Tuple):
        for element in exc.elts:
            if isinstance(element, ast.Name) and element.id in (
                "Exception", "BaseException",
            ):
                return True
    return False


def _handler_re_raises(handler: ast.ExceptHandler) -> bool:
    """True iff the handler body terminates with a ``raise`` (bare
    or with an argument). Walks the statement list directly because
    ``ast.walk`` would descend into nested function defs whose
    raises are unrelated to the outer handler."""
    for stmt in handler.body:
        if isinstance(stmt, ast.Raise):
            return True
    return False


def _handler_is_allow_listed(
    source_lines: list[str], handler: ast.ExceptHandler,
) -> bool:
    """True iff the handler is annotated with the allow-list comment
    marker. The marker may appear on the line directly above the
    ``except`` keyword, on the ``except`` line itself (after the
    colon), or on the first line of the handler body."""
    candidate_linenos = [handler.lineno - 1, handler.lineno]
    if handler.body:
        candidate_linenos.append(handler.body[0].lineno)
    for lineno in candidate_linenos:
        # Source lines are 1-indexed in AST; list is 0-indexed.
        idx = lineno - 1
        if 0 <= idx < len(source_lines):
            if _ALLOW_MARKER in source_lines[idx]:
                return True
    return False


def _scan_module(py_file: Path) -> list[str]:
    """Return one diagnostic per offending broad-except handler in
    ``py_file``. An offender is a broad handler that neither re-
    raises nor carries the allow-list marker."""
    text = py_file.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(py_file))
    source_lines = text.splitlines()

    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for handler in node.handlers:
            if not _handler_is_broad(handler):
                continue
            if _handler_re_raises(handler):
                continue
            if _handler_is_allow_listed(source_lines, handler):
                continue
            label = "bare except" if handler.type is None else "except Exception"
            offenders.append(
                f"{py_file.relative_to(_REPO_ROOT)}:{handler.lineno} "
                f"({label}) — neither re-raises nor carries the "
                f"`{_ALLOW_MARKER}` allow-list marker"
            )
    return offenders


class TestVendorNoBareExcept(unittest.TestCase):
    """Every broad ``except`` handler in ``prismpy/vendor/**/*.py``
    must either re-raise or be explicitly allow-listed. The F-AL-v2
    audit converted three upstream silent-skip sites in
    ``vendor/sarra_data_download/get_AgERA5_data.py`` to fail-loud
    ``raise`` patterns; this pin keeps a future refactor (or a
    re-vendor pass that pulls upstream-verbatim) from silently
    re-introducing the silent-skip class.
    """

    def test_no_silent_swallow_in_vendor_modules(self):
        py_files = sorted(_VENDOR_DIR.rglob("*.py"))
        self.assertGreater(
            len(py_files), 0,
            f"expected at least one .py file under {_VENDOR_DIR}",
        )
        all_offenders: list[str] = []
        for py_file in py_files:
            all_offenders.extend(_scan_module(py_file))
        self.assertEqual(
            all_offenders, [],
            "Vendored modules must not silent-swallow errors. The "
            "F-AL-v2 substrate-hardening sweep converted three "
            "upstream sites to fail-loud raises; a regression here "
            "would re-introduce the silent-skip chain that hid CDS-"
            "side errors behind a misleading FileNotFoundError. "
            "Either re-raise from the broad handler, narrow the "
            "exception class, or add the explicit allow-list "
            "comment with a short rationale:\n  "
            + "\n  ".join(all_offenders),
        )


if __name__ == "__main__":
    unittest.main()
