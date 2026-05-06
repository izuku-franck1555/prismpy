"""Vendored third-party libraries.

This namespace bundles open-source libraries that prismpy depends on
operationally but cannot pull from PyPI as standard dependencies. Each
vendored package keeps its original ``LICENSE`` file alongside the
source so attribution is preserved verbatim.

Why vendor instead of declare-as-dep:
- The library is published only as a git repository without a
  ``pyproject.toml`` / ``setup.py`` build backend, so pip cannot
  install it from a git URL.
- A ``file://`` path-dependency works locally but breaks CI portability
  across machines and shared runners.
- Vendoring with attribution preserves the open-source contract while
  giving prismpy a deterministic, single-source build artifact.

Each vendored package's ``__init__.py`` carries the upstream attribution
header. The ``LICENSE`` file is co-located so wheel builds keep the
license text alongside the redistributed code.
"""
