"""Structural test bucket — AST / source-grep regression nets.

The tests in this package read the source of code modules and
assert that bug-class patterns (silent zero-fill, broad-except
sites without a sibling carve-out, hardcoded literals that should
flow from a shared constant, etc.) do not reappear after their
fix has shipped. Structural tests complement behavioral tests:
the behavior may pass green for the wrong reason; a structural
test catches the source-shape regression even when no fixture
exercises the code path.
"""
