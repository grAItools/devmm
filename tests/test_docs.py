"""Executable documentation: every Python code block in the README and the
API docs is a doctest session and runs green, so the docs can never drift
from the library (release gate: docs executed as tests).
"""

from __future__ import annotations

import doctest
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
DOC_FILES = ("README.md", "docs/api.md")

_PYTHON_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)


@pytest.mark.parametrize("relpath", DOC_FILES)
def test_every_python_block_is_a_doctest_session(relpath: str) -> None:
    text = (_ROOT / relpath).read_text(encoding="utf-8")
    blocks = _PYTHON_FENCE.findall(text)
    assert blocks, f"{relpath} documents no executable Python examples"
    for block in blocks:
        assert ">>> " in block, (
            f"{relpath} has a python code block without doctest prompts; "
            f"docs examples must execute:\n{block}"
        )


@pytest.mark.parametrize("relpath", DOC_FILES)
def test_doc_examples_execute_green(relpath: str) -> None:
    result = doctest.testfile(
        str(_ROOT / relpath),
        module_relative=False,
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
    )
    assert result.attempted > 0
    assert result.failed == 0
