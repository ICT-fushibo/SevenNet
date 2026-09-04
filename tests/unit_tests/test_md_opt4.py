"""CPU contract checks for the SevenNet Opt4 route."""

import pytest

from sevenn.md_stages import opt4


def test_opt4_rejects_other_route() -> None:
    with pytest.raises(ValueError, match="SevenNet Opt4 route"):
        opt4.run_md(type("Request", (), {"model": "dpa4", "stage": "opt4"})())
