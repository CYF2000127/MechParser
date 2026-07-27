import pytest

from mechparser.smsmarts import (
    SMSMARTSParseError,
    arrows_to_steps,
    decode_steps,
    parse_arrows,
)


CASES = [
    (
        "[O-:1]-[C:2]",
        "[Arrow 1: Start: atom [O-:1] -> End: Single Bond between atoms [O-:1] and [C:2]]",
    ),
    (
        "[C:1]=[O:2]",
        "[Arrow 1: Start: Double Bond between atoms [C:1] and [O:2] -> End: atom [O:2]]",
    ),
    (
        "[C:1]=[C:2]-[O:3]",
        "[Arrow 1: Start: Double Bond between atoms [C:1] and [C:2] -> End: Single Bond between atoms [C:2] and [O:3]]",
    ),
    (
        "[O-:1].[C:2]",
        "[Arrow 1: Start: atom [O-:1] -> End: atom [C:2]]",
    ),
    (
        "[C:1]=[C:2].[O:3]",
        "[Arrow 1: Start: Double Bond between atoms [C:1] and [C:2] -> End: atom [O:3]]",
    ),
]


@pytest.mark.parametrize(("base_smarts", "arrow_text"), CASES)
def test_encode_decode_round_trip(base_smarts, arrow_text):
    steps = arrows_to_steps(base_smarts, arrow_text)
    decoded = decode_steps(base_smarts, steps)
    assert parse_arrows(decoded["arrows"]) == parse_arrows(arrow_text)


def test_decode_rejects_unpaired_source():
    with pytest.raises(SMSMARTSParseError, match="no electron target"):
        decode_steps("[O-:1].[C:2]", {"A": "[aO-:1].[C:2]"})