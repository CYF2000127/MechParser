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
    # implicit single bond between adjacent bracket atoms
    (
        "[O-:1][C:2]",
        "[Arrow 1: Start: atom [O-:1] -> End: Single Bond between atoms [O-:1] and [C:2]]",
    ),
    # stereo bond symbols are plain single bonds for the parser
    (
        "[C:1]/[C:2]=[C:3]",
        "[Arrow 1: Start: Double Bond between atoms [C:2] and [C:3] -> End: atom [C:3]]",
    ),
    # triple bond as the electron source (nitrile / alkyne chemistry)
    (
        "[C:1]#[N:2]",
        "[Arrow 1: Start: Triple Bond between atoms [C:1] and [N:2] -> End: atom [N:2]]",
    ),
    (
        "[C:1]#[C:2].[H+:3]",
        "[Arrow 1: Start: Triple Bond between atoms [C:1] and [C:2] -> End: atom [H+:3]]",
    ),
    # 1,2-hydride shift: the through atom H3 is listed first and is not bonded to C1 -> new H3-C1 bond
    (
        "[C+:1]-[C:2]-[H:3]",
        "[Arrow 1: Start: Single Bond between atoms [H:3] and [C:2] -> End: atom [C+:1]]",
    ),
    # elimination written with the adjacent carbon first: C2 is bonded to C1 -> pi bond, C1 is a sink
    (
        "[C+:1]-[C:2]-[H:3]",
        "[Arrow 1: Start: Single Bond between atoms [C:2] and [H:3] -> End: atom [C+:1]]",
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


def test_implicit_bond_annotation_is_inserted():
    steps = arrows_to_steps(
        "[O-:1][C:2]",
        "[Arrow 1: Start: atom [O-:1] -> End: Single Bond between atoms [O-:1] and [C:2]]",
    )
    assert steps == [("A", "[aO-:1];[C:2]")]


def test_triple_bond_source_is_decoded_as_source():
    # ';' means "becomes double": a strengthened single bond or a weakened
    # triple bond. The base scaffold must decide which one it is.
    base = "[C:1]#[N:2]"
    steps = arrows_to_steps(
        base, "[Arrow 1: Start: Triple Bond between atoms [C:1] and [N:2] -> End: atom [N:2]]"
    )
    assert steps == [("A", "[C:1];[yN:2]")]
    op = decode_steps(base, steps)["operations"][0]
    assert op["signature"] == {"source": "bond", "target": "atom", "new_bond": False}
    assert op["start"] == "Triple Bond between atoms [C:1] and [N:2]"
    assert op["end"] == "atom [N:2]"


def test_same_bond_weakened_twice_stays_a_source():
    base = "[C:1]#[N:2].[H+:3]"
    arrows = (
        "[Arrow 1: Start: Triple Bond between atoms [C:1] and [N:2] -> End: atom [H+:3]; "
        "Arrow 2: Start: Double Bond between atoms [C:1] and [N:2] -> End: atom [N:2]]"
    )
    steps = arrows_to_steps(base, arrows)
    assert [s for _, s in steps] == ["[C:1]δ;[N:2].[H+:3]δ", "[C:1]δ_[yN:2].[H+:3]δ"]
    decoded = decode_steps(base, steps)
    assert [op["signature"]["source"] for op in decoded["operations"]] == ["bond", "bond"]
    assert parse_arrows(decoded["arrows"]) == parse_arrows(arrows)


def test_decode_rejects_unchanged_bond_order():
    with pytest.raises(SMSMARTSParseError, match="does not change"):
        decode_steps("[C:1]-[C:2]", {"A": "[C:1]_[yC:2]"})


def test_hydride_shift_marks_the_new_bond_with_deltas():
    steps = arrows_to_steps(
        "[C+:1]-[C:2]-[H:3]",
        "[Arrow 1: Start: Single Bond between atoms [H:3] and [C:2] -> End: atom [C+:1]]",
    )
    step = steps[0][1]
    assert "[C+:1]δ" in step and "[H:3]δ" in step        # new bond between the through atom and the target
    assert "!" in step and "[yC+:1]" not in step         # C2-H3 cleaved; C1 is not a mere sink
    decoded = decode_steps("[C+:1]-[C:2]-[H:3]", steps)
    assert decoded["operations"][0]["signature"]["new_bond"] is True
    assert decoded["arrows"].startswith("[Arrow 1: Start: Single Bond between atoms [H:3] and [C:2]")


def test_elimination_written_carbon_first_has_no_new_bond():
    steps = arrows_to_steps(
        "[C+:1]-[C:2]-[H:3]",
        "[Arrow 1: Start: Single Bond between atoms [C:2] and [H:3] -> End: atom [C+:1]]",
    )
    step = steps[0][1]
    assert "δ" not in step and "[yC+:1]" in step
