"""
Reward functions for chemical arrow prediction GRPO training.

Naming convention: functions ending with _reward are auto-discovered by load_reward_funcs().
"""
import re


# ---------------------------------------------------------------------------
#  Parsing & normalization helpers
# ---------------------------------------------------------------------------

def _parse_arrows(text):
    """Parse arrow predictions/references into structured list.
    Returns list of dicts: [{'start': str, 'end': str}, ...]
    """
    arrows = []
    parts = re.split(r'Arrow\s+\d+\s*:', text)
    for part in parts:
        part = part.strip().rstrip(';').strip()
        if not part:
            continue
        m = re.match(r'Start:\s*(.+?)\s*->\s*End:\s*(.+)', part)
        if m:
            arrows.append({'start': m.group(1).strip(), 'end': m.group(2).strip()})
    return arrows


def _normalize_bond_entity(entity_str):
    """Normalize bond entity so atom order doesn't matter.
    'Double Bond between atoms [C:5] and [C:4]' -> 'double bond between atoms [c:4] and [c:5]'
    For non-bond entities, return as-is (lowered).
    """
    entity_lower = entity_str.strip().lower()
    m = re.match(
        r'(single|double|triple|aromatic)\s+bond\s+between\s+atoms\s+(\[[^\]]+\])\s+and\s+(\[[^\]]+\])',
        entity_lower,
    )
    if m:
        bond_type, atom1, atom2 = m.group(1), m.group(2), m.group(3)
        a, b = sorted([atom1, atom2])
        return f"{bond_type} bond between atoms {a} and {b}"
    return entity_lower


_CHARGE_RE = re.compile(r'(?<=\[)(\w+?)([+\-]+)(?=:\w+\])')


def _strip_charges(entity_str):
    """Strip charge symbols (+/-) from atom labels inside brackets.
    '[N+:3]' -> '[N:3]',  '[O-:5]' -> '[O:5]',  '[Al-:8]' -> '[Al:8]'
    Handles multi-char charges like '2+' / '++' (rare).
    """
    return _CHARGE_RE.sub(r'\1', entity_str)


def _normalize_charge_agnostic(entity_str):
    """Bond-order normalize AND strip charges."""
    return _normalize_bond_entity(_strip_charges(entity_str))


def _extract_atom_ids(entity_str):
    """Extract all atom IDs like [C:21], [N+:3], [H+:unknown] from an entity string."""
    return set(re.findall(r'\[\w+[+\-]*:\w+\]', entity_str.lower()))


def _extract_atom_ids_no_charge(entity_str):
    """Extract atom IDs after stripping charges, e.g. [N+:3]->[N:3]."""
    return _extract_atom_ids(_strip_charges(entity_str))


def _extract_bond_type(entity_str):
    """Extract bond type (single/double/triple/aromatic), or None for atoms."""
    m = re.match(r'(single|double|triple|aromatic)\s+bond', entity_str.strip().lower())
    return m.group(1) if m else None


def _is_atom_entity(entity_str):
    return entity_str.strip().lower().startswith('atom ')


def _is_bond_entity(entity_str):
    return _extract_bond_type(entity_str) is not None


# ---------------------------------------------------------------------------
#  Entity-level scoring (single Start or End entity)
# ---------------------------------------------------------------------------

def _score_entity(pred_e, ref_e):
    """Score a single entity (Start or End) with multi-tier matching.

    Returns a float in [0, 1]:
      1.0  - normalized exact match  (bond-order invariant)
      0.85 - charge-agnostic match   (element+id correct, charge wrong)
      0.4  - bond: atoms match but bond type wrong
      0.3  - bond: atoms match (charge-agnostic) but bond type wrong
      0.25 - atom IDs overlap (charge-agnostic) but entity type differs (atom vs bond)
      0.0  - nothing matches
    """
    # Tier 1: full normalized match (bond-order invariant, exact charges)
    if _normalize_bond_entity(pred_e) == _normalize_bond_entity(ref_e):
        return 1.0

    # Tier 2: charge-agnostic normalized match
    if _normalize_charge_agnostic(pred_e) == _normalize_charge_agnostic(ref_e):
        return 0.85

    # From here, extract structural info
    pred_bt = _extract_bond_type(pred_e)
    ref_bt = _extract_bond_type(ref_e)
    pred_atoms = _extract_atom_ids(pred_e)
    ref_atoms = _extract_atom_ids(ref_e)
    pred_atoms_nc = _extract_atom_ids_no_charge(pred_e)
    ref_atoms_nc = _extract_atom_ids_no_charge(ref_e)

    # Both are bonds
    if pred_bt is not None and ref_bt is not None:
        if pred_atoms == ref_atoms:
            # Tier 3a: atoms exact, bond type wrong
            return 0.4
        if pred_atoms_nc == ref_atoms_nc and ref_atoms_nc:
            # Tier 3b: atoms charge-agnostic match, bond type wrong
            return 0.3

    # Both are atoms
    if _is_atom_entity(pred_e) and _is_atom_entity(ref_e):
        if pred_atoms_nc == ref_atoms_nc and ref_atoms_nc:
            # Tier 2b: atom entity, element+id correct, charge wrong
            # (should be caught by tier 2 above, but fallback)
            return 0.85
        return 0.0

    # Entity type mismatch (atom vs bond) — still give partial if atoms overlap
    if ref_atoms_nc and pred_atoms_nc & ref_atoms_nc:
        return 0.25

    return 0.0


# ---------------------------------------------------------------------------
#  Main reward functions
# ---------------------------------------------------------------------------

def arrow_accuracy_reward(completions, assistant, **kwargs):
    """Per-arrow partial match reward with bond-order normalization and charge awareness.
    
    Per-arrow scoring (matched positionally: Arrow 1 vs Arrow 1):
      - start_score = _score_entity(pred_start, ref_start)  [0..1]
      - end_score   = _score_entity(pred_end,   ref_end)    [0..1]
      - arrow_score = (start_score + end_score) / 2
    
    Arrow count penalty: -0.2 per extra or missing arrow
    Final reward = sum(arrow_scores) / num_ref_arrows, clamped to [0, 1]
    """
    rewards = []

    for completion, sol in zip(completions, assistant):
        completion = completion.strip()
        sol = sol.strip()

        # Exact match shortcut
        if completion.lower() == sol.lower():
            rewards.append(1.0)
            continue

        pred_arrows = _parse_arrows(completion)
        ref_arrows = _parse_arrows(sol)

        if not ref_arrows:
            rewards.append(1.0 if not pred_arrows else 0.0)
            continue

        score = 0.0
        num_ref = len(ref_arrows)

        for i, ref_a in enumerate(ref_arrows):
            if i >= len(pred_arrows):
                continue
            pred_a = pred_arrows[i]

            s_score = _score_entity(pred_a['start'], ref_a['start'])
            e_score = _score_entity(pred_a['end'], ref_a['end'])
            score += (s_score + e_score) / 2.0

        # Penalty for wrong arrow count
        count_diff = abs(len(pred_arrows) - num_ref)
        score -= count_diff * 0.2

        reward = max(0.0, min(1.0, score / num_ref))
        rewards.append(reward)

    return rewards


def arrow_format_reward(completions, **kwargs):
    """Reward for correct output format: [Arrow 1: Start: ... -> End: ...; Arrow 2: ...]
    
    Graduated scoring:
      1.0  - full valid format with parseable arrows
      0.5  - has brackets and arrow pattern but can't parse arrows
      0.3  - starts with [ and contains "Arrow" keyword
      0.1  - starts with [ (at least attempted bracket structure)
      0.0  - no structure at all
    """
    rewards = []
    pattern = r'\[Arrow\s+\d+\s*:\s*Start:\s*.+?\s*->\s*End:\s*.+?\]'

    for completion in completions:
        completion = completion.strip()

        if not completion.startswith('['):
            rewards.append(0.0)
            continue

        if not completion.endswith(']'):
            # Starts with [ but doesn't end with ] — partial credit
            if re.search(r'Arrow\s+\d+\s*:', completion):
                arrows = _parse_arrows(completion)
                if len(arrows) > 0:
                    rewards.append(0.5)
                else:
                    rewards.append(0.3)
            else:
                rewards.append(0.1)
            continue

        # Both brackets present
        if re.search(pattern, completion, re.DOTALL):
            arrows = _parse_arrows(completion)
            if len(arrows) > 0:
                rewards.append(1.0)
            else:
                rewards.append(0.5)
        elif re.search(r'Arrow\s+\d+\s*:', completion):
            rewards.append(0.3)
        else:
            rewards.append(0.1)

    return rewards


def _arrow_count_score(completions, assistant, **kwargs):
    """Arrow count scoring helper (not auto-discovered as reward).
    
    Kept for reference / potential manual use.
    1.0 if arrow count matches exactly,
    0.5 if off by 1,
    0.0 otherwise.
    """
    rewards = []
    for completion, sol in zip(completions, assistant):
        pred_arrows = _parse_arrows(completion.strip())
        ref_arrows = _parse_arrows(sol.strip())
        if not ref_arrows:
            rewards.append(1.0 if not pred_arrows else 0.0)
            continue
        diff = abs(len(pred_arrows) - len(ref_arrows))
        if diff == 0:
            rewards.append(1.0)
        elif diff == 1:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards
