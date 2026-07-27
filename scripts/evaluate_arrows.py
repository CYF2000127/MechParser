"""
Full eval script: load a LoRA checkpoint and run inference on ALL eval samples
with detailed arrow-level metrics (bond-order-invariant).

Usage: python scripts/full_eval.py --checkpoint <path> --eval_json <path> --image_folder <path>
"""
import argparse
import json
import os
import re
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None, help="Path to LoRA checkpoint dir")
    p.add_argument("--base_model", type=str, default=None, help="Path to base model (no LoRA)")
    p.add_argument("--eval_json", type=str, required=True, help="Path to eval JSON file")
    p.add_argument("--image_folder", type=str, required=True, help="Path to image folder")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--num_samples", type=int, default=None, help="Limit number of samples (None=all)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_json", type=str, default=None, help="Save metrics and predictions as JSON")
    return p.parse_args()


def _normalize_bond_entity(entity_str):
    """Normalize bond entity so atom order doesn't matter.
    'Double Bond between atoms [C:5] and [C:4]' -> 'Double Bond between atoms [C:4] and [C:5]'
    For non-bond entities, return as-is (lowered).
    """
    entity_lower = entity_str.strip().lower()
    # Match: <type> bond between atoms [X:N] and [Y:M]
    m = re.match(
        r'(single|double|triple|aromatic)\s+bond\s+between\s+atoms\s+(\[[^\]]+\])\s+and\s+(\[[^\]]+\])',
        entity_lower
    )
    if m:
        bond_type = m.group(1)
        atom1 = m.group(2)
        atom2 = m.group(3)
        # Sort atoms alphabetically for canonical form
        a, b = sorted([atom1, atom2])
        return f"{bond_type} bond between atoms {a} and {b}"
    return entity_lower


def _parse_arrows(text):
    """Parse arrow predictions/references into structured list."""
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


def _extract_atom_ids(entity_str):
    """Extract all atom IDs like [C:21] from an entity string."""
    return set(re.findall(r'\[\w+:\d+\]', entity_str))


def _extract_bond_type(entity_str):
    """Extract bond type (single/double/triple/aromatic) from entity string, or None if it's an atom."""
    m = re.match(
        r'(single|double|triple|aromatic)\s+bond',
        entity_str.strip().lower()
    )
    return m.group(1) if m else None


def bond_type_and_atoms_match(pred_entity, ref_entity):
    """Check if bond type matches AND atom ID sets match (order-invariant).
    For atom entities (no bond type), just check atom IDs match.
    """
    pred_bt = _extract_bond_type(pred_entity)
    ref_bt = _extract_bond_type(ref_entity)
    # Both must be bond or both atom
    if (pred_bt is None) != (ref_bt is None):
        return False
    # If bonds, check bond type matches
    if pred_bt is not None and pred_bt != ref_bt:
        return False
    # Check atom IDs match (set comparison, order-invariant)
    pred_atoms = _extract_atom_ids(pred_entity)
    ref_atoms = _extract_atom_ids(ref_entity)
    if not ref_atoms:
        return False
    return pred_atoms == ref_atoms


def entities_match(pred_entity, ref_entity):
    """Compare two entities with bond-order-invariant normalization."""
    return _normalize_bond_entity(pred_entity) == _normalize_bond_entity(ref_entity)


def _arrow_key(arrow):
    """Canonical key for order-free arrow matching."""
    return (_normalize_bond_entity(arrow['start']),
            _normalize_bond_entity(arrow['end']))


# ── Equivalence-aware exact match (EA-EM) ───────────────────────────────────
# The ground-truth arrow order is only a *partial* order. It is induced by
# electron-flow chaining: arrows that do not feed one another are unordered,
# and a closed cycle has no chemically determined first arrow. EA-EM accepts
# any linear extension of that partial order, with cyclic components matched
# modulo rotation. Arrow direction, identity and count are never relaxed.
# Seq-EM (`exact_match_normalized`) additionally demands the single stored
# serialization and is therefore strictly stronger.

SHAPES = ("total_order", "cycle", "branching_or_disconnected")


def _atom_ids(entity_str):
    """Atom-map indices referenced by an entity string."""
    return {int(m) for m in re.findall(r'\[\w+:(\d+)\]', entity_str)}


def _chain_edges(arrows):
    """i -> j when arrow i's end feeds arrow j's start (they share an atom)."""
    edges = set()
    ends = [_atom_ids(a['end']) for a in arrows]
    starts = [_atom_ids(a['start']) for a in arrows]
    for i in range(len(arrows)):
        for j in range(len(arrows)):
            if i != j and ends[i] & starts[j]:
                edges.add((i, j))
    return edges


def _closure(n, edges):
    """Transitive closure; n is the arrow count, so an O(n^3) pass is ample."""
    reach = [[False] * n for _ in range(n)]
    for i, j in edges:
        reach[i][j] = True
    for k in range(n):
        for i in range(n):
            if reach[i][k]:
                for j in range(n):
                    if reach[k][j]:
                        reach[i][j] = True
    return reach


def _components(n, reach):
    """Mutually reachable arrows form one component (a cycle if size > 1)."""
    comp = [-1] * n
    groups = []
    for i in range(n):
        if comp[i] >= 0:
            continue
        comp[i] = len(groups)
        group = [i]
        for j in range(i + 1, n):
            if comp[j] < 0 and reach[i][j] and reach[j][i]:
                comp[j] = len(groups)
                group.append(j)
        groups.append(group)
    return comp, groups


def classify_shape(ref_arrows):
    """Shape of the partial order: total order, single cycle, or neither."""
    n = len(ref_arrows)
    if n <= 1:
        return "total_order"
    reach = _closure(n, _chain_edges(ref_arrows))
    _, groups = _components(n, reach)
    if len(groups) == 1 and len(groups[0]) == n:
        return "cycle"
    comparable = all(reach[i][j] or reach[j][i]
                     for i in range(n) for j in range(i + 1, n))
    return "total_order" if comparable else "branching_or_disconnected"


def _is_rotation(seq, ref):
    """True when seq is a cyclic rotation of ref (direction preserved)."""
    if len(seq) != len(ref):
        return False
    if not ref:
        return True
    doubled = ref + ref
    return any(doubled[s:s + len(ref)] == seq for s in range(len(ref)))


def ea_em_match(pred_arrows, ref_arrows):
    """Equivalence-aware exact match: any linear extension of the reference
    partial order, cyclic components modulo rotation."""
    n = len(ref_arrows)
    if len(pred_arrows) != n:
        return False

    # 1. every predicted arrow must realise a distinct reference arrow
    used = [False] * n
    perm = []
    for pred_a in pred_arrows:
        hit = -1
        for j, ref_a in enumerate(ref_arrows):
            if used[j]:
                continue
            if (entities_match(pred_a['start'], ref_a['start'])
                    and entities_match(pred_a['end'], ref_a['end'])):
                hit = j
                break
        if hit < 0:
            return False
        used[hit] = True
        perm.append(hit)
    if n <= 1:
        return True

    reach = _closure(n, _chain_edges(ref_arrows))
    comp, groups = _components(n, reach)
    pos = {ref_idx: k for k, ref_idx in enumerate(perm)}

    # 2. a real dependency between components may never be inverted
    for i in range(n):
        for j in range(n):
            if i == j or comp[i] == comp[j]:
                continue
            if reach[i][j] and pos[i] > pos[j]:
                return False

    # 3. within a cycle only rotations of the stored order are admissible
    for group in groups:
        if len(group) < 2:
            continue
        members = set(group)
        predicted = [r for r in perm if r in members]
        if not _is_rotation(predicted, sorted(group)):
            return False
    return True


def compute_metrics(predictions, references):
    """Compute full arrow-level metrics with bond-order-invariant comparison."""
    n = len(predictions)
    if n == 0:
        return {}

    exact_match = 0
    exact_match_normalized = 0  # with bond-order normalization
    arrow_count_correct = 0
    start_correct = 0
    end_correct = 0
    start_atoms_correct = 0
    end_atoms_correct = 0
    start_bt_atoms_correct = 0  # bond type + atoms both correct
    end_bt_atoms_correct = 0
    bt_atoms_arrow_correct = 0  # both start & end bond_type+atoms correct
    total_ref_arrows = 0
    total_pred_arrows = 0
    arrows_fully_correct = 0        # order-free (multiset) matching
    arrows_correct_positional = 0   # positional, kept as a diagnostic
    all_arrows_correct = 0
    chain_correct = 0
    total_chains = 0

    error_total = 0
    error_correct = 0
    arrow_n = 0

    ea_exact_match = 0
    shape_total = {s: 0 for s in SHAPES}
    shape_ea = {s: 0 for s in SHAPES}
    shape_seq = {s: 0 for s in SHAPES}

    for pred_text, ref_text in zip(predictions, references):
        pred_text = pred_text.strip()
        ref_text = ref_text.strip()

        is_error_sample = ref_text.startswith("Error:")

        # Strict exact match
        if pred_text == ref_text:
            exact_match += 1

        # Track error/rejection samples
        if is_error_sample:
            error_total += 1
            if pred_text.strip().startswith("Error:"):
                error_correct += 1
            continue

        arrow_n += 1

        pred_arrows = _parse_arrows(pred_text)
        ref_arrows = _parse_arrows(ref_text)
        total_pred_arrows += len(pred_arrows)

        if len(pred_arrows) == len(ref_arrows):
            arrow_count_correct += 1

        sample_all_correct = True
        for i, ref_a in enumerate(ref_arrows):
            total_ref_arrows += 1
            if i >= len(pred_arrows):
                sample_all_correct = False
                continue
            pred_a = pred_arrows[i]

            s_ok = entities_match(pred_a['start'], ref_a['start'])
            e_ok = entities_match(pred_a['end'], ref_a['end'])

            if s_ok:
                start_correct += 1
            else:
                sample_all_correct = False
            if e_ok:
                end_correct += 1
            else:
                sample_all_correct = False

            if s_ok and e_ok:
                arrows_correct_positional += 1

            # Atom-level match (only atom IDs, ignoring bond type)
            pred_start_atoms = _extract_atom_ids(pred_a['start'])
            ref_start_atoms = _extract_atom_ids(ref_a['start'])
            if ref_start_atoms and pred_start_atoms == ref_start_atoms:
                start_atoms_correct += 1

            pred_end_atoms = _extract_atom_ids(pred_a['end'])
            ref_end_atoms = _extract_atom_ids(ref_a['end'])
            if ref_end_atoms and pred_end_atoms == ref_end_atoms:
                end_atoms_correct += 1

            # Bond type + atoms match (键类型+原子序号都对)
            s_bt_ok = bond_type_and_atoms_match(pred_a['start'], ref_a['start'])
            e_bt_ok = bond_type_and_atoms_match(pred_a['end'], ref_a['end'])
            if s_bt_ok:
                start_bt_atoms_correct += 1
            if e_bt_ok:
                end_bt_atoms_correct += 1
            if s_bt_ok and e_bt_ok:
                bt_atoms_arrow_correct += 1

        if len(pred_arrows) != len(ref_arrows):
            sample_all_correct = False
        if sample_all_correct:
            all_arrows_correct += 1
            exact_match_normalized += 1

        # Arrow-level matching is order-free: an arrow counts as recovered when
        # the prediction contains it anywhere, so a cyclic mechanism reported
        # from a different starting arrow is not penalised at the arrow level.
        # (The positional variant is retained above as a diagnostic.)
        pool = [_arrow_key(a) for a in pred_arrows]
        for ref_a in ref_arrows:
            key = _arrow_key(ref_a)
            if key in pool:
                pool.remove(key)
                arrows_fully_correct += 1

        # Equivalence-aware exact match, broken down by partial-order shape
        shape = classify_shape(ref_arrows)
        shape_total[shape] += 1
        if sample_all_correct:
            shape_seq[shape] += 1
        if ea_em_match(pred_arrows, ref_arrows):
            ea_exact_match += 1
            shape_ea[shape] += 1

        # Chain continuity
        for i in range(len(pred_arrows) - 1):
            total_chains += 1
            if entities_match(pred_arrows[i]['end'], pred_arrows[i + 1]['start']):
                chain_correct += 1

    total_arrows = total_ref_arrows if total_ref_arrows > 0 else 1
    total_chain_denom = total_chains if total_chains > 0 else 1
    pred_denom = total_pred_arrows if total_pred_arrows > 0 else 1
    arrow_n_denom = arrow_n if arrow_n > 0 else 1
    arrow_recall = arrows_fully_correct / total_arrows
    arrow_precision = arrows_fully_correct / pred_denom
    arrow_f1 = (2 * arrow_precision * arrow_recall / (arrow_precision + arrow_recall)
                if (arrow_precision + arrow_recall) > 0 else 0.0)
    pos_recall = arrows_correct_positional / total_arrows
    pos_precision = arrows_correct_positional / pred_denom
    pos_f1 = (2 * pos_precision * pos_recall / (pos_precision + pos_recall)
              if (pos_precision + pos_recall) > 0 else 0.0)

    bt_atoms_recall = bt_atoms_arrow_correct / total_arrows
    bt_atoms_precision = bt_atoms_arrow_correct / pred_denom
    bt_atoms_f1 = (2 * bt_atoms_precision * bt_atoms_recall / (bt_atoms_precision + bt_atoms_recall)
                   if (bt_atoms_precision + bt_atoms_recall) > 0 else 0.0)

    rejection_bonus = error_correct if error_total > 0 else 0
    metrics = {
        "exact_match": exact_match / n,
        # Seq-EM: the stored serialization must be reproduced exactly
        "exact_match_normalized": (exact_match_normalized + rejection_bonus) / n,
        # EA-EM: any linear extension of the reference partial order
        "ea_exact_match": (ea_exact_match + rejection_bonus) / n,
        "all_arrows_acc": all_arrows_correct / arrow_n_denom,
        "arrow_count_acc": arrow_count_correct / arrow_n_denom,
        "arrow_precision": arrow_precision,
        "arrow_recall": arrow_recall,
        "arrow_f1": arrow_f1,
        "arrow_f1_positional": pos_f1,
        "chain_continuity": chain_correct / total_chain_denom,
        "start_entity_acc": start_correct / total_arrows,
        "end_entity_acc": end_correct / total_arrows,
        "start_atoms_acc": start_atoms_correct / total_arrows,
        "end_atoms_acc": end_atoms_correct / total_arrows,
        "start_bondtype_atoms_acc": start_bt_atoms_correct / total_arrows,
        "end_bondtype_atoms_acc": end_bt_atoms_correct / total_arrows,
        "bondtype_atoms_arrow_precision": bt_atoms_precision,
        "bondtype_atoms_arrow_recall": bt_atoms_recall,
        "bondtype_atoms_arrow_f1": bt_atoms_f1,
    }
    if error_total > 0:
        metrics["rejection_acc"] = error_correct / error_total
        metrics["rejection_total"] = error_total
    metrics["arrow_samples"] = arrow_n
    metrics["total_samples"] = n
    # Per-shape breakdown: EA-EM and Seq-EM coincide on total orders, and the
    # gap between them is confined to cyclic and branching/disconnected cases.
    metrics["shape_breakdown"] = {
        s: {"n": shape_total[s], "ea_em": shape_ea[s], "seq_em": shape_seq[s]}
        for s in SHAPES if shape_total[s] > 0
    }
    return metrics


def main():
    args = parse_args()

    # Load eval data
    with open(args.eval_json) as f:
        eval_data = json.load(f)

    if args.num_samples and args.num_samples < len(eval_data):
        import random
        random.seed(args.seed)
        eval_data = random.sample(eval_data, args.num_samples)

    print(f"Evaluating {len(eval_data)} samples", flush=True)

    # Load model
    if args.base_model:
        base_model_path = args.base_model
        print(f"Base model (no LoRA): {base_model_path}", flush=True)
    else:
        assert args.checkpoint, "Must provide --checkpoint or --base_model"
        adapter_config_path = os.path.join(args.checkpoint, "adapter_config.json")
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        base_model_path = adapter_config["base_model_name_or_path"]
        print(f"Base model: {base_model_path}", flush=True)
        print(f"LoRA checkpoint: {args.checkpoint}", flush=True)

    print("Loading base model...", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda:0")

    if args.checkpoint:
        print("Loading LoRA adapter...", flush=True)
        model = PeftModel.from_pretrained(model, args.checkpoint)
    model.eval()

    processor = AutoProcessor.from_pretrained(base_model_path)

    predictions = []
    references = []
    records = []

    for i, sample in enumerate(eval_data):
        human_msg = sample["conversations"][0]["value"]
        reference = sample["conversations"][1]["value"]
        image_file = sample["image"]
        image_path = os.path.join(args.image_folder, image_file)

        prompt_text = human_msg.replace("<image>\n", "").replace("<image>", "")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda:0")

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_len:]
        prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        predictions.append(prediction)
        references.append(reference)
        records.append({
            "id": sample.get("id"),
            "image": image_file,
            "reference": reference,
            "prediction": prediction,
        })

        # Determine match type
        is_error_ref = reference.strip().startswith("Error:")
        strict_match = prediction == reference
        # For arrow samples, check normalized match
        if not is_error_ref:
            pred_arrows = _parse_arrows(prediction)
            ref_arrows = _parse_arrows(reference)
            norm_match = True
            if len(pred_arrows) != len(ref_arrows):
                norm_match = False
            else:
                for pa, ra in zip(pred_arrows, ref_arrows):
                    if not entities_match(pa['start'], ra['start']) or not entities_match(pa['end'], ra['end']):
                        norm_match = False
                        break
        else:
            norm_match = prediction.strip().startswith("Error:")

        if strict_match:
            status = "✓ EXACT"
        elif norm_match:
            status = "✓ NORM"
        else:
            status = "✗"

        sample_type = "[ERROR]" if is_error_ref else "[ARROW]"
        print(f"\n{'─' * 100}", flush=True)
        print(f"{sample_type} [{i+1}/{len(eval_data)}] Image: {image_file}  -> {status}", flush=True)
        print(f"  Reference:  {reference}", flush=True)
        print(f"  Prediction: {prediction}", flush=True)

    # Final metrics
    metrics = compute_metrics(predictions, references)
    print("\n" + "=" * 80, flush=True)
    print("  FINAL METRICS (bond-order-invariant)", flush=True)
    print("=" * 80, flush=True)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}", flush=True)
        else:
            print(f"  {k:30s}: {v}", flush=True)
    print("=" * 80, flush=True)

    if args.output_json:
        output_dir = os.path.dirname(os.path.abspath(args.output_json))
        os.makedirs(output_dir, exist_ok=True)
        result = {
            "checkpoint": args.checkpoint,
            "base_model": base_model_path,
            "eval_json": args.eval_json,
            "image_folder": args.image_folder,
            "max_new_tokens": args.max_new_tokens,
            "num_samples": len(records),
            "seed": args.seed,
            "metrics": metrics,
            "records": records,
        }
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
