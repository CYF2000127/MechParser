# Example subset

A 20-diagram illustrative subset of the real mechanism data, provided so that
the output format and the evaluation metrics can be inspected without access to
the full corpus.

```text
images/             20 mechanism diagrams, renamed 1.png ... 20.png
ground_truth.json   expert arrow annotations, keyed by image
predictions.json    MechParser-VL output for the same 20 diagrams
metrics.json        expected output of scripts/evaluate_arrows.py on this subset
```

`ground_truth.json` and `predictions.json` share the `id` / `image` keys, so the
two can be joined without any bookkeeping on the reader's side.

## Reproducing the numbers

```bash
python scripts/evaluate_arrows.py \
  --eval_json examples/ground_truth.json \
  --image_folder examples/images \
  --checkpoint /path/to/adapter \
  --output_json outputs/example_eval.json
```

The metrics in `metrics.json` are computed directly from the stored
`predictions.json`, so they can also be checked without a GPU by feeding the two
files to `compute_metrics()` in `scripts/evaluate_arrows.py`.

## Metrics on this subset

| Metric | Value |
|---|---:|
| Arrow precision / recall / F1 (order-free) | 90.0 / 90.0 / **90.0** |
| Arrow F1 (positional, diagnostic only) | 86.0 |
| EA-EM | **16/20 = 80.0%** |
| Seq-EM | 15/20 = 75.0% |
| Reference arrows | 50 |

Partial-order shapes present: 17 total-order, 3 cyclic, 0 branching.

## Why two exact-match numbers

Arrow order in a mechanism is only a *partial* order: arrows that do not feed
one another are unordered, and a closed cycle has no chemically determined first
arrow. **EA-EM** accepts any linear extension of that partial order, with cyclic
components matched modulo rotation; arrow direction, identity and count are never
relaxed. **Seq-EM** additionally requires the single stored serialization and is
therefore strictly stronger.

Sample 5 is the case that separates them: the prediction traverses the same
directed cycle starting from the other arrow. It is chemically identical to the
reference, so EA-EM accepts it and Seq-EM does not. For the same reason
arrow-level matching is order-free — a rotated cycle should not be scored as two
wrong arrows once the mechanism as a whole has been accepted.

## Scope

This subset is illustrative, not a benchmark. It is not the Real-Eval test set
and the numbers above are not comparable to the results reported in the paper;
they exist so that the output format, the parser and the metric definitions can
be checked end to end.
