# MechParser

MechParser converts cropped chemical mechanism diagrams containing curved
electron-pushing arrows into ordered arrow operations and cumulative SM-SMARTS.
This repository consolidates synthetic data generation, Qwen3-VL training,
SAM3 + OCSR inference, SM-SMARTS parsing, and model evaluation.

## Scope

The implemented end-to-end path is:

```text
cropped mechanism diagram
  -> SAM3 arrow removal
  -> MolNexTR OCSR
  -> atom-mapped SMARTS and indexed RDKit reference
  -> left-right visual prompt
  -> Qwen3-VL arrow prediction
  -> cumulative SM-SMARTS
```

Raw-PDF figure extraction, cross-figure mechanisms, and catalytic-cycle layout
reconstruction are outside the current pipeline.

## Repository Layout

```text
mechparser/                 End-to-end inference and SM-SMARTS parser
src/                        Qwen3-VL SFT/GRPO training implementation
scripts/                    Data generation, training, inference, and evaluation
configs/                    DeepSpeed and path configuration templates
examples/                   20-diagram example subset with expected metrics
tests/                      Lightweight parser regression tests
```

## Installation

Python 3.11, Linux, and an NVIDIA GPU are recommended. CUDA 12.8 matches the
provided PyTorch wheel index.

```bash
git clone <ANONYMOUS_REPOSITORY_URL> mechparser
cd mechparser
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python scripts/validate_install.py
pytest -q
```

FlashAttention 2 is optional. The training and inference code falls back to
SDPA when it is unavailable:

```bash
python -m pip install flash-attn --no-build-isolation
```

## Configuration

Copy the path template and edit it for the local machine or cluster:

```bash
cp configs/paths.env.example .env
source .env
```

All model, data, output, partition, and GPU-resource paths can be overridden by
`MECHPARSER_*` environment variables. No script depends on any absolute path
from the environment the experiments were run in.

The default data layout is:

```text
data/
  stage1/
    arrow_train.json
    arrow_eval.json
    images/
  stage2/
    arrow_train.json
    arrow_eval.json
    images/
  real/
    mixed_train.json
    grpo_train.json
    real_eval.json
    images/
```

Manifests follow the LLaVA conversation schema and may contain relative image
paths resolved against the corresponding image folder, or absolute image paths.

## Synthetic Data Generation

The source CSV must contain `SMILES` and `pubchem_cid` columns.

Generate Stage 1 single-arrow data:

```bash
python scripts/generate_single_arrow.py \
  --csv /path/to/pubchem.csv \
  --output-dir "$MECHPARSER_DATA_ROOT/stage1" \
  --num-train 300000 \
  --num-eval 10000 \
  --seed 42
```

Generate Stage 2 multi-arrow data while excluding Stage 1 molecules:

```bash
python scripts/generate_multi_arrow.py \
  --csv /path/to/pubchem.csv \
  --output-dir "$MECHPARSER_DATA_ROOT/stage2" \
  --exclude-manifest "$MECHPARSER_DATA_ROOT/stage1/arrow_train.json" \
  --exclude-manifest "$MECHPARSER_DATA_ROOT/stage1/arrow_eval.json" \
  --num-train 200000 \
  --num-eval 500 \
  --workers 16 \
  --seed 42
```

Both commands write `generation_metadata.json` containing source, seed, requested
counts, and generated counts. Generation failures are skipped, so generated
counts may be below requested counts.

## Four-Stage Training

The consolidated defaults reproduce the historical training commands:

| Stage | Data | Method | Key defaults |
|---|---|---|---|
| 1 | Single-arrow synthetic | LoRA SFT | 2 epochs, global batch 120, LR 5e-5 |
| 2 | Multi-arrow synthetic | Vision + language LoRA SFT | 2 epochs, global batch 64, LR 2e-5 |
| 3a | 3K synthetic + real warmup | Mixed LoRA SFT | 8 epochs, global batch 16, LR 5e-6 |
| 3b | Disjoint real subset | GRPO | 8 epochs, 6 generations, LR 1e-6 |

The historical Stage 1 log printed a nominal batch of 128, but six GPUs x batch
4 x gradient accumulation 5 yields an actual global batch of 120. This repository
uses 120 explicitly.

Run one stage interactively or inside an allocation:

```bash
scripts/train_stage1.sh
scripts/merge_stage.sh \
  "$MECHPARSER_OUTPUT_ROOT/adapters/stage1" \
  "$MECHPARSER_BASE_MODEL" \
  "$MECHPARSER_OUTPUT_ROOT/merged/stage1"

scripts/train_stage2.sh
scripts/train_stage3a.sh
scripts/train_stage3b.sh
```

Submit the complete dependency chain on Slurm:

```bash
source .env
scripts/submit_four_stage.sh
```

The driver submits:

```text
Stage 1 -> merge -> Stage 2 -> merge -> Stage 3a -> merge -> Stage 3b -> final merge
```

GPU resources are passed explicitly by `submit_four_stage.sh`; Stage 3a no longer
contains the malformed combined `--mem/--gres` directive from the legacy script.
Training-time evaluation is disabled by default so final test data cannot affect
checkpoint selection. Set `MECHPARSER_ENABLE_TRAIN_EVAL=1` only with a separate
validation split.

## End-to-End Inference

Run a merged model or a PEFT adapter on one cropped mechanism diagram:

```bash
python -m mechparser input.png \
  --output outputs/example \
  --llm-model /path/to/merged-model-or-adapter \
  --sam-device cuda \
  --ocsr-device cpu \
  --llm-device cuda
```

Use local weights instead of automatic downloads:

```bash
python -m mechparser input.png \
  --output outputs/example \
  --llm-model /path/to/qwen-or-adapter \
  --sam-checkpoint /path/to/sam3.pt \
  --ocsr-checkpoint /path/to/molnextr.pth
```

Passing a directory processes all supported images and writes `summary.json`.

## Bidirectional SM-SMARTS Parser

Encode an atom-mapped scaffold and ordered arrows:

```bash
python -m mechparser.smsmarts \
  --smarts '[O-:1]-[C:2]' \
  --arrows '[Arrow 1: Start: atom [O-:1] -> End: Single Bond between atoms [O-:1] and [C:2]]'
```

Decode cumulative steps:

```bash
python -m mechparser.smsmarts --decode \
  --smarts '[O-:1]-[C:2]' \
  --steps-json '{"A":"[aO-:1];[C:2]"}'
```

The base SMARTS is required for decoding because legacy SM-SMARTS annotations
replace native bond symbols. For example, `_` records that a bond became single
but does not encode whether its original type was double or aromatic. Supplying
the OCSR scaffold removes that ambiguity. The decoder validates arrow-type-specific
conservation signatures, including concerted bond-to-bond and bond-to-new-bond
transfers.

## Evaluation

Evaluate a PEFT adapter:

```bash
python scripts/evaluate_arrows.py \
  --checkpoint /path/to/adapter \
  --eval_json /path/to/eval.json \
  --image_folder /path/to/images \
  --max_new_tokens 512 \
  --output_json outputs/eval_adapter.json
```

Evaluate a merged/base model:

```bash
python scripts/evaluate_arrows.py \
  --base_model /path/to/model \
  --eval_json /path/to/eval.json \
  --image_folder /path/to/images \
  --output_json outputs/eval_model.json
```

The evaluator reports two sample-level exact-match numbers. **EA-EM**
(`ea_exact_match`) accepts any linear extension of the partial order that
electron-flow chaining induces over the reference arrows, with cyclic components
matched modulo rotation; arrow direction, identity and count are never relaxed.
**Seq-EM** (`exact_match_normalized`) additionally requires the stored
serialization and is therefore strictly stronger. A `shape_breakdown` field
splits both by partial-order shape (total order, cycle, branching/disconnected).

Arrow precision/recall/F1 use order-free matching, so a cyclic mechanism reported
from a different starting arrow is not penalised at the arrow level;
`arrow_f1_positional` retains the positional variant as a diagnostic. The
evaluator also reports strict exact match, arrow-count accuracy, entity accuracy,
atom accuracy, and bond-type-plus-atom metrics.

### Example subset

`examples/` contains 20 real mechanism diagrams with their expert annotations
(`ground_truth.json`), the model output (`predictions.json`), and the metrics the
evaluator produces on them (`metrics.json`), so the output format and the metric
definitions can be checked without the full corpus. See
[examples/README.md](examples/README.md).

## Tests

```bash
pytest -q
python scripts/validate_install.py
bash -n scripts/*.sh
```

The parser tests cover atom-to-bond, bond-to-atom, bond-to-bond, atom-to-new-bond,
bond-to-new-bond, and invalid unpaired edits. Full GPU inference requires the
external model weights and is intentionally not part of CPU unit tests.

## License

The MechParser code and the consolidated Qwen-VL training code are released
under Apache License 2.0. External models and libraries retain their respective
licenses.
