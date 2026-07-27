import os
import torch
from peft import LoraConfig, get_peft_model
import ast
from transformers import (
    AutoProcessor,
    AutoConfig,
    BitsAndBytesConfig, 
    Qwen2VLForConditionalGeneration, 
    HfArgumentParser, 
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)
from src.trainer import QwenSFTTrainer, GenerativeEvalPrediction
from src.dataset import make_supervised_data_module
import re
from src.params import DataArguments, ModelArguments, TrainingArguments
from train.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer
import pathlib
from monkey_patch_forward import (
    replace_qwen3_with_mixed_modality_forward,
    replace_qwen2_5_with_mixed_modality_forward, 
    replace_qwen_2_with_mixed_modality_forward,
    replace_qwen3_vl_moe_with_mixed_modality_forward
)
from monkey_patch_vision import replace_qwen2_5_vision

local_rank = None

def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names

def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad

def configure_vision_tower(model, training_args, compute_dtype, device):
    vision_tower = model.visual
    vision_tower.to(dtype=compute_dtype, device=device)

    vision_model_params = model.visual.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)
    
    # Handle merger specifically
    merger_params = model.visual.merger.parameters()
    set_requires_grad(merger_params, not training_args.freeze_merger)

    if hasattr(model.visual, "deepstack_merger_list"):
        deepstack_merger_list_params = model.visual.deepstack_merger_list.parameters()
        set_requires_grad(deepstack_merger_list_params, not training_args.freeze_merger)

def configure_llm(model, training_args):
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_llm)

    llm_params = model.language_model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)

def unfreeze_topk_layers(model, k_llm: int = 0, k_vis: int = 0):
    if k_llm and hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        for layer in model.language_model.layers[-k_llm:]:
            for p in layer.parameters():
                p.requires_grad = True

    if k_vis and hasattr(model, "visual") and hasattr(model.visual, "blocks"):
        for blk in model.visual.blocks[-k_vis:]:
            for p in blk.parameters():
                p.requires_grad = True


def _parse_arrows(text):
    """Parse arrow predictions/references into structured list.
    Returns list of dicts: [{'start': str, 'end': str}, ...]
    """
    arrows = []
    # Split by "Arrow N:" pattern
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


def compute_arrow_metrics(eval_pred: GenerativeEvalPrediction):
    predictions = eval_pred.predictions
    references = eval_pred.references
    n = len(predictions)
    if n == 0:
        return {}

    exact_match = 0
    arrow_count_correct = 0
    start_correct = 0
    end_correct = 0
    start_atoms_correct = 0
    end_atoms_correct = 0
    total_ref_arrows = 0
    total_pred_arrows = 0
    arrows_fully_correct = 0  # per arrow: both Start and End correct
    all_arrows_correct = 0    # every arrow in the sample correct
    chain_correct = 0         # chaining: End of arrow N == Start of arrow N+1
    total_chains = 0

    # Separate tracking for Error (rejection) vs normal (arrow) samples
    error_total = 0
    error_correct = 0
    arrow_n = 0  # number of normal (non-error) samples

    for pred_text, ref_text in zip(predictions, references):
        pred_text = pred_text.strip()
        ref_text = ref_text.strip()

        is_error_sample = ref_text.startswith("Error:")

        # Exact match (case-insensitive)
        if pred_text.lower() == ref_text.lower():
            exact_match += 1

        # Track error/rejection samples separately
        if is_error_sample:
            error_total += 1
            if pred_text.strip().startswith("Error:"):
                error_correct += 1
            continue  # skip arrow-level metrics for error samples

        arrow_n += 1

        pred_arrows = _parse_arrows(pred_text)
        ref_arrows = _parse_arrows(ref_text)
        total_pred_arrows += len(pred_arrows)

        # Arrow count accuracy
        if len(pred_arrows) == len(ref_arrows):
            arrow_count_correct += 1

        # Per-arrow matching (order-sensitive, matching by index)
        sample_all_correct = True
        for i, ref_a in enumerate(ref_arrows):
            total_ref_arrows += 1
            if i >= len(pred_arrows):
                sample_all_correct = False
                continue
            pred_a = pred_arrows[i]

            s_ok = pred_a['start'].lower() == ref_a['start'].lower()
            e_ok = pred_a['end'].lower() == ref_a['end'].lower()

            if s_ok:
                start_correct += 1
            else:
                sample_all_correct = False
            if e_ok:
                end_correct += 1
            else:
                sample_all_correct = False

            if s_ok and e_ok:
                arrows_fully_correct += 1

            # Atom-level match (partial credit)
            pred_start_atoms = _extract_atom_ids(pred_a['start'])
            ref_start_atoms = _extract_atom_ids(ref_a['start'])
            if ref_start_atoms and pred_start_atoms == ref_start_atoms:
                start_atoms_correct += 1

            pred_end_atoms = _extract_atom_ids(pred_a['end'])
            ref_end_atoms = _extract_atom_ids(ref_a['end'])
            if ref_end_atoms and pred_end_atoms == ref_end_atoms:
                end_atoms_correct += 1

        if len(pred_arrows) != len(ref_arrows):
            sample_all_correct = False
        if sample_all_correct:
            all_arrows_correct += 1

        # Chain continuity: check if pred Arrow N end == pred Arrow N+1 start
        for i in range(len(pred_arrows) - 1):
            total_chains += 1
            if pred_arrows[i]['end'].lower() == pred_arrows[i+1]['start'].lower():
                chain_correct += 1

    total_arrows = total_ref_arrows if total_ref_arrows > 0 else 1
    total_chain_denom = total_chains if total_chains > 0 else 1
    pred_denom = total_pred_arrows if total_pred_arrows > 0 else 1
    arrow_recall = arrows_fully_correct / total_arrows
    arrow_precision = arrows_fully_correct / pred_denom
    arrow_f1 = (2 * arrow_precision * arrow_recall / (arrow_precision + arrow_recall)
                if (arrow_precision + arrow_recall) > 0 else 0.0)
    arrow_n_denom = arrow_n if arrow_n > 0 else 1
    metrics = {
        "exact_match": exact_match / n,
        "all_arrows_acc": all_arrows_correct / arrow_n_denom,
        "arrow_count_acc": arrow_count_correct / arrow_n_denom,
        "arrow_precision": arrow_precision,
        "arrow_recall": arrow_recall,
        "arrow_f1": arrow_f1,
        "chain_continuity": chain_correct / total_chain_denom,
        "start_entity_acc": start_correct / total_arrows,
        "end_entity_acc": end_correct / total_arrows,
        "start_atoms_acc": start_atoms_correct / total_arrows,
        "end_atoms_acc": end_atoms_correct / total_arrows,
    }
    if error_total > 0:
        metrics["rejection_acc"] = error_correct / error_total
        metrics["rejection_total"] = error_total
    return metrics


def train():
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    if data_args.nframes is not None and data_args.fps is not None:
        raise ValueError("You cannot set both `nframes` and `fps` at the same time. Please set only one of them.")

    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("If `lora_enable` is True, `freeze_llm` must also be True.")

    if not training_args.lora_enable:
        assert not training_args.vision_lora, \
            "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."
        
    if training_args.vision_lora and not training_args.freeze_vision_tower:
        raise ValueError("If `vision_lora` is True, `freeze_vision_tower` must also be True.")

    else:
        if training_args.lora_namespan_exclude is not None:
            training_args.lora_namespan_exclude = ast.literal_eval(training_args.lora_namespan_exclude)
        else:
            training_args.lora_namespan_exclude = []

        if not training_args.vision_lora:
            training_args.lora_namespan_exclude += ["visual"]

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4,8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"":training_args.device},
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=training_args.bits==4,
                load_in_8bit=training_args.bits==8,
                llm_int8_skip_modules=["visual", "lm_head"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            )
        ))

    config = AutoConfig.from_pretrained(model_args.model_id)

    if config.model_type == "qwen3_vl_moe":
        replace_qwen3_vl_moe_with_mixed_modality_forward()
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_id,
            dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
            **bnb_model_from_pretrained_args
        )

    elif config.model_type == "qwen3_vl":
        replace_qwen3_with_mixed_modality_forward()
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_id,
            dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
            **bnb_model_from_pretrained_args
        )

    elif config.model_type == "qwen2_5_vl":
        replace_qwen2_5_with_mixed_modality_forward()
        replace_qwen2_5_vision()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_id,
            dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa", 
            **bnb_model_from_pretrained_args
        )
        
    else:
        replace_qwen_2_with_mixed_modality_forward()
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_id,
            dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa", 
            **bnb_model_from_pretrained_args
        )

    model.config.use_cache = False
    model_to_configure = model
    configure_llm(model_to_configure, training_args)
    configure_vision_tower(model_to_configure, training_args, compute_dtype, training_args.device)

    unfreeze_topk_layers(
        model_to_configure,
        k_llm=getattr(training_args, "unfreeze_topk_llm", 0),
        k_vis=getattr(training_args, "unfreeze_topk_vision", 0),
    )

    if training_args.gradient_checkpointing:
        if training_args.vision_lora:
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        else:
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
        
        model.enable_input_require_grads()

    if training_args.bits in [4,8]:
        model.config.dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing, gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs)
    
    if training_args.lora_enable:
        lora_namespan_exclude = training_args.lora_namespan_exclude
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(model, lora_namespan_exclude=lora_namespan_exclude, num_lora_modules=training_args.num_lora_modules),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA to the model...")
        model = get_peft_model(model, peft_config)

        # Peft maodel makes vision tower and merger freezed again.
        # Configuring fuction could be called here, but sometimes it does not work properly.
        # So I just made it this way.
        # Need to be fixed in the future.

        if not training_args.freeze_vision_tower:
            for name, param in model.named_parameters():
                if "visual" in name:
                    param.requires_grad = True

        if not training_args.freeze_merger:
            for name, param in model.named_parameters():
                if "merger" in name:
                    param.requires_grad = True

    processor = AutoProcessor.from_pretrained(model_args.model_id)

    # model.config.tokenizer_model_max_length = processor.tokenizer.model_max_length

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            
            if 'lm_head' in name or 'embed_token' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(model_id=model_args.model_id,
                                              processor=processor,
                                              data_args=data_args)

    trainer = QwenSFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        compute_metrics=compute_arrow_metrics,
        **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True
    
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )

        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=True
        )

        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            processor.save_pretrained(training_args.output_dir)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)



if __name__ == "__main__":
    train()
