"""
End-to-end pipeline：原始图像 → 移除箭头 → 预测分子 → 生成SMARTS → 拼接高清图 → LLM预测箭头
输入：原始带箭头的分子图像
输出：3x 高清拼接图（左=原图放大，右=RDKit 矢量重绘 atom indices）+ final SMARTS + 箭头预测
"""

import os, json, re, cv2, torch, numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple
from PIL import Image
from io import BytesIO

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from molnextr import MolNexTR
from huggingface_hub import hf_hub_download

from .smsmarts import arrows_to_steps, format_simple as format_steps

SCALE = 3


# ────────────────────────────────────────────────────────────
# 1. 模型加载（全局单例）
# ────────────────────────────────────────────────────────────
_sam3_model = None
_sam3_processor = None
_molnextr_model = None
_llm_model = None
_llm_processor = None


def _load_sam3(device="cuda", checkpoint_path=None):
    global _sam3_model, _sam3_processor
    if _sam3_model is None:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        print("[Pipeline] 加载 SAM3 模型...")
        _sam3_model = build_sam3_image_model(
            device=device,
            checkpoint_path=checkpoint_path,
            load_from_HF=checkpoint_path is None,
        )
        _sam3_processor = Sam3Processor(_sam3_model, device=device)
        print("[Pipeline] SAM3 模型加载完成")
    return _sam3_processor


def _load_molnextr(device="cpu", checkpoint_path=None,
                   repo_id="CYF200127/ChemEAGLEModel",
                   filename="molnextr.pth"):
    global _molnextr_model
    if _molnextr_model is None:
        print("[Pipeline] 加载 MolNexTR 模型...")
        ckpt = checkpoint_path or hf_hub_download(repo_id, filename)
        _molnextr_model = MolNexTR(ckpt, device=torch.device(device))
        print("[Pipeline] MolNexTR 模型加载完成")
    return _molnextr_model


def _load_llm(model_path: str, device="cuda"):
    """加载 Qwen3-VL 箭头预测模型（全局单例）。
    支持两种模式：
      - merged 模型目录（直接加载）
      - LoRA checkpoint 目录（含 adapter_config.json，自动加载 base + adapter）
    """
    global _llm_model, _llm_processor
    if _llm_model is None:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        print(f"[Pipeline] 加载 LLM 箭头预测模型: {model_path}")
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        try:
            from flash_attn import flash_attn_func  # noqa: F401
            _attn_impl = "flash_attention_2"
        except ImportError:
            _attn_impl = "sdpa"
        print(f"[Pipeline]   Attention: {_attn_impl}")

        if os.path.exists(adapter_config_path):
            from peft import PeftModel
            with open(adapter_config_path) as f:
                adapter_config = json.load(f)
            base_model_path = adapter_config["base_model_name_or_path"]
            print(f"[Pipeline]   LoRA 模式, Base model: {base_model_path}")
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=_attn_impl,
            ).to(device)
            model = PeftModel.from_pretrained(model, model_path)
            _llm_processor = AutoProcessor.from_pretrained(base_model_path)
        else:
            print(f"[Pipeline]   Merged 模式, 直接加载")
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=_attn_impl,
            ).to(device)
            _llm_processor = AutoProcessor.from_pretrained(model_path)
        model.eval()
        _llm_model = model
        print("[Pipeline] LLM 模型加载完成")
    return _llm_model, _llm_processor


# ────────────────────────────────────────────────────────────
# 2. 箭头移除
# ────────────────────────────────────────────────────────────
def remove_arrows(image_rgb: np.ndarray, device="cuda",
                  checkpoint_path=None) -> np.ndarray:
    """用 SAM3 检测并移除弯曲箭头，返回去箭头的 RGB 图。"""
    processor = _load_sam3(device, checkpoint_path)
    pil_img = Image.fromarray(image_rgb)
    img_h, img_w = image_rgb.shape[:2]

    inference_state = processor.set_image(pil_img)
    output = processor.set_text_prompt(state=inference_state, prompt="curved arrow")
    masks = output["masks"]

    combined_mask = np.zeros((img_h, img_w), dtype=bool)
    for mask in masks:
        m = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else np.array(mask)
        while m.ndim > 2:
            m = m.squeeze(0) if m.shape[0] == 1 else (m.squeeze(-1) if m.shape[-1] == 1 else m[0])
        if m.ndim != 2:
            continue
        if m.shape[0] != img_h or m.shape[1] != img_w:
            pil_m = Image.fromarray((m * 255).astype(np.uint8) if m.max() <= 1 else m.astype(np.uint8))
            m = np.array(pil_m.resize((img_w, img_h), Image.NEAREST)) > 128
        combined_mask |= (m > 0)

    result = image_rgb.copy()
    result[combined_mask] = [255, 255, 255]
    return result


# ────────────────────────────────────────────────────────────
# 3. LLM 箭头预测
# ────────────────────────────────────────────────────────────
ARROW_PROMPT = (
    "Task: Analyze the provided chemical image.\n"
    "- The left panel shows a molecule with curved arrows indicating sequential electron flow. "
    "If the left panel contains no curved arrows or lacks complete arrows, strictly output: "
    "\"Error: No suitable curved arrows found in the source image.\"\n"
    "- The right panel shows the same molecule with atom indices for reference.\n\n"
    "Your goal is to identify the Start and End points of each curved arrow in the left panel "
    "by matching them with the atom indices in the right panel. The arrows are numbered in the "
    "order of electron flow (the end of one arrow connects to the start of the next).\n\n"
    "Entity naming rules:\n"
    "- For an atom, use: atom [Element:ID]. Include formal charges if present "
    "(e.g., atom [N+:5], atom [O-:3]). If an entity is present in the left panel but missing "
    "from the right panel, use \"unknown\" as its ID (e.g., atom [-O2CCF3:unknown]).\n"
    "- For a bond, use: <Type> Bond between atoms [Element:ID] and [Element:ID] "
    "(Types: Single, Double, Triple, Aromatic)\n\n"
    "Strictly output your answer in the following format, with no extra text or explanations:\n"
    "[Arrow 1: Start: <Start_Entity> -> End: <End_Entity>; "
    "Arrow 2: Start: <Start_Entity> -> End: <End_Entity>; ...]\n\n"
    "Example of expected output:\n"
    "[Arrow 1: Start: atom [O-:3] -> End: atom [C:4]; "
    "Arrow 2: Start: Double Bond between atoms [C:4] and [C:5] -> End: atom [N+:6]]"
)


def predict_arrows(image_path: str, llm_model_path: str,
                   device: str = "cuda",
                   max_new_tokens: int = 512) -> str:
    """用 Qwen3-VL 模型预测拼接图中的弯曲箭头。"""
    from qwen_vl_utils import process_vision_info

    model, processor = _load_llm(llm_model_path, device=device)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": ARROW_PROMPT},
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
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]
    prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return prediction


# ────────────────────────────────────────────────────────────
# 4. SMARTS 生成辅助
# ────────────────────────────────────────────────────────────
def _apply_patch(pred: Dict):
    for key in ('smiles', 'molfile'):
        if key in pred:
            pred[key] = pred[key].replace('R7', 'R1').replace('7*', '1*')
            pred[key] = pred[key].replace('Rf', 'R1').replace('f*', '1*')
    if 'symbols' in pred:
        pred['symbols'] = [s.replace('R7', 'R1').replace('7*', '1*').replace('Rf', 'R1').replace('f*', '1*')
                           for s in pred['symbols']]
    if 'atoms' in pred:
        for atom in pred['atoms']:
            if 'atom_symbol' in atom:
                atom['atom_symbol'] = atom['atom_symbol'].replace('R7', 'R1').replace('7*', '1*')
                atom['atom_symbol'] = atom['atom_symbol'].replace('Rf', 'R1').replace('f*', '1*')


def _fixed_order_smiles(molfile: str) -> str:
    mol = Chem.MolFromMolBlock(molfile.replace(' R ', ' * '), removeHs=False)
    if mol is None:
        return ""
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(i + 1)
    return Chem.MolToSmiles(mol, canonical=False, rootedAtAtom=0, allHsExplicit=True)


def _strip_labels(mapped_smi: str) -> str:
    mol = Chem.MolFromSmarts(mapped_smi)
    if not mol:
        return ""
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=False, rootedAtAtom=0, allBondsExplicit=False)


def _smarts_with_symbols(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return ""
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    smarts = Chem.MolToSmarts(mol)

    ATOMIC_NUM_TO_SYM = {
        "#53": "I", "#35": "Br", "#17": "Cl", "#16": "S",
        "#15": "P", "#14": "Si", "#13": "Al", "#12": "Mg",
        "#30": "Zn", "#26": "Fe", "#20": "Ca", "#19": "K",
        "#9": "F", "#8": "O", "#7": "N", "#6": "C",
        "#5": "B", "#1": "H", "#0": "*",
    }

    for atom in mol.GetAtoms():
        anum = atom.GetAtomicNum()
        mapn = atom.GetAtomMapNum()
        sym = "*" if anum == 0 else atom.GetSymbol()
        smarts = smarts.replace(f"[#{anum}:{mapn}]", f"[{sym}:{mapn}]")

    for num, sym in ATOMIC_NUM_TO_SYM.items():
        smarts = smarts.replace(num, sym)

    return smarts


def _simplify_smarts(smarts: str, symbols: list) -> str:
    def shift_labels(text):
        result, i = [], 0
        while i < len(text):
            if text[i] == '[':
                j = text.find(']', i)
                if j != -1:
                    inner = re.sub(r':(\d+)', lambda m: f":{int(m.group(1)) - 1}", text[i+1:j])
                    result.append(f'[{inner}]')
                    i = j + 1
                    continue
            result.append(text[i])
            i += 1
        return ''.join(result)

    smarts = shift_labels(smarts)

    def replace_wc(match):
        label = int(match.group(2))
        if label < len(symbols):
            sym = symbols[label]
            sym = sym[1:-1] if sym.startswith('[') and sym.endswith(']') else sym
            return f"[{sym}:{label}]"
        return match.group(0)

    return re.sub(r'\[(\d*)\*:(\d+)\]', replace_wc, smarts)


def _build_mol_from_predictions(pred_data: Dict):
    """从 predictions 的 symbols/edges 直接构建分子（不依赖 molfile）。"""
    symbols = pred_data.get('symbols', [])
    edges = pred_data.get('edges')
    if not symbols or edges is None:
        return None
    edges = np.array(edges, dtype=int)
    em = Chem.RWMol()
    for s in symbols:
        if s.startswith('[') and s.endswith(']'):
            inner = s[1:-1]
            if inner == '*' or (inner.endswith('*') and inner[:-1].replace(' ', '').isdigit()):
                a = Chem.Atom(0)
            elif inner.startswith('R'):
                a = Chem.Atom(0)
            else:
                core = ''.join(c for c in inner if c.isalpha())
                try:
                    a = Chem.Atom(core if core else '*')
                except Exception:
                    a = Chem.Atom(0)
            if '+' in s:
                a.SetFormalCharge(1)
            if '-' in s:
                a.SetFormalCharge(-1)
        elif s == '*' or (s.startswith('R') and (len(s) == 1 or s[1:].isdigit())):
            a = Chem.Atom(0)
        else:
            try:
                a = Chem.Atom(s)
            except Exception:
                a = Chem.Atom(0)
        em.AddAtom(a)
    n = len(symbols)
    bt_map = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
              3: Chem.BondType.TRIPLE, 4: Chem.BondType.AROMATIC}
    for i in range(n):
        for j in range(i + 1, n):
            bo = int(edges[i][j])
            if bo > 0:
                em.AddBond(i, j, bt_map.get(bo, Chem.BondType.SINGLE))
    try:
        return em.GetMol()
    except Exception:
        return None


def generate_smarts(predictions: Dict) -> str:
    """从 MolScribe predictions 生成 final SMARTS。"""
    _apply_patch(predictions)
    molfile = predictions.get('molfile', '')
    symbols = predictions.get('symbols', [])

    smi_fixed = _fixed_order_smiles(molfile)
    if not smi_fixed:
        mol = _build_mol_from_predictions(predictions)
        if mol is None:
            return ""
        for i, atom in enumerate(mol.GetAtoms()):
            atom.SetAtomMapNum(i + 1)
        smi_fixed = Chem.MolToSmiles(mol, canonical=False, rootedAtAtom=0, allHsExplicit=True)
        if not smi_fixed:
            return ""
    smi_plain = _strip_labels(smi_fixed)
    if not smi_plain:
        return ""
    smarts = _smarts_with_symbols(smi_plain)
    if not smarts:
        return ""
    return _simplify_smarts(smarts, symbols)


def generate_step_smarts(input_smarts: str, arrows_str: str) -> Optional[Dict]:
    """从 input SMARTS + 标准箭头描述 → step SMARTS dict。

    返回:
        {"A": "step_smarts_A", "B": "step_smarts_B", ...} 或 None
    """
    if not input_smarts or not arrows_str:
        return None
    try:
        steps = arrows_to_steps(input_smarts, arrows_str)
        if not steps:
            return None
        return {label: sma for label, sma in steps}
    except Exception as e:
        print(f"      Step SMARTS 生成失败: {e}")
        return None


# ────────────────────────────────────────────────────────────
# 5. 高清右图绘制（RDKit 矢量重绘）
# ────────────────────────────────────────────────────────────
def _cropwhite_bbox(img_bgr):
    h, w, _ = img_bgr.shape
    x = (img_bgr != 255).any(axis=2)
    if not x.any():
        return 0, h, 0, w
    rows = x.any(axis=1)
    cols = x.any(axis=0)
    top = rows.argmax()
    bottom = h - rows[::-1].argmax()
    left = cols.argmax()
    right = w - cols[::-1].argmax()
    return int(top), int(bottom), int(left), int(right)


def _draw_molecule_hd(pred_data: Dict, molfile: str,
                      orig_bgr: np.ndarray, scale: int) -> Optional[np.ndarray]:
    """用 RDKit 矢量重绘右图（带 atom indices），返回 BGR 图或 None。"""
    mol = Chem.MolFromMolBlock(molfile.replace(' R ', ' * '), removeHs=False)

    if mol is None:
        mol = _build_mol_from_predictions(pred_data)
    if mol is None:
        return None

    for atom in mol.GetAtoms():
        try:
            alias = Chem.GetAtomAlias(atom)
            if alias:
                atom.SetProp('atomLabel', alias)
        except Exception:
            pass

    atom_sets = pred_data.get('atoms', [])
    bond_sets = pred_data.get('bonds', [])

    img_h, img_w = orig_bgr.shape[:2]
    top, bottom, left, right = _cropwhite_bbox(orig_bgr)
    crop_w, crop_h = right - left, bottom - top

    if bond_sets:
        rw = Chem.RWMol(mol)
        for b in list(rw.GetBonds()):
            rw.RemoveBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        chiral = {idx for idx, ai in enumerate(atom_sets) if '@' in ai.get('atom_symbol', '')}
        for b in bond_sets:
            if "endpoint_atoms" not in b:
                continue
            i_r, j_r = map(int, b["endpoint_atoms"])
            bt = b["bond_type"]
            begin, end, flipped = i_r, j_r, False
            if bt in ("solid wedge", "dashed wedge"):
                if j_r in chiral and i_r not in chiral:
                    begin, end, flipped = j_r, i_r, True
            order_map = {
                "single":      (Chem.rdchem.BondType.SINGLE,   Chem.rdchem.BondDir.NONE),
                "double":      (Chem.rdchem.BondType.DOUBLE,   Chem.rdchem.BondDir.NONE),
                "triple":      (Chem.rdchem.BondType.TRIPLE,   Chem.rdchem.BondDir.NONE),
                "aromatic":    (Chem.rdchem.BondType.AROMATIC,  Chem.rdchem.BondDir.NONE),
                "solid wedge": (Chem.rdchem.BondType.SINGLE,
                                Chem.rdchem.BondDir.BEGINDASH if flipped else Chem.rdchem.BondDir.BEGINWEDGE),
                "dashed wedge":(Chem.rdchem.BondType.SINGLE,
                                Chem.rdchem.BondDir.BEGINWEDGE if flipped else Chem.rdchem.BondDir.BEGINDASH),
            }
            if bt not in order_map:
                continue
            order, direction = order_map[bt]
            rw.AddBond(begin, end, order)
            rw.GetBondBetweenAtoms(begin, end).SetBondDir(direction)
        mol = rw.GetMol()

    for idx, ai in enumerate(atom_sets):
        sym = ai.get('atom_symbol', '')
        if '@' in sym and (sym.startswith('C') or sym.startswith('[C')):
            a = mol.GetAtomWithIdx(idx)
            a.SetNoImplicit(True)
            a.SetNumExplicitHs(0)

    if mol.GetNumConformers() == 0:
        mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()))
    conf = mol.GetConformer(0)

    PAD = 50
    W1, H1 = crop_w + 2 * PAD, crop_h + 2 * PAD
    coords_list = []
    for i, ai in enumerate(atom_sets):
        if i >= mol.GetNumAtoms():
            break
        if 'x' in ai and 'y' in ai:
            coords_list.append((ai['x'] * W1 - PAD, ai['y'] * H1 - PAD))

    if coords_list:
        xs, ys = zip(*coords_list)
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        sc = min(6.0 / max(x_max - x_min, 1.0), 6.0 / max(y_max - y_min, 1.0))
        xc, yc = (x_min + x_max) / 2, (y_min + y_max) / 2
        for i, (x, y) in enumerate(coords_list):
            conf.SetAtomPosition(i, ((x - xc) * sc + 5.0, 10.0 - ((y - yc) * sc + 5.0), 0))

    hd_cw, hd_ch = crop_w * scale, crop_h * scale
    drawer = rdMolDraw2D.MolDraw2DCairo(hd_cw, hd_ch)
    opts = drawer.drawOptions()
    opts.addStereoAnnotation = False
    opts.useBWAtomPalette()
    opts.fixedScale = 20.0 * scale
    opts.addAtomIndices = True
    try:
        opts.atomCircleScaling = True
    except AttributeError:
        pass
    try:
        opts.includeAtomCircles = True
    except AttributeError:
        pass

    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    img_cropped = Image.open(BytesIO(drawer.GetDrawingText()))
    img_final = Image.new('RGB', (img_w * scale, img_h * scale), 'white')
    img_final.paste(img_cropped, (left * scale, top * scale))
    return np.array(img_final)[:, :, ::-1]  # BGR


# ────────────────────────────────────────────────────────────
# 6. 拼接
# ────────────────────────────────────────────────────────────
def _concat_lr(left_bgr: np.ndarray, right_bgr: np.ndarray, scale: int) -> np.ndarray:
    """左右拼接，对齐高度，返回 RGB numpy。"""
    left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)

    h1, w1 = left_rgb.shape[:2]
    h2, w2 = right_rgb.shape[:2]
    if h1 != h2:
        tgt = max(h1, h2)
        left_rgb = cv2.resize(left_rgb, (int(round(w1 * tgt / h1)), tgt), interpolation=cv2.INTER_LANCZOS4)
        right_rgb = cv2.resize(right_rgb, (int(round(w2 * tgt / h2)), tgt), interpolation=cv2.INTER_LANCZOS4)

    sep_w = max(5, 5 * scale // 2)
    pad_w = max(10, 10 * scale // 2)
    hh = left_rgb.shape[0]
    pad = np.ones((hh, pad_w, 3), dtype=np.uint8) * 255
    sep = np.zeros((hh, sep_w, 3), dtype=np.uint8)
    return np.concatenate([left_rgb, pad, sep, pad, right_rgb], axis=1)


# ────────────────────────────────────────────────────────────
# 7. End-to-end 主函数
# ────────────────────────────────────────────────────────────
def process_image(image_path: str, output_dir: str,
                  llm_model_path: Optional[str] = None,
                  scale: int = SCALE,
                  sam_device: str = "cuda",
                  molnextr_device: str = "cpu",
                  llm_device: str = "cuda",
                  sam_checkpoint: Optional[str] = None,
                  ocsr_checkpoint: Optional[str] = None,
                  ocsr_repo: str = "CYF200127/ChemEAGLEModel",
                  ocsr_filename: str = "molnextr.pth") -> Optional[Dict]:
    """
    End-to-end: 原图 → 去箭头 → 预测分子 → SMARTS → 高清拼接图 → LLM预测箭头 → Step SMARTS

    参数:
        image_path:       原始带箭头图像路径
        output_dir:       输出目录
        llm_model_path:   LLM 模型路径（merged 模型或 LoRA checkpoint）
                          为 None 时跳过箭头预测和 Step SMARTS 步骤
        scale:            放大倍数（默认 3）
        sam_device:       SAM3 设备
        molscribe_device: MolScribe 设备
        llm_device:       LLM 推理设备

    返回:
        dict 包含:
          中间输出: image_path, smarts, arrow_prediction
          最终输出: step_smarts (dict), step_smarts_str (格式化文本)
        或 None
    """
    n_steps = 7 if llm_model_path else 5
    os.makedirs(output_dir, exist_ok=True)
    base_name = Path(image_path).stem

    # ① 读取原图
    orig_bgr = cv2.imread(image_path)
    if orig_bgr is None:
        pil = Image.open(image_path).convert('RGB')
        orig_bgr = np.array(pil)[:, :, ::-1].copy()
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    print(f"[1/{n_steps}] 读取原图: {image_path}  ({orig_bgr.shape[1]}x{orig_bgr.shape[0]})")

    # ② 移除箭头
    print(f"[2/{n_steps}] SAM3 检测并移除箭头...")
    clean_rgb = remove_arrows(
        orig_rgb,
        device=sam_device,
        checkpoint_path=sam_checkpoint,
    )

    # ③ MolScribe 预测
    print(f"[3/{n_steps}] MolScribe 预测分子结构...")
    model = _load_molnextr(
        molnextr_device,
        checkpoint_path=ocsr_checkpoint,
        repo_id=ocsr_repo,
        filename=ocsr_filename,
    )
    predictions = model.predict_image(clean_rgb, return_atoms_bonds=True)
    _apply_patch(predictions)

    # ④ 生成 SMARTS
    print(f"[4/{n_steps}] 生成 SMARTS...")
    final_smarts = generate_smarts(predictions)
    predictions['final_smarts'] = final_smarts
    print(f"      SMILES:  {predictions.get('smiles', '')}")
    print(f"      Symbols: {predictions.get('symbols', [])}")
    print(f"      SMARTS:  {final_smarts}")

    # ⑤ 高清拼接
    print(f"[5/{n_steps}] 生成 {scale}x 高清拼接图...")
    h, w = orig_bgr.shape[:2]
    left_hd = cv2.resize(orig_bgr, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)

    molfile = predictions.get('molfile', '')
    right_hd = None
    try:
        right_hd = _draw_molecule_hd(predictions, molfile, orig_bgr, scale)
    except Exception as e:
        print(f"      RDKit 绘制失败 ({e})，fallback Lanczos 放大 clean 图")

    if right_hd is None:
        clean_bgr = cv2.cvtColor(clean_rgb, cv2.COLOR_RGB2BGR)
        right_hd = cv2.resize(clean_bgr, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)

    concat_rgb = _concat_lr(left_hd, right_hd, scale)

    out_img_path = os.path.join(output_dir, f"{base_name}_result.png")
    Image.fromarray(concat_rgb).save(out_img_path)

    # ⑥ LLM 箭头预测
    arrow_prediction = None
    step_smarts_dict = None
    step_smarts_str = None
    if llm_model_path:
        print(f"[6/{n_steps}] LLM 预测弯曲箭头...")
        arrow_prediction = predict_arrows(out_img_path, llm_model_path,
                                          device=llm_device)
        predictions['arrow_prediction'] = arrow_prediction
        print(f"      箭头预测: {arrow_prediction}")

        # ⑦ 生成 Step SMARTS
        print(f"[7/{n_steps}] 生成 Step SMARTS...")
        if final_smarts and arrow_prediction and not arrow_prediction.startswith("Error:"):
            step_smarts_dict = generate_step_smarts(final_smarts, arrow_prediction)
            if step_smarts_dict:
                step_smarts_str = "\n".join(
                    f"{label}: {sma}" for label, sma in sorted(step_smarts_dict.items()))
                predictions['step_smarts'] = step_smarts_dict
                predictions['step_smarts_str'] = step_smarts_str
                print(f"      Step SMARTS:")
                for label in sorted(step_smarts_dict):
                    print(f"        {label}: {step_smarts_dict[label]}")
            else:
                print("      Step SMARTS 生成失败（arrows_to_steps 返回空）")
        else:
            reason = "SMARTS 为空" if not final_smarts else "箭头预测为空或报错"
            print(f"      跳过 Step SMARTS（{reason}）")

    # 保存完整数据 JSON
    out_json_path = os.path.join(output_dir, f"{base_name}_data.json")
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    # 打印汇总
    print(f"\n  === {base_name} 结果汇总 ===")
    print(f"  [中间] 拼接图像:    {out_img_path}")
    print(f"  [中间] 原始 SMARTS: {final_smarts}")
    if arrow_prediction is not None:
        print(f"  [中间] LLM 箭头:    {arrow_prediction}")
    if step_smarts_str:
        print(f"  [最终] Step SMARTS:")
        for line in step_smarts_str.split("\n"):
            print(f"           {line}")
    else:
        print(f"  [最终] Step SMARTS: (无)")
    print(f"  [数据] JSON:        {out_json_path}")

    result = {
        "smarts": final_smarts,
        "image_path": out_img_path,
        "arrow_prediction": arrow_prediction,
        "step_smarts": step_smarts_dict,
        "step_smarts_str": step_smarts_str,
        "predictions": predictions,
    }
    return result


# ────────────────────────────────────────────────────────────
# 8. 批量处理
# ────────────────────────────────────────────────────────────
def batch_process(input_dir: str, output_dir: str,
                  llm_model_path: Optional[str] = None,
                  scale: int = SCALE,
                  sam_device: str = "cuda",
                  molnextr_device: str = "cpu",
                  llm_device: str = "cuda",
                  sam_checkpoint: Optional[str] = None,
                  ocsr_checkpoint: Optional[str] = None,
                  ocsr_repo: str = "CYF200127/ChemEAGLEModel",
                  ocsr_filename: str = "molnextr.pth"):
    """批量处理文件夹中所有图像。"""
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    files = [p for p in Path(input_dir).iterdir()
             if p.suffix.lower() in exts]
    print(f"找到 {len(files)} 张图像\n")

    results = []
    ok = fail = 0
    for i, fp in enumerate(sorted(files)):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(files)}] {fp.name}")
        try:
            ret = process_image(str(fp), output_dir, llm_model_path,
                                scale, sam_device, molnextr_device, llm_device,
                                sam_checkpoint, ocsr_checkpoint,
                                ocsr_repo, ocsr_filename)
            if ret:
                results.append({"file": fp.name, **ret})
                ok += 1
            else:
                fail += 1
        except Exception as e:
            print(f"  ERR {e}")
            fail += 1

    summary_path = os.path.join(output_dir, "summary.json")
    summary_data = []
    for r in results:
        entry = {
            "file": r["file"],
            "image": r["image_path"],
            "smarts": r["smarts"],
            "arrow_prediction": r.get("arrow_prediction"),
            "step_smarts": r.get("step_smarts"),
            "step_smarts_str": r.get("step_smarts_str"),
        }
        summary_data.append(entry)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    n_with_steps = sum(1 for r in results if r.get("step_smarts"))
    print(f"\n{'='*60}")
    print(f"完成！成功 {ok} / 失败 {fail}")
    if llm_model_path:
        print(f"生成 Step SMARTS: {n_with_steps}/{ok}")
    print(f"汇总: {summary_path}")
    return results


# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="End-to-end: 原图 → 去箭头 → 预测 → SMARTS → 高清图 → LLM箭头预测")
    parser.add_argument("input", help="单张图像路径 或 文件夹路径")
    parser.add_argument("-o", "--output", default="pipeline_output", help="输出目录")
    parser.add_argument("--llm-model", default=None,
                        help="LLM 模型路径（merged 模型或 LoRA checkpoint），不传则跳过箭头预测")
    parser.add_argument("--scale", type=int, default=SCALE, help="放大倍数")
    parser.add_argument("--sam-device", default="cuda")
    parser.add_argument("--mol-device", default="cpu")
    parser.add_argument("--llm-device", default="cuda")
    parser.add_argument("--sam-checkpoint", default=None,
                        help="Optional local SAM3 checkpoint; otherwise download from Hugging Face")
    parser.add_argument("--ocsr-checkpoint", default=None,
                        help="Optional local MolNexTR checkpoint")
    parser.add_argument("--ocsr-repo", default="CYF200127/ChemEAGLEModel",
                        help="Hugging Face repository used when --ocsr-checkpoint is omitted")
    parser.add_argument("--ocsr-filename", default="molnextr.pth")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        batch_process(args.input, args.output, args.llm_model,
                      args.scale, args.sam_device, args.mol_device, args.llm_device,
                      args.sam_checkpoint, args.ocsr_checkpoint,
                      args.ocsr_repo, args.ocsr_filename)
    else:
        process_image(args.input, args.output, args.llm_model,
                      args.scale, args.sam_device, args.mol_device, args.llm_device,
                      args.sam_checkpoint, args.ocsr_checkpoint,
                      args.ocsr_repo, args.ocsr_filename)
