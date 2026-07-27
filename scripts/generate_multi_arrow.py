import argparse, os, sys, json, csv, random, cv2, re, numpy as np
from pathlib import Path
from collections import deque
from io import BytesIO
from multiprocessing import Pool, cpu_count

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from molnextr.dataset import generate_indigo_image
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image

OUT_DIR = None
BASE_SEED = 42

# ============ 工具函数 ============
KWARGS = dict(mol_augment=True, default_option=True, shuffle_nodes=False, pseudo_coords=False)

BOND_NAME = {1: 'Single', 2: 'Double', 3: 'Triple', 4: 'Aromatic', 5: 'Single', 6: 'Single'}

PROMPT = """<image>
Task: Analyze the provided chemical image.
- The left panel shows a molecule with curved arrows indicating sequential electron flow.
- The right panel shows the same molecule with atom indices for reference.

Your goal is to identify the Start and End points of each curved arrow in the left panel by matching them with the atom indices in the right panel. The arrows are numbered in the order of electron flow (the end of one arrow connects to the start of the next).

Entity naming rules:
- For an atom, use: atom [Element:ID]
- For a bond, use: <Type> Bond between atoms [Element:ID] and [Element:ID] (Types: Single, Double, Triple, Aromatic)

Strictly output your answer in the following format, with no extra text or explanations:
[Arrow 1: Start: <Start_Entity> -> End: <End_Entity>; Arrow 2: Start: <Start_Entity> -> End: <End_Entity>; ...]

Example of expected output:
[Arrow 1: Start: Single Bond between atoms [C:15] and [N:16] -> End: atom [C:14]; Arrow 2: Start: atom [C:14] -> End: atom [O:3]]"""


def fmt_ent(symbols, typ, val):
    if typ == 'atom':
        return f"atom [{symbols[val]}:{val}]"
    i, j, o = val
    return f"{BOND_NAME.get(o,'Single')} Bond between atoms [{symbols[i]}:{i}] and [{symbols[j]}:{j}]"


def build_adj(n, bonds):
    adj = [[] for _ in range(n)]
    for i, j, _ in bonds:
        adj[i].append(j)
        adj[j].append(i)
    return adj


def topo_dist(adj, a, b):
    if a == b: return 0
    visited = {a}
    q = deque([(a, 0)])
    while q:
        cur, d = q.popleft()
        for nb in adj[cur]:
            if nb == b: return d + 1
            if nb not in visited:
                visited.add(nb)
                q.append((nb, d + 1))
    return 999


def dseg(pt, a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    ap = np.array(pt, dtype=float) - a; ab = b - a
    t = np.clip(np.dot(ap, ab) / (np.dot(ab, ab) + 1e-10), 0, 1)
    return np.linalg.norm(a + t * ab - np.array(pt, dtype=float))


def seg_intersect(p1, p2, p3, p4):
    p1, p2, p3, p4 = [np.array(x, dtype=float) for x in [p1, p2, p3, p4]]
    def cross(a, b): return a[0]*b[1] - a[1]*b[0]
    d1 = cross(p4-p3, p1-p3); d2 = cross(p4-p3, p2-p3)
    d3 = cross(p2-p1, p3-p1); d4 = cross(p2-p1, p4-p1)
    return ((d1>0 and d2<0) or (d1<0 and d2>0)) and ((d3>0 and d4<0) or (d3<0 and d4>0))


BOND_TYPE_MAP = {
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE,
    4: Chem.rdchem.BondType.AROMATIC,
    5: Chem.rdchem.BondType.SINGLE,
    6: Chem.rdchem.BondType.SINGLE,
}


def _make_rdkit_atom(symbol):
    m = re.match(r'^([A-Z][a-z]?)(\d*)([\+\-])$', symbol)
    if m:
        base_elem, count, sign = m.groups()
        charge_mag = int(count) if count else 1
        charge = charge_mag if sign == '+' else -charge_mag
        try:
            atom = Chem.Atom(base_elem)
            atom.SetFormalCharge(charge)
            return atom
        except Exception:
            pass
    try:
        return Chem.Atom(symbol)
    except Exception:
        pass
    atom = Chem.Atom(0)
    atom.SetProp('atomLabel', symbol)
    return atom


def _build_rdkit_mol(graph):
    symbols = graph['symbols']
    edges   = np.array(graph['edges'])
    coords  = np.array(graph['coords'])
    n       = int(graph['num_atoms'])

    rw_mol = Chem.RWMol()
    for i in range(n):
        rw_mol.AddAtom(_make_rdkit_atom(symbols[i]))

    for i in range(n):
        for j in range(i + 1, n):
            bo = int(edges[i][j])
            if bo >= 1:
                bt = BOND_TYPE_MAP.get(bo, Chem.rdchem.BondType.SINGLE)
                rw_mol.AddBond(i, j, bt)
                if bo == 5:
                    rw_mol.GetBondBetweenAtoms(i, j).SetBondDir(
                        Chem.rdchem.BondDir.BEGINWEDGE)
                elif bo == 6:
                    rw_mol.GetBondBetweenAtoms(i, j).SetBondDir(
                        Chem.rdchem.BondDir.BEGINDASH)

    mol = rw_mol.GetMol()
    conf = Chem.Conformer(n)
    for i in range(n):
        conf.SetAtomPosition(i, (float(coords[i][0]), float(coords[i][1]), 0.0))
    mol.AddConformer(conf)
    return mol, n


def draw_mol_rdkit(graph, width, height, add_atom_indices=True, bw=True):
    """渲染分子并返回 (image, atom_pixel_coords)。
    atom_pixel_coords: list of (px, py) 每个原子在图像上的精确像素位置。
    """
    mol, n = _build_rdkit_mol(graph)

    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    opts = drawer.drawOptions()
    if bw:
        opts.useBWAtomPalette()
    opts.addAtomIndices = add_atom_indices
    opts.addStereoAnnotation = False

    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    atom_px = []
    for i in range(n):
        pt = drawer.GetDrawCoords(i)
        atom_px.append((int(round(pt.x)), int(round(pt.y))))

    png_bytes = drawer.GetDrawingText()
    img_pil = Image.open(BytesIO(png_bytes))
    img = np.ascontiguousarray(np.array(img_pil.convert('RGB'))[:, :, ::-1])
    return img, atom_px


def generate_one(smiles_str, cid):
    try:
      return _generate_one_inner(smiles_str, cid)
    except Exception as e:
      print(f"  [ERR] cid={cid}: {type(e).__name__}: {e}")
      return None

def _generate_one_inner(smiles_str, cid):
    seed = random.randint(0, 2**32 - 1)

    random.seed(seed)
    img_indigo, _, graph, ok_l = generate_indigo_image(smiles_str, **KWARGS, show_atom_ids=False)

    if not (ok_l and graph):
        return None
    coords = np.array(graph.get('coords', []))
    if coords.ndim != 2 or coords.shape[0] < 2 or coords.shape[1] < 2:
        return None

    h1, w1 = img_indigo.shape[:2]

    img_left, ats = draw_mol_rdkit(graph, w1, h1, add_atom_indices=False, bw=False)
    img_right, _ = draw_mol_rdkit(graph, w1, h1, add_atom_indices=True, bw=True)
    h1, w1 = img_left.shape[:2]
    h2, w2 = img_right.shape[:2]

    symbols = graph['symbols']
    edges = graph['edges']
    n = int(graph['num_atoms'])

    bonds = [(i, j, int(edges[i,j])) for i in range(n) for j in range(i+1, n) if edges[i,j] >= 1]
    if not bonds:
        return None
    bond_px_lens = [np.linalg.norm(np.array(ats[i], dtype=float) - np.array(ats[j], dtype=float))
                    for i, j, _ in bonds]
    bond_scale_px = np.median(bond_px_lens)
    adj = build_adj(n, bonds)

    center_px = np.mean(ats, axis=0)
    segs = [(ats[i], ats[j], i, j) for i, j, _ in bonds]
    d_bond_thr, d_atom_thr = 8, 8
    drawn_arrow_pts = []

    def curve_ok(pts, ign_a, ign_b):
        for pt in pts:
            if pt[0] < 0 or pt[0] >= w1 or pt[1] < 0 or pt[1] >= h1: return False
            for ia, ap in enumerate(ats):
                if ia in ign_a: continue
                if np.linalg.norm(np.array(pt,dtype=float)-np.array(ap,dtype=float)) < d_atom_thr: return False
            for (pa, pb, bi, bj) in segs:
                if (min(bi,bj), max(bi,bj)) in ign_b: continue
                if dseg(pt, pa, pb) < d_bond_thr: return False
        for k in range(len(pts)-1):
            for (pa, pb, bi, bj) in segs:
                if (min(bi,bj), max(bi,bj)) in ign_b: continue
                if seg_intersect(pts[k], pts[k+1], pa, pb): return False
        for prev_pts in drawn_arrow_pts:
            for k in range(len(pts)-1):
                for m in range(len(prev_pts)-1):
                    if seg_intersect(pts[k], pts[k+1], prev_pts[m], prev_pts[m+1]):
                        return False
        return True

    # 随机样式
    rs = random.Random(seed + 2)
    line_w = rs.choice([1, 1, 2, 2])
    alen = rs.randint(8, 11)
    arrow_spread = rs.uniform(0.30, 0.45)

    r = random.Random(seed + 1)
    num_arrows = r.choice([2, 2, 2, 3, 3])
    arrow_gts = []
    prev_end_typ = None
    prev_end_val = None
    used_entities = set()

    def _ekey(typ, val):
        if typ == 'atom': return ('a', val)
        return ('b', min(val[0], val[1]), max(val[0], val[1]))

    for arrow_idx in range(num_arrows):
        arrow_found = False
        for trial in range(400):
            if arrow_idx == 0:
                st_typ = r.choice(['atom', 'bond'])
                st_val = r.randrange(n) if st_typ == 'atom' else r.choice(bonds)
                if _ekey(st_typ, st_val) in used_entities: continue
            else:
                roll = r.random()
                if roll < 0.5:
                    st_typ, st_val = prev_end_typ, prev_end_val
                elif prev_end_typ == 'atom':
                    adj_bonds = [b for b in bonds if prev_end_val in (b[0], b[1])]
                    if adj_bonds:
                        st_typ, st_val = 'bond', r.choice(adj_bonds)
                    else:
                        st_typ, st_val = prev_end_typ, prev_end_val
                else:
                    st_typ = 'atom'
                    st_val = r.choice([prev_end_val[0], prev_end_val[1]])

            en_typ = r.choice(['atom', 'bond'])
            en_val = r.randrange(n) if en_typ == 'atom' else r.choice(bonds)
            if st_typ == 'atom' and en_typ == 'atom' and st_val == en_val: continue
            if st_typ == 'bond' and en_typ == 'bond' and sorted(st_val[:2]) == sorted(en_val[:2]): continue
            if _ekey(en_typ, en_val) in used_entities: continue

            st_atoms = [st_val] if st_typ == 'atom' else [st_val[0], st_val[1]]
            en_atoms = [en_val] if en_typ == 'atom' else [en_val[0], en_val[1]]
            min_td = min(topo_dist(adj, a, b) for a in st_atoms for b in en_atoms)
            if min_td < 1 or min_td > 3: continue

            p0 = np.asarray(ats[st_val] if st_typ=='atom'
                            else ((np.array(ats[st_val[0]], dtype=float) + np.array(ats[st_val[1]], dtype=float)) / 2),
                            dtype=float)
            p2 = np.asarray(ats[en_val] if en_typ=='atom'
                            else ((np.array(ats[en_val[0]], dtype=float) + np.array(ats[en_val[1]], dtype=float)) / 2),
                            dtype=float)

            ign_a = set(st_atoms + en_atoms)
            ign_b = set()
            if st_typ == 'bond': ign_b.add((min(st_val[0],st_val[1]), max(st_val[0],st_val[1])))
            if en_typ == 'bond': ign_b.add((min(en_val[0],en_val[1]), max(en_val[0],en_val[1])))

            d_vec = p2 - p0
            dist_se = max(np.linalg.norm(d_vec), 1e-5)
            perp = np.array([-d_vec[1], d_vec[0]]) / dist_se
            mid = (p0 + p2) / 2
            to_center = center_px - mid
            sign_inward = 1 if np.dot(to_center, perp) > 0 else -1

            offset_px = 0.12 * bond_scale_px

            for sign in [sign_inward, -sign_inward]:
                for kk in [0.40, 0.30, 0.50, 0.25]:
                    ctrl = mid + sign * kk * dist_se * perp

                    # --- 起点漂移 ---
                    if st_typ == 'atom':
                        tan0 = 2.0 * (ctrl - p0)
                        tn0 = np.linalg.norm(tan0)
                        tan0 = tan0 / tn0 if tn0 > 1e-6 else np.array([0., 1.])
                        p0_off = p0 + offset_px * tan0
                    else:
                        bv = np.array(ats[st_val[1]], dtype=float) - np.array(ats[st_val[0]], dtype=float)
                        bp = np.array([-bv[1], bv[0]])
                        bpn = np.linalg.norm(bp)
                        bp = bp / bpn if bpn > 1e-6 else np.array([0., 1.])
                        s_bp = 1 if np.dot(ctrl - p0, bp) > 0 else -1
                        p0_off = p0 + offset_px * s_bp * bp

                    # --- 终点漂移 ---
                    if en_typ == 'atom':
                        tan2 = 2.0 * (p2 - ctrl)
                        tn2 = np.linalg.norm(tan2)
                        tan2 = tan2 / tn2 if tn2 > 1e-6 else np.array([0., 1.])
                        p2_off = p2 - offset_px * tan2
                    else:
                        bv = np.array(ats[en_val[1]], dtype=float) - np.array(ats[en_val[0]], dtype=float)
                        bp = np.array([-bv[1], bv[0]])
                        bpn = np.linalg.norm(bp)
                        bp = bp / bpn if bpn > 1e-6 else np.array([0., 1.])
                        s_bp = 1 if np.dot(ctrl - p2, bp) > 0 else -1
                        p2_off = p2 + offset_px * s_bp * bp

                    pts = [(int(round((1-t)**2*p0_off[0] + 2*(1-t)*t*ctrl[0] + t**2*p2_off[0])),
                            int(round((1-t)**2*p0_off[1] + 2*(1-t)*t*ctrl[1] + t**2*p2_off[1])))
                           for t in np.linspace(0, 1, 50)]
                    if curve_ok(pts, ign_a, ign_b):
                        arr = np.array(pts, dtype=np.int32)
                        cv2.polylines(img_left, [arr], False, (0,0,0), line_w, cv2.LINE_AA)
                        tip, tail = pts[-1], pts[-3]
                        dx = tip[0]-tail[0]; dy = tip[1]-tail[1]
                        L = max((dx*dx+dy*dy)**0.5, 1); dx, dy = dx/L, dy/L
                        a1 = (int(tip[0]-alen*dx+alen*arrow_spread*dy), int(tip[1]-alen*dy-alen*arrow_spread*dx))
                        a2 = (int(tip[0]-alen*dx-alen*arrow_spread*dy), int(tip[1]-alen*dy+alen*arrow_spread*dx))
                        cv2.fillConvexPoly(img_left, np.array([tip, a1, a2]), (0,0,0))
                        drawn_arrow_pts.append(pts)
                        st_str = fmt_ent(symbols, st_typ, st_val)
                        en_str = fmt_ent(symbols, en_typ, en_val)
                        arrow_gts.append(f"Arrow {arrow_idx+1}: Start: {st_str} -> End: {en_str}")
                        used_entities.add(_ekey(st_typ, st_val))
                        used_entities.add(_ekey(en_typ, en_val))
                        prev_end_typ = en_typ
                        prev_end_val = en_val
                        arrow_found = True
                        break
                if arrow_found: break
            if arrow_found: break

        if not arrow_found:
            break

    if len(arrow_gts) < 2:
        return None

    arrow_gt = "[" + "; ".join(arrow_gts) + "]"

    # 拼接左右图
    h = max(h1, h2)
    if h1 != h: img_left = cv2.resize(img_left, (int(w1 * h / h1), h))
    if h2 != h: img_right = cv2.resize(img_right, (int(w2 * h / h2), h))
    sep = np.zeros((h, 3, 3), dtype=np.uint8)
    combined = np.hstack([img_left, sep, img_right])

    fname = f"{cid}.png"
    cv2.imwrite(os.path.join(OUT_DIR, "images", fname), combined)
    return fname, arrow_gt


# ============ 批量生成（多进程） ============
def _worker(args):
    """Worker function for multiprocessing."""
    row, label, idx = args
    random.seed(BASE_SEED + idx)
    smi = row['SMILES']
    cid = row['pubchem_cid']
    ret = generate_one(smi, cid)
    if ret is None:
        return None
    fname, gt = ret
    entry = {
        "id": f"chem_task_{cid}",
        "image": fname,
        "conversations": [
            {"from": "human", "value": PROMPT},
            {"from": "gpt", "value": gt}
        ]
    }
    return entry


def generate_split_parallel(rows, label, num_workers):
    args_list = [(row, label, i) for i, row in enumerate(rows)]
    results = []
    ok = 0
    skip = 0
    with Pool(num_workers) as pool:
        for i, entry in enumerate(pool.imap(_worker, args_list, chunksize=64)):
            if entry is not None:
                results.append(entry)
                ok += 1
            else:
                skip += 1
            if (i + 1) % 1000 == 0:
                print(f"[{label}] {i+1}/{len(rows)} processed, {ok} OK, {skip} SKIP")
    print(f"[{label}] Done: {ok} OK, {skip} SKIP out of {len(rows)}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Generate multi-arrow left-right training images")
    parser.add_argument("--csv", required=True, help="CSV containing SMILES and pubchem_cid columns")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--exclude-manifest", action="append", default=[],
                        help="Manifest whose chem_task_<CID> entries must be excluded; repeatable")
    parser.add_argument("--num-train", type=int, default=200_000)
    parser.add_argument("--num-eval", type=int, default=500)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    global OUT_DIR, BASE_SEED
    args = parse_args()
    OUT_DIR = os.path.abspath(args.output_dir)
    BASE_SEED = args.seed
    os.makedirs(os.path.join(OUT_DIR, "images"), exist_ok=True)

    used_cids = set()
    for manifest in args.exclude_manifest:
        with open(manifest, "r", encoding="utf-8") as handle:
            for item in json.load(handle):
                used_cids.add(item["id"].removeprefix("chem_task_"))

    with open(args.csv, "r", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["pubchem_cid"] not in used_cids]
    requested = args.num_train + args.num_eval
    if requested > len(rows):
        raise ValueError(f"Requested {requested:,} unique molecules, found {len(rows):,} after exclusions")

    sampled = random.Random(args.seed).sample(rows, requested)
    train_rows = sampled[:args.num_train]
    eval_rows = sampled[args.num_train:]
    print(f"Excluded {len(used_cids):,} CIDs; generating with {args.workers} workers")

    train_results = generate_split_parallel(train_rows, "TRAIN", args.workers)
    eval_results = generate_split_parallel(eval_rows, "EVAL", args.workers)
    train_path = os.path.join(OUT_DIR, "arrow_train.json")
    eval_path = os.path.join(OUT_DIR, "arrow_eval.json")
    with open(train_path, "w", encoding="utf-8") as handle:
        json.dump(train_results, handle, indent=2, ensure_ascii=False)
    with open(eval_path, "w", encoding="utf-8") as handle:
        json.dump(eval_results, handle, indent=2, ensure_ascii=False)

    metadata = {
        "source_csv": os.path.abspath(args.csv),
        "excluded_manifests": [os.path.abspath(path) for path in args.exclude_manifest],
        "excluded_cids": len(used_cids),
        "seed": args.seed,
        "workers": args.workers,
        "requested_train": args.num_train,
        "requested_eval": args.num_eval,
        "generated_train": len(train_results),
        "generated_eval": len(eval_results),
    }
    with open(os.path.join(OUT_DIR, "generation_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Generated train={len(train_results):,}, eval={len(eval_results):,} in {OUT_DIR}")


if __name__ == "__main__":
    main()