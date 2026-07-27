"""
reverse_arrows.py

从 input SMARTS + 箭头描述 [Arrow 1: Start: ... -> End: ...; ...]
反向生成各步的 annotated step SMARTS。
是 extract_arrows.py 的逆过程。

注释系统：
  - a/y 前缀: [aX:n] = 箭头起点原子(source), [yX:n] = 箭头终点原子(target)
  - 键符号变化:
      ! = 键断裂
      _ = 变为单键 (原来不是单键)
      ; = 变为双键 (原来不是双键)
      ^ = 变为三键 (原来不是三键)
  - δ 标记: 新键形成 (原图中不相邻的两个原子之间形成新键)

用法:
  # 单条测试
  python reverse_arrows.py --smarts "[O-:17]-[C:4](=[C:5]-[C:6]=[O:14])" \
    --arrows "[Arrow 1: Start: atom [O-:17] -> End: atom [C:4]; ...]"

  # 批量处理 JSON (需要 input_smarts 字段)
  python reverse_arrows.py --json data.json --output data_with_steps.json
"""

import re
import json
import sys
import argparse


class SMSMARTSParseError(ValueError):
    """Raised when a cumulative SM-SMARTS step cannot be decoded safely."""

# ===== 常量 =====

BOND_TYPE_MAP = {'Single': '-', 'Double': '=', 'Triple': '#', 'Aromatic': ':'}

BOND_ORDER = {
    '-': 1, '/': 1, '\\': 1, '_': 1,
    '=': 2, ';': 2,
    '#': 3, '^': 3,
    ':': 1.5, '!': 0
}

# 注释版键符号 (区别于标准键符号, 表示"变为该键序")
ANNOT_SYM = {1: '_', 2: ';', 3: '^'}


def _weaken(cur_order):
    """键失去电子后的新键序 (source)"""
    if cur_order >= 3:
        return 2
    if cur_order >= 2:
        return 1
    if cur_order >= 1.5:
        return 1    # aromatic → single
    if cur_order >= 1:
        return 0    # single → breaks
    return 0


def _strengthen(cur_order):
    """键获得电子后的新键序 (target)"""
    if cur_order <= 0:
        return 1
    if cur_order <= 1:
        return 2
    if cur_order <= 1.5:
        return 2    # aromatic → double
    if cur_order <= 2:
        return 3
    return int(cur_order) + 1


# ====================================================================
# Part 1 - 箭头描述解析器
# ====================================================================

def parse_arrows(text):
    """解析箭头字符串 → [(start_entity, end_entity), ...]

    每个 entity 是 tuple:
      atom: ('atom', element_str, mapping_id_or_None)
      bond: ('bond', bond_symbol, atom1_id, atom2_id)
    """
    text = text.strip()
    if text.startswith('[') and text.endswith(']'):
        text = text[1:-1]
    arrows = []
    for m in re.finditer(
            r'Arrow\s+\d+:\s*Start:\s*(.*?)\s*->\s*End:\s*(.*?)(?=;\s*Arrow\s+\d+:|$)',
            text):
        s = _parse_entity(m.group(1).strip())
        e = _parse_entity(m.group(2).strip())
        if s and e:
            arrows.append((s, e))
    return arrows


def _parse_entity(t):
    # atom [Element:ID]  或  atom [Element:unknown]
    m = re.match(r'atom\s+\[([^\]]*?):(\w+)\]', t)
    if m:
        mid = int(m.group(2)) if m.group(2) != 'unknown' else None
        return ('atom', m.group(1), mid)

    # <Type> Bond between atoms [E1:ID1] and [E2:ID2]
    m = re.match(
        r'(Single|Double|Triple|Aromatic)\s+Bond\s+between\s+atoms\s+'
        r'\[([^\]]*?):(\d+)\]\s+and\s+\[([^\]]*?):(\d+)\]', t)
    if m:
        return ('bond', BOND_TYPE_MAP[m.group(1)],
                int(m.group(3)), int(m.group(5)))
    return None


# ====================================================================
# Part 2 - SMARTS 分词器 & 结构映射
# ====================================================================

def tokenize(smarts):
    """将 SMARTS 字符串拆分为 token 列表。
    方括号原子整体作为一个 token，其余每个字符独立。"""
    toks = []
    i, n = 0, len(smarts)
    while i < n:
        if smarts[i] == '[':
            j = smarts.index(']', i)
            toks.append(smarts[i:j + 1])
            i = j + 1
        else:
            toks.append(smarts[i])
            i += 1
    return toks


def build_graph(tokens):
    """从 token 列表构建:
      atom_pos: {mapping_id: token_index}
      bond_pos: {(min_id, max_id): {'i': token_idx, 's': symbol, 'imp': bool}}
    """
    atom_pos = {}
    bond_pos = {}
    stack = []
    curr = None
    psym, pidx = None, None
    rings = {}

    for i, t in enumerate(tokens):
        if t.startswith('['):
            m = re.search(r':(\d+)\]$', t)
            if not m:
                continue
            mp = int(m.group(1))
            atom_pos[mp] = i
            if curr is not None:
                k = (min(curr, mp), max(curr, mp))
                if psym and psym not in '/\\':
                    bond_pos[k] = {'i': pidx, 's': psym, 'imp': False}
                elif psym in '/\\':
                    bond_pos[k] = {'i': pidx, 's': '-', 'imp': False}
                else:
                    bond_pos[k] = {'i': i, 's': '-', 'imp': True}
            curr, psym, pidx = mp, None, None

        elif t == '(':
            stack.append(curr)
            psym, pidx = None, None
        elif t == ')':
            curr = stack.pop() if stack else curr
            psym, pidx = None, None
        elif t == '.':
            curr, psym, pidx = None, None, None
        elif len(t) == 1 and t in '-=#!/;_^:\\/':
            psym, pidx = t, i
        elif t.isdigit():
            d = int(t)
            bs, bi = psym or '-', pidx
            psym, pidx = None, None
            if d in rings:
                om, obs, obi, _odi = rings.pop(d)
                k = (min(curr, om), max(curr, om))
                if bs not in ('-', '/', '\\') and bi is not None:
                    bond_pos[k] = {'i': bi, 's': bs, 'imp': False, 'ring': True}
                elif obs not in ('-', '/', '\\') and obi is not None:
                    bond_pos[k] = {'i': obi, 's': obs, 'imp': False, 'ring': True}
                else:
                    bond_pos[k] = {'i': i, 's': '-', 'imp': True, 'ring': True}
            else:
                rings[d] = (curr, bs, bi, i)

    return atom_pos, bond_pos


# ====================================================================
# Part 3 - 核心: 箭头 → step SMARTS 生成
# ====================================================================

def arrows_to_steps(input_smarts, arrows_str):
    """核心函数: input SMARTS + 箭头描述 → [(label, step_smarts), ...]

    每步的 step SMARTS 是 input SMARTS 加上累计注释:
      - 原子 a/y 前缀
      - 键符号变化 (!, _, ;, ^)
      - δ 标记 (新键形成)
    """
    tokens = tokenize(input_smarts)
    atom_pos, bond_pos = build_graph(tokens)
    arrows = parse_arrows(arrows_str)

    # 累计状态
    prefixes = {}       # mapping_id → 'a' / 'y'
    bond_mods = {}      # (min_id, max_id) → 注释键符号
    bond_state = {}     # (min_id, max_id) → 当前键序 (累计追踪)
    deltas = set()      # 有 δ 标记的原子集合

    for k, info in bond_pos.items():
        bond_state[k] = BOND_ORDER.get(info['s'], 1)

    def _is_adjacent(x, y):
        return (min(x, y), max(x, y)) in bond_pos

    steps = []
    for idx, (start, end) in enumerate(arrows):
        st, et = start[0], end[0]

        # ---- 新键形成检测 (覆盖 atom→atom, bond→atom, atom→bond) ----
        new_bond = False
        new_bond_donor = None  # bond 侧参与新键的原子

        if st == 'atom' and et == 'atom':
            if start[2] is not None and end[2] is not None:
                if not _is_adjacent(start[2], end[2]):
                    new_bond = True

        elif st == 'bond' and et == 'atom' and end[2] is not None:
            _, _, ba1, ba2 = start
            a3 = end[2]
            if not _is_adjacent(ba1, a3) and not _is_adjacent(ba2, a3):
                new_bond = True
                ba1_touched = (ba1 in prefixes
                               or any(ba1 in bk for bk in bond_mods))
                ba2_touched = (ba2 in prefixes
                               or any(ba2 in bk for bk in bond_mods))
                new_bond_donor = ba1 if (ba2_touched and not ba1_touched) else (
                                 ba2 if (ba1_touched and not ba2_touched) else ba1)

        elif st == 'atom' and et == 'bond' and start[2] is not None:
            _, _, ba1, ba2 = end
            a3 = start[2]
            if not _is_adjacent(ba1, a3) and not _is_adjacent(ba2, a3):
                new_bond = True
                ba1_touched = (ba1 in prefixes
                               or any(ba1 in bk for bk in bond_mods))
                ba2_touched = (ba2 in prefixes
                               or any(ba2 in bk for bk in bond_mods))
                new_bond_donor = ba1 if (ba2_touched and not ba1_touched) else (
                                 ba2 if (ba1_touched and not ba2_touched) else ba1)

        # ----- 处理 Start -----
        if st == 'atom' and start[2] is not None:
            prefixes[start[2]] = 'a'
            if new_bond:
                deltas.add(start[2])
        elif st == 'bond':
            _sym, _s, a1, a2 = start
            k = (min(a1, a2), max(a1, a2))
            cur = bond_state.get(k, 1)
            nw = _weaken(cur)
            bond_mods[k] = '!' if nw <= 0 else ANNOT_SYM[nw]
            bond_state[k] = nw
            if new_bond and new_bond_donor is not None:
                deltas.add(new_bond_donor)

        # ----- 处理 End -----
        if et == 'atom' and end[2] is not None:
            if new_bond:
                deltas.add(end[2])
            else:
                prefixes[end[2]] = 'y'
        elif et == 'bond':
            _sym, _s, a1, a2 = end
            k = (min(a1, a2), max(a1, a2))
            cur = bond_state.get(k, 1)
            nw = _strengthen(cur)
            bond_mods[k] = ANNOT_SYM.get(nw, '^')
            bond_state[k] = nw
            if new_bond and new_bond_donor is not None:
                deltas.add(new_bond_donor)

        # 生成本步的 step SMARTS
        step = _build_step(tokens, atom_pos, bond_pos,
                           prefixes, bond_mods, deltas)
        steps.append((chr(65 + idx), step))

    return steps


def _build_step(tokens, atom_pos, bond_pos, prefixes, bond_mods, deltas):
    """根据当前累计的注释状态，生成一步 annotated SMARTS。"""
    new_toks = list(tokens)

    # 1) 键符号替换 / 插入
    inserts = {}
    for k, sym in bond_mods.items():
        if k not in bond_pos:
            continue
        info = bond_pos[k]
        if info['imp']:
            inserts[info['i']] = sym
        else:
            new_toks[info['i']] = sym

    # 2) 原子前缀
    for mp, pf in prefixes.items():
        if mp not in atom_pos:
            continue
        i = atom_pos[mp]
        new_toks[i] = '[' + pf + tokens[i][1:]

    # 3) 拼接，并添加 δ 标记
    parts = []
    for i, tok in enumerate(new_toks):
        if i in inserts:
            parts.append(inserts[i])
        parts.append(tok)
        if tok.startswith('['):
            m = re.search(r':(\d+)\]$', tok)
            if m and int(m.group(1)) in deltas:
                parts.append('\u03b4')

    return ''.join(parts)


# ====================================================================
# Part 4 - 累积 step SMARTS -> 有序箭头描述
# ====================================================================

_ANNOTATED_BONDS = {'!', '_', ';', '^'}
_BOND_LABELS = {
    '-': 'Single', '/': 'Single', '\\': 'Single',
    '=': 'Double', '#': 'Triple', ':': 'Aromatic',
}


def _extract_step_state(step_smarts):
    tokens = tokenize(step_smarts)
    prefixes = {}
    deltas = set()
    atoms = {}

    for index, token in enumerate(tokens):
        if not token.startswith('['):
            continue
        match = re.match(r'^\[([ay])?(.+):(\d+)\]$', token)
        if not match:
            raise SMSMARTSParseError(f'Cannot parse annotated atom token: {token}')
        prefix, symbol, map_id = match.groups()
        map_id = int(map_id)
        atoms[map_id] = symbol
        if prefix:
            prefixes[map_id] = prefix
        if index + 1 < len(tokens) and tokens[index + 1] == 'δ':
            deltas.add(map_id)

    _, bond_positions = build_graph(tokens)
    bonds = {
        key: info['s']
        for key, info in bond_positions.items()
        if info['s'] in _ANNOTATED_BONDS
    }
    return {
        'prefixes': prefixes,
        'deltas': deltas,
        'bonds': bonds,
        'atoms': atoms,
    }


def _base_graph(base_smarts):
    tokens = tokenize(base_smarts)
    atom_positions, bond_positions = build_graph(tokens)
    atoms = {}
    for map_id, index in atom_positions.items():
        match = re.match(r'^\[(.+):(\d+)\]$', tokens[index])
        if not match:
            raise SMSMARTSParseError(f'Base atom is not atom-mapped: {tokens[index]}')
        atoms[map_id] = match.group(1)
    bond_orders = {key: info['s'] for key, info in bond_positions.items()}
    return atoms, bond_orders


def _format_atom(atoms, map_id):
    if map_id not in atoms:
        raise SMSMARTSParseError(f'Unknown atom-map ID: {map_id}')
    return f'atom [{atoms[map_id]}:{map_id}]'


def _format_bond(atoms, key, bond_orders):
    if key not in bond_orders:
        raise SMSMARTSParseError(f'Unknown bond: {key[0]}-{key[1]}')
    symbol = bond_orders[key]
    bond_type = _BOND_LABELS.get(symbol)
    if bond_type is None:
        raise SMSMARTSParseError(f'Unsupported source bond order {symbol!r} for {key}')
    left, right = key
    return (f'{bond_type} Bond between atoms '
            f'[{atoms[left]}:{left}] and [{atoms[right]}:{right}]')


def _changed(current, previous):
    return {
        key: value for key, value in current.items()
        if previous.get(key) != value
    }


def _decode_step(previous, current, atoms, bond_orders, label):
    prefix_changes = _changed(current['prefixes'], previous['prefixes'])
    bond_changes = _changed(current['bonds'], previous['bonds'])
    new_deltas = current['deltas'] - previous['deltas']

    source_atoms = [map_id for map_id, prefix in prefix_changes.items() if prefix == 'a']
    target_atoms = [map_id for map_id, prefix in prefix_changes.items() if prefix == 'y']
    source_bonds = [key for key, symbol in bond_changes.items() if symbol in {'!', '_'}]
    target_bonds = [key for key, symbol in bond_changes.items() if symbol in {';', '^'}]

    if len(source_atoms) + len(source_bonds) != 1:
        raise SMSMARTSParseError(
            f'{label}: expected one electron source, found atoms={source_atoms}, bonds={source_bonds}'
        )
    if len(target_atoms) > 1 or len(target_bonds) > 1:
        raise SMSMARTSParseError(
            f'{label}: multiple targets are ambiguous: atoms={target_atoms}, bonds={target_bonds}'
        )
    if new_deltas and len(new_deltas) != 2:
        raise SMSMARTSParseError(f'{label}: a new bond requires exactly two delta endpoints')

    if source_atoms:
        source_kind, source_value = 'atom', source_atoms[0]
        source_text = _format_atom(atoms, source_value)
    else:
        source_kind, source_value = 'bond', source_bonds[0]
        source_text = _format_bond(atoms, source_value, bond_orders)

    if target_bonds:
        target_kind, target_value = 'bond', target_bonds[0]
        target_text = _format_bond(atoms, target_value, bond_orders)
    elif target_atoms:
        target_kind, target_value = 'atom', target_atoms[0]
        target_text = _format_atom(atoms, target_value)
    elif new_deltas:
        if source_kind == 'atom':
            candidates = new_deltas - {source_value}
        else:
            candidates = new_deltas - set(source_value)
        if len(candidates) != 1:
            raise SMSMARTSParseError(
                f'{label}: cannot identify the new-bond target from delta endpoints {sorted(new_deltas)}'
            )
        target_kind, target_value = 'atom', candidates.pop()
        target_text = _format_atom(atoms, target_value)
    else:
        raise SMSMARTSParseError(f'{label}: no electron target marker found')

    signature = (source_kind, target_kind, bool(new_deltas))
    allowed = {
        ('atom', 'atom', False),
        ('atom', 'atom', True),
        ('atom', 'bond', False),
        ('atom', 'bond', True),
        ('bond', 'atom', False),
        ('bond', 'atom', True),
        ('bond', 'bond', False),
    }
    if signature not in allowed:
        raise SMSMARTSParseError(f'{label}: unsupported conservation signature {signature}')

    for key, symbol in bond_changes.items():
        bond_orders[key] = {'!': None, '_': '-', ';': '=', '^': '#'}[symbol]

    return {
        'label': label,
        'start': source_text,
        'end': target_text,
        'signature': {
            'source': source_kind,
            'target': target_kind,
            'new_bond': bool(new_deltas),
        },
        'edits': {
            'prefixes': prefix_changes,
            'bonds': {f'{left}-{right}': symbol
                      for (left, right), symbol in bond_changes.items()},
            'delta_atoms': sorted(new_deltas),
        },
    }


def decode_steps(base_smarts, steps):
    """Decode cumulative step SMARTS into an ordered arrow list.

    ``base_smarts`` is required because the legacy annotation symbols replace
    native bond symbols, so the original bond type is not recoverable from an
    annotated step alone.
    """
    if isinstance(steps, dict):
        items = list(steps.items())
    else:
        items = list(steps)
    if not items:
        raise SMSMARTSParseError('At least one cumulative step is required')

    atoms, bond_orders = _base_graph(base_smarts)
    previous = {'prefixes': {}, 'deltas': set(), 'bonds': {}, 'atoms': atoms}
    decoded = []
    for label, step_smarts in items:
        current = _extract_step_state(step_smarts)
        decoded.append(_decode_step(previous, current, atoms, bond_orders, str(label)))
        previous = current

    arrow_text = '[' + '; '.join(
        f"Arrow {index}: Start: {arrow['start']} -> End: {arrow['end']}"
        for index, arrow in enumerate(decoded, start=1)
    ) + ']'
    return {
        'base_smarts': base_smarts,
        'arrows': arrow_text,
        'operations': decoded,
    }


# ====================================================================
# Part 5 - 格式化输出
# ====================================================================

def format_answer(steps):
    """将 steps 列表格式化为 <answer>...</answer> 格式。"""
    parts = []
    for label, smarts in steps:
        parts.append(f'"{label}: {smarts}"')
    inner = ', '.join(parts)
    return f'<answer>{inner}</answer>'


def format_simple(steps):
    """简单的 label: smarts 格式，每步一行。"""
    lines = []
    for label, smarts in steps:
        lines.append(f'{label}: {smarts}')
    return '\n'.join(lines)


# ====================================================================
# Part 6 - CLI & 批量处理
# ====================================================================

def main():
    sys.stdout.reconfigure(encoding='utf-8')

    parser = argparse.ArgumentParser(
        description='Arrow descriptions + input SMARTS → step SMARTS (reverse of extract_arrows.py)')
    parser.add_argument('--smarts', type=str,
                        help='Input SMARTS string (single mode)')
    parser.add_argument('--arrows', type=str,
                        help='Arrow description string (single mode)')
    parser.add_argument('--json', type=str,
                        help='Input JSON for batch processing')
    parser.add_argument('--smarts-field', type=str, default='input_smarts',
                        help='JSON field containing input SMARTS (default: input_smarts)')
    parser.add_argument('--arrows-field', type=str, default=None,
                        help='JSON field with arrow string; default: from conversations[gpt]')
    parser.add_argument('--output', type=str,
                        help='Output JSON file (batch mode)')
    parser.add_argument('--format', type=str, default='simple',
                        choices=['simple', 'answer'],
                        help='Output format: simple or answer-tag')
    parser.add_argument('--decode', action='store_true',
                        help='Decode cumulative steps back to an ordered arrow list')
    parser.add_argument('--steps-json', type=str,
                        help='JSON object such as {"A": "...", "B": "..."}')
    args = parser.parse_args()

    if args.decode:
        if not args.smarts or not args.steps_json:
            parser.error('--decode requires --smarts and --steps-json')
        result = decode_steps(args.smarts, json.loads(args.steps_json))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    # ---- Single mode ----
    if args.smarts and args.arrows:
        steps = arrows_to_steps(args.smarts, args.arrows)
        print('Input SMARTS: ' + args.smarts)
        print('Arrows:       ' + args.arrows)
        print()
        if args.format == 'answer':
            print(format_answer(steps))
        else:
            print(format_simple(steps))
        return

    # ---- Batch mode ----
    if args.json:
        with open(args.json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        ok = err = skip = 0
        for item in data:
            smarts = item.get(args.smarts_field, '')
            if args.arrows_field:
                answer = item.get(args.arrows_field, '')
            else:
                answer = ''
                for conv in item.get('conversations', []):
                    if conv.get('from') == 'gpt':
                        answer = conv['value']
                        break

            if not smarts or not answer or answer.startswith('Error:'):
                item['step_smarts'] = None
                skip += 1
                continue

            try:
                steps = arrows_to_steps(smarts, answer)
                item['step_smarts'] = {label: s for label, s in steps}
                if args.format == 'answer':
                    item['step_smarts_str'] = format_answer(steps)
                ok += 1
            except Exception as e:
                item['step_smarts'] = None
                item['step_error'] = str(e)
                err += 1

        out_path = args.output or args.json.replace('.json', '_with_steps.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print('OK: ' + str(ok) + ', Error: ' + str(err) + ', Skip: ' + str(skip))
        print('Output: ' + out_path)
        return

    parser.print_help()


if __name__ == '__main__':
    main()
