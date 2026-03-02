"""
训练数据分布分析脚本
分析 train.jsonl 中的 step_index, total_steps, tactic 分布特征
@author ygw
"""

import json
import sys
import io
from collections import Counter, defaultdict

# 修复 Windows 下 GBK 编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

DATA_PATH = r"D:\RTAP\data\processed\cos_dataset\train.jsonl"

def extract_tactic_name(tactic_str):
    """从 tactic 字符串中提取主要的 tactic 名称（第一个词）"""
    if not tactic_str:
        return "<empty>"
    # 取第一个空白分隔的 token
    first_token = tactic_str.strip().split()[0] if tactic_str.strip() else "<empty>"
    # 去掉可能的前缀符号如 · 或 <;>
    if first_token in ('·', '<;>', '--', '|'):
        parts = tactic_str.strip().split()
        if len(parts) > 1:
            first_token = parts[1]
    return first_token

def main():
    print("=" * 80)
    print("训练数据分布分析")
    print("=" * 80)

    # 1. 加载数据
    samples = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    total_samples = len(samples)
    print(f"\n总样本数: {total_samples}")

    # 区分 original 和 error_injection
    original_samples = [s for s in samples if s.get('source') == 'original']
    error_samples = [s for s in samples if s.get('source') == 'error_injection']
    print(f"  original 样本数: {len(original_samples)}")
    print(f"  error_injection 样本数: {len(error_samples)}")

    # =========================================================================
    # 2. total_steps 分布（仅 original 样本有 total_steps）
    # =========================================================================
    print("\n" + "=" * 80)
    print("一、total_steps 分布（仅 original 样本）")
    print("=" * 80)

    total_steps_counter = Counter()
    for s in original_samples:
        ts = s.get('total_steps')
        if ts is not None:
            total_steps_counter[ts] += 1

    # 也检查 error_injection 样本是否有 total_steps
    error_with_ts = sum(1 for s in error_samples if s.get('total_steps') is not None)
    print(f"  (error_injection 中有 total_steps 的样本数: {error_with_ts})")

    # 合并统计
    all_steps_counter = Counter()
    for s in samples:
        ts = s.get('total_steps')
        if ts is not None:
            all_steps_counter[ts] += 1

    no_ts_count = sum(1 for s in samples if s.get('total_steps') is None)
    print(f"  无 total_steps 的样本数: {no_ts_count}")

    print(f"\ntotal_steps 分布直方图（所有有 total_steps 的样本）:")
    print(f"{'total_steps':>12} | {'count':>8} | {'占比':>8} | 直方图")
    print("-" * 70)
    for steps in sorted(all_steps_counter.keys()):
        count = all_steps_counter[steps]
        pct = count / sum(all_steps_counter.values()) * 100
        bar = '#' * max(1, int(pct))
        print(f"{steps:>12} | {count:>8} | {pct:>6.1f}% | {bar}")

    if all_steps_counter:
        vals = []
        for steps, count in all_steps_counter.items():
            vals.extend([steps] * count)
        vals.sort()
        avg = sum(vals) / len(vals)
        median = vals[len(vals) // 2]
        print(f"\n  平均 total_steps: {avg:.2f}")
        print(f"  中位数 total_steps: {median}")
        print(f"  最小值: {min(vals)}, 最大值: {max(vals)}")

    # =========================================================================
    # 3. step_index 分布
    # =========================================================================
    print("\n" + "=" * 80)
    print("二、step_index 分布")
    print("=" * 80)

    step_index_counter = Counter()
    for s in samples:
        si = s.get('step_index')
        if si is not None:
            step_index_counter[si] += 1

    print(f"\nstep_index 分布:")
    print(f"{'step_index':>12} | {'count':>8} | {'占比':>8}")
    print("-" * 40)
    for si in sorted(step_index_counter.keys()):
        count = step_index_counter[si]
        pct = count / total_samples * 100
        print(f"{si:>12} | {count:>8} | {pct:>6.1f}%")

    # =========================================================================
    # 4. 第一步 (step_index=0) 的 tactic 分布
    # =========================================================================
    print("\n" + "=" * 80)
    print("三、第一步 (step_index=0) 的 tactic 分布")
    print("=" * 80)

    first_step_tactics = Counter()
    first_step_raw = []
    for s in samples:
        si = s.get('step_index')
        if si == 0:
            # 对于 original 样本用 tactic 字段，对于 error_injection 用 original_tactic
            tactic = s.get('tactic') or s.get('original_tactic', '')
            tname = extract_tactic_name(tactic)
            first_step_tactics[tname] += 1
            first_step_raw.append(tactic)

    print(f"\n第一步 (step_index=0) 样本总数: {sum(first_step_tactics.values())}")
    print(f"\nTop 30 tactic 名称:")
    print(f"{'tactic':>30} | {'count':>8} | {'占比':>8}")
    print("-" * 55)
    for tname, count in first_step_tactics.most_common(30):
        pct = count / sum(first_step_tactics.values()) * 100
        print(f"{tname:>30} | {count:>8} | {pct:>6.1f}%")

    # =========================================================================
    # 5. 最后一步 (step_index == total_steps - 1) 的 tactic 分布
    # =========================================================================
    print("\n" + "=" * 80)
    print("四、最后一步 (step_index == total_steps - 1) 的 tactic 分布")
    print("=" * 80)

    last_step_tactics = Counter()
    last_step_count = 0
    for s in samples:
        si = s.get('step_index')
        ts = s.get('total_steps')
        if si is not None and ts is not None and si == ts - 1:
            tactic = s.get('tactic') or s.get('original_tactic', '')
            tname = extract_tactic_name(tactic)
            last_step_tactics[tname] += 1
            last_step_count += 1

    print(f"\n最后一步样本总数: {last_step_count}")
    print(f"\nTop 30 tactic 名称:")
    print(f"{'tactic':>30} | {'count':>8} | {'占比':>8}")
    print("-" * 55)
    for tname, count in last_step_tactics.most_common(30):
        pct = count / last_step_count * 100 if last_step_count > 0 else 0
        print(f"{tname:>30} | {count:>8} | {pct:>6.1f}%")

    # 统计 "一击必杀" 类策略占比
    killer_tactics = {'simp', 'ring', 'norm_num', 'omega', 'decide', 'trivial', 'tauto',
                      'aesop', 'linarith', 'nlinarith', 'positivity', 'norm_cast',
                      'ring_nf', 'field_simp', 'contradiction', 'absurd', 'exact', 'rfl',
                      'assumption', 'grind'}
    killer_count = sum(count for tname, count in last_step_tactics.items() if tname in killer_tactics)
    print(f"\n'一击必杀'类策略 (simp/ring/norm_num/omega/decide/exact/rfl/assumption/grind等) 占比: "
          f"{killer_count}/{last_step_count} = {killer_count/last_step_count*100:.1f}%" if last_step_count > 0 else "无数据")

    # =========================================================================
    # 5.5 补充：state_after == "no goals" 的样本（真正的最后一步）
    # =========================================================================
    print("\n" + "=" * 80)
    print("四-b、state_after == 'no goals' 的样本的 tactic 分布（真正完成证明的步骤）")
    print("=" * 80)

    no_goals_tactics = Counter()
    no_goals_count = 0
    for s in samples:
        sa = s.get('state_after', '')
        if sa.strip() == 'no goals':
            tactic = s.get('tactic') or s.get('original_tactic', '')
            tname = extract_tactic_name(tactic)
            no_goals_tactics[tname] += 1
            no_goals_count += 1

    print(f"\nstate_after == 'no goals' 的样本总数: {no_goals_count}")
    print(f"\nTop 30 tactic 名称:")
    print(f"{'tactic':>30} | {'count':>8} | {'占比':>8}")
    print("-" * 55)
    for tname, count in no_goals_tactics.most_common(30):
        pct = count / no_goals_count * 100 if no_goals_count > 0 else 0
        print(f"{tname:>30} | {count:>8} | {pct:>6.1f}%")

    if no_goals_count > 0:
        killer_count_ng = sum(count for tname, count in no_goals_tactics.items() if tname in killer_tactics)
        print(f"\n'一击必杀'类策略占比: {killer_count_ng}/{no_goals_count} = {killer_count_ng/no_goals_count*100:.1f}%")

    # =========================================================================
    # 6. 中间步 tactic 分布
    # =========================================================================
    print("\n" + "=" * 80)
    print("五、中间步 (非第一步且非最后一步) 的 tactic 分布")
    print("=" * 80)

    mid_step_tactics = Counter()
    mid_step_count = 0
    for s in samples:
        si = s.get('step_index')
        ts = s.get('total_steps')
        if si is not None and ts is not None:
            if si > 0 and si < ts - 1:
                tactic = s.get('tactic') or s.get('original_tactic', '')
                tname = extract_tactic_name(tactic)
                mid_step_tactics[tname] += 1
                mid_step_count += 1

    print(f"\n中间步样本总数: {mid_step_count}")
    print(f"\nTop 30 tactic 名称:")
    print(f"{'tactic':>30} | {'count':>8} | {'占比':>8}")
    print("-" * 55)
    for tname, count in mid_step_tactics.most_common(30):
        pct = count / mid_step_count * 100 if mid_step_count > 0 else 0
        print(f"{tname:>30} | {count:>8} | {pct:>6.1f}%")

    # =========================================================================
    # 7. 各 step_index 位置的 tactic 类型 Top 5
    # =========================================================================
    print("\n" + "=" * 80)
    print("六、各 step_index 位置的 tactic 类型 Top 5")
    print("=" * 80)

    step_tactic_map = defaultdict(Counter)
    for s in samples:
        si = s.get('step_index')
        if si is not None:
            tactic = s.get('tactic') or s.get('original_tactic', '')
            tname = extract_tactic_name(tactic)
            step_tactic_map[si][tname] += 1

    for si in sorted(step_tactic_map.keys()):
        total_at_step = sum(step_tactic_map[si].values())
        top5 = step_tactic_map[si].most_common(5)
        top5_str = ", ".join([f"{t}({c}, {c/total_at_step*100:.0f}%)" for t, c in top5])
        print(f"  step_index={si:>2} (n={total_at_step:>5}): {top5_str}")

    # =========================================================================
    # 8. 前 20 条完整证明的 step_index 和 tactic 序列
    # =========================================================================
    print("\n" + "=" * 80)
    print("七、前 20 条完整证明的 step_index 和 tactic 序列")
    print("=" * 80)

    # 按 theorem_full_name 分组，仅看 original 样本
    proofs = defaultdict(list)
    for s in original_samples:
        tname = s.get('theorem_full_name', s.get('theorem_name', 'unknown'))
        proofs[tname].append(s)

    # 排序每个证明内的步骤
    shown = 0
    for theorem_name in proofs:
        if shown >= 20:
            break
        steps = sorted(proofs[theorem_name], key=lambda x: x.get('step_index', 0))
        total_steps = steps[0].get('total_steps', '?')
        print(f"\n证明 #{shown+1}: {theorem_name} (total_steps={total_steps})")
        for step in steps:
            si = step.get('step_index', '?')
            tactic = step.get('tactic', '')
            # 截断过长的 tactic
            if len(tactic) > 100:
                tactic = tactic[:100] + "..."
            sa = step.get('state_after', '')
            is_done = " [DONE]" if sa.strip() == "no goals" else ""
            print(f"    step {si}: {tactic}{is_done}")
        shown += 1

    # =========================================================================
    # 9. tactic 长度/复杂度分析
    # =========================================================================
    print("\n" + "=" * 80)
    print("八、tactic 字符长度分布（反映复杂度）")
    print("=" * 80)

    # 按 step position 分类
    positions = {'first': [], 'middle': [], 'last': [], 'unknown': []}
    for s in samples:
        si = s.get('step_index')
        ts = s.get('total_steps')
        tactic = s.get('tactic') or s.get('original_tactic', '')
        tac_len = len(tactic)

        if si is not None and ts is not None:
            if si == 0:
                positions['first'].append(tac_len)
            elif si == ts - 1:
                positions['last'].append(tac_len)
            else:
                positions['middle'].append(tac_len)
        else:
            positions['unknown'].append(tac_len)

    for pos_name, lengths in positions.items():
        if lengths:
            avg_len = sum(lengths) / len(lengths)
            sorted_lens = sorted(lengths)
            median_len = sorted_lens[len(sorted_lens) // 2]
            max_len = max(lengths)
            min_len = min(lengths)
            print(f"\n  {pos_name:>8} 步 (n={len(lengths):>5}):")
            print(f"    平均长度: {avg_len:.1f} 字符")
            print(f"    中位数: {median_len} 字符")
            print(f"    最小: {min_len}, 最大: {max_len}")

    # =========================================================================
    # 10. error_injection 样本的 error_type 分布
    # =========================================================================
    print("\n" + "=" * 80)
    print("九、error_injection 样本的 error_type 分布")
    print("=" * 80)

    error_type_counter = Counter()
    for s in error_samples:
        et = s.get('error_type', 'unknown')
        error_type_counter[et] += 1

    print(f"\n{'error_type':>20} | {'count':>8} | {'占比':>8}")
    print("-" * 45)
    for et, count in error_type_counter.most_common():
        pct = count / len(error_samples) * 100 if error_samples else 0
        print(f"{et:>20} | {count:>8} | {pct:>6.1f}%")

    print("\n" + "=" * 80)
    print("分析完成")
    print("=" * 80)

if __name__ == '__main__':
    main()
