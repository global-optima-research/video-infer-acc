#!/usr/bin/env python3
"""
Phase 0: Analyze attention profiling results.

Reads JSON output from phase0_attention_profiling.py and generates:
  - Per-property analysis and plots
  - Summary markdown report

Usage:
  python scripts/phase0_analyze.py results/phase0/attention_stats_XXXXXXXX_XXXX.json
"""

import json
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
from collections import defaultdict


def load_stats(path):
    with open(path) as f:
        return json.load(f)


def analyze_property_1(stats, config):
    """① Block diagonal dominance: same-frame vs cross-frame attention."""
    ratios_by_head = defaultdict(list)  # head -> list of ratios across layers/steps/prompts

    for p_idx, prompt_data in stats.items():
        for s_idx, step_data in prompt_data.items():
            for l_idx, layer_stats in step_data.items():
                if 'block_diag_ratio' not in layer_stats:
                    continue
                for h, ratio in enumerate(layer_stats['block_diag_ratio']):
                    ratios_by_head[h].append(ratio)

    if not ratios_by_head:
        return None

    # Aggregate: mean ratio per head, and overall
    head_means = {h: np.mean(v) for h, v in ratios_by_head.items()}
    overall_mean = np.mean([v for vals in ratios_by_head.values() for v in vals])
    overall_median = np.median([v for vals in ratios_by_head.values() for v in vals])

    return {
        'overall_mean_ratio': round(float(overall_mean), 3),
        'overall_median_ratio': round(float(overall_median), 3),
        'per_head_mean': {str(h): round(v, 3) for h, v in sorted(head_means.items())},
        'expected': '>2.5x (literature: 2.8x)',
        'verdict': 'CONFIRMED' if overall_mean > 2.0 else 'WEAK' if overall_mean > 1.5 else 'NOT CONFIRMED',
    }


def analyze_property_2(stats, config):
    """② Temporal locality decay: attention mass vs frame distance."""
    decay_by_dist = defaultdict(list)  # dist -> list of masses across heads/layers/prompts

    for p_idx, prompt_data in stats.items():
        for s_idx, step_data in prompt_data.items():
            for l_idx, layer_stats in step_data.items():
                if 'temporal_decay' not in layer_stats:
                    continue
                for dist_str, masses in layer_stats['temporal_decay'].items():
                    dist = int(dist_str)
                    for m in masses:
                        decay_by_dist[dist].append(m)

    if not decay_by_dist:
        return None

    decay_means = {d: float(np.mean(v)) for d, v in sorted(decay_by_dist.items())}

    # Check if monotonically decreasing
    dists = sorted(decay_means.keys())
    is_monotonic = all(
        decay_means[dists[i]] >= decay_means[dists[i+1]]
        for i in range(len(dists) - 1)
    )

    # Check distance-0 concentration
    total_mass = sum(decay_means.values())
    dist0_pct = decay_means.get(0, 0) / total_mass * 100 if total_mass > 0 else 0

    return {
        'decay_curve': {str(d): round(v, 6) for d, v in sorted(decay_means.items())},
        'is_monotonic_decay': is_monotonic,
        'dist0_concentration_pct': round(dist0_pct, 1),
        'expected': 'Monotonic decay, dist=0 should have highest mass',
        'verdict': 'CONFIRMED' if is_monotonic and dist0_pct > 30 else 'PARTIAL',
    }


def analyze_property_3(stats, config):
    """③ Cross-prompt invariance: top-k support set IoU across prompts."""
    # For each (layer, step), compare top-k indices across prompts
    prompt_indices = list(stats.keys())
    if len(prompt_indices) < 2:
        return {'verdict': 'INSUFFICIENT DATA (need >=2 prompts)'}

    # Method: compare per-head attention statistics across prompts
    # (top-k index IoU is unreliable because sampled queries differ across prompts)
    # Instead, compare block_diag_ratio and entropy profiles (input-independent metrics)
    correlations = []

    # For each (step, layer), compare per-head block_diag_ratio vectors across prompts
    for s_idx in stats[prompt_indices[0]]:
        for l_idx in stats[prompt_indices[0]].get(s_idx, {}):
            prompt_vectors = {}
            for p_idx in prompt_indices:
                ls = stats.get(p_idx, {}).get(s_idx, {}).get(l_idx, {})
                if 'block_diag_ratio' in ls and 'entropy_normalized' in ls:
                    # Use block_diag_ratio + entropy as feature vector per head
                    vec = ls['block_diag_ratio'] + ls['entropy_normalized']
                    prompt_vectors[p_idx] = np.array(vec)

            # Pairwise cosine similarity
            p_list = list(prompt_vectors.keys())
            for i in range(len(p_list)):
                for j in range(i + 1, len(p_list)):
                    v1 = prompt_vectors[p_list[i]]
                    v2 = prompt_vectors[p_list[j]]
                    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                    correlations.append(float(cos))

    if not correlations:
        return {'verdict': 'NO DATA'}

    mean_cos = float(np.mean(correlations))
    median_cos = float(np.median(correlations))

    return {
        'method': 'cosine similarity of per-head [block_diag_ratio, entropy] vectors across prompts',
        'mean_cosine_sim': round(mean_cos, 4),
        'median_cosine_sim': round(median_cos, 4),
        'num_comparisons': len(correlations),
        'expected': '>0.9 (literature: >0.9 index overlap)',
        'verdict': 'CONFIRMED' if mean_cos > 0.9 else 'WEAK' if mean_cos > 0.7 else 'NOT CONFIRMED',
    }


def analyze_property_4(stats, config):
    """④ Step-to-step stability: attention change across denoising steps."""
    # Collect step_rel_change and step_cosine_sim across layers
    changes_by_step = defaultdict(list)
    cos_by_step = defaultdict(list)

    for p_idx, prompt_data in stats.items():
        steps_sorted = sorted(prompt_data.keys(), key=int)
        for s_idx in steps_sorted:
            step_data = prompt_data[s_idx]
            for l_idx, layer_stats in step_data.items():
                if 'step_rel_change' in layer_stats:
                    changes_by_step[int(s_idx)].append(layer_stats['step_rel_change'])
                if 'step_cosine_sim' in layer_stats:
                    cos_by_step[int(s_idx)].append(layer_stats['step_cosine_sim'])

    if not changes_by_step:
        return {'verdict': 'NO DATA'}

    total_steps = config.get('num_steps', 50)
    early_boundary = int(total_steps * 0.15)
    late_boundary = int(total_steps * 0.85)

    early_changes = [v for s, vals in changes_by_step.items() for v in vals if s <= early_boundary]
    mid_changes = [v for s, vals in changes_by_step.items() for v in vals if early_boundary < s <= late_boundary]
    late_changes = [v for s, vals in changes_by_step.items() for v in vals if s > late_boundary]

    result = {
        'per_step_mean_change': {str(s): round(float(np.mean(v)), 6) for s, v in sorted(changes_by_step.items())},
        'per_step_mean_cosim': {str(s): round(float(np.mean(v)), 6) for s, v in sorted(cos_by_step.items())},
    }

    if early_changes:
        result['early_mean_change'] = round(float(np.mean(early_changes)), 6)
    if mid_changes:
        result['mid_mean_change'] = round(float(np.mean(mid_changes)), 6)
    if late_changes:
        result['late_mean_change'] = round(float(np.mean(late_changes)), 6)

    # Check U-shape: early > mid and late > mid
    is_u_shape = False
    if early_changes and mid_changes and late_changes:
        e, m, l = np.mean(early_changes), np.mean(mid_changes), np.mean(late_changes)
        is_u_shape = e > m and l > m
        result['u_shape_ratio'] = round(float(max(e, l) / (m + 1e-10)), 2)

    result['is_u_shape'] = is_u_shape
    result['expected'] = 'U-shape: high→low→high change, mid ~70% steps stable'
    result['verdict'] = 'CONFIRMED' if is_u_shape else 'NOT CONFIRMED'

    return result


def analyze_property_5(stats, config):
    """⑤ Head specialization: classify heads by attention pattern."""
    # Use block_diag_ratio and temporal_specificity to classify
    head_features = defaultdict(lambda: {'block_diag': [], 'temporal_spec': [], 'entropy': []})

    for p_idx, prompt_data in stats.items():
        for s_idx, step_data in prompt_data.items():
            for l_idx, layer_stats in step_data.items():
                n_heads = layer_stats.get('num_heads', 0)
                for h in range(n_heads):
                    key = f"L{l_idx}_H{h}"
                    if 'block_diag_ratio' in layer_stats:
                        head_features[key]['block_diag'].append(layer_stats['block_diag_ratio'][h])
                    if 'temporal_specificity' in layer_stats:
                        head_features[key]['temporal_spec'].append(layer_stats['temporal_specificity'][h])
                    if 'entropy_normalized' in layer_stats:
                        head_features[key]['entropy'].append(layer_stats['entropy_normalized'][h])

    if not head_features:
        return {'verdict': 'NO DATA'}

    # Classify heads
    classifications = defaultdict(int)
    head_types = {}

    for key, feats in head_features.items():
        bd = np.mean(feats['block_diag']) if feats['block_diag'] else 0
        ts = np.mean(feats['temporal_spec']) if feats['temporal_spec'] else 0
        ent = np.mean(feats['entropy']) if feats['entropy'] else 0

        if ent > 0.95:
            htype = 'global'
        elif bd > 5.0 and ts < 0.01:
            htype = 'spatial'
        elif ts > 0.05:
            htype = 'temporal'
        else:
            htype = 'mixed'

        classifications[htype] += 1
        head_types[key] = {
            'type': htype,
            'block_diag_ratio': round(bd, 2),
            'temporal_specificity': round(ts, 4),
            'entropy': round(ent, 4),
        }

    return {
        'classification_counts': dict(classifications),
        'total_heads': sum(classifications.values()),
        'head_types_sample': dict(list(head_types.items())[:20]),  # first 20 for brevity
        'expected': '3-4 distinct types (spatial, temporal, global, sink)',
        'verdict': 'CONFIRMED' if len(classifications) >= 3 else 'PARTIAL' if len(classifications) >= 2 else 'NOT CONFIRMED',
    }


def analyze_property_7(stats, config):
    """⑦ Entropy distribution: per-head normalized entropy."""
    entropies = defaultdict(list)  # head_key -> list of entropies

    for p_idx, prompt_data in stats.items():
        for s_idx, step_data in prompt_data.items():
            for l_idx, layer_stats in step_data.items():
                if 'entropy_normalized' not in layer_stats:
                    continue
                for h, ent in enumerate(layer_stats['entropy_normalized']):
                    entropies[f"L{l_idx}_H{h}"].append(ent)

    if not entropies:
        return {'verdict': 'NO DATA'}

    all_ent = [np.mean(v) for v in entropies.values()]

    # Check for bimodal distribution
    low_ent = sum(1 for e in all_ent if e < 0.5)
    high_ent = sum(1 for e in all_ent if e > 0.8)

    return {
        'mean_entropy': round(float(np.mean(all_ent)), 4),
        'std_entropy': round(float(np.std(all_ent)), 4),
        'min_entropy': round(float(np.min(all_ent)), 4),
        'max_entropy': round(float(np.max(all_ent)), 4),
        'low_entropy_heads_pct': round(low_ent / len(all_ent) * 100, 1),
        'high_entropy_heads_pct': round(high_ent / len(all_ent) * 100, 1),
        'expected': 'Bimodal: some near 0 (identity-like), some near 1 (uniform-like)',
        'verdict': 'CONFIRMED' if low_ent > 0 and high_ent > 0 else 'PARTIAL',
    }


def analyze_attention_concentration(stats, config):
    """Bonus: How much attention mass is in top-k% of tokens."""
    concentration = {}
    for pct in [1, 5, 10, 20]:
        key = f'top{pct}pct_mass'
        values = []
        for p_idx, prompt_data in stats.items():
            for s_idx, step_data in prompt_data.items():
                for l_idx, layer_stats in step_data.items():
                    if key in layer_stats:
                        values.extend(layer_stats[key])
        if values:
            concentration[f'top_{pct}pct'] = {
                'mean_mass': round(float(np.mean(values)), 4),
                'std': round(float(np.std(values)), 4),
            }

    return concentration


def generate_report(results, analysis, output_path):
    """Generate markdown experiment report."""
    config = results['config']
    env = results['env']

    report = f"""# 实验：Phase 0 注意力矩阵采集与分析
> 日期：{datetime.now().strftime("%Y-%m-%d %H:%M")}

## 环境
- Commit: `{env.get('git_commit', 'unknown')}`
- GPU: {env.get('gpu_name', 'unknown')} ({env.get('gpu_memory_gb', '?')} GB)
- Python: {env.get('python_version', '?')}
- PyTorch: {env.get('torch_version', '?')}
- CUDA: {env.get('cuda_version', '?')}
- diffusers: {env.get('diffusers_version', '?')}

## 运行命令
```bash
python scripts/phase0_attention_profiling.py \\
  --model {config.get('model', '?')} \\
  --height {config.get('height', '?')} \\
  --width {config.get('width', '?')} \\
  --num_frames {config.get('num_frames', '?')} \\
  --num_steps {config.get('num_steps', '?')} \\
  --num_prompts {config.get('num_prompts', '?')} \\
  --sample_queries {config.get('sample_queries', '?')} \\
  --seed {config.get('seed', '?')} \\
  --output_dir {config.get('output_dir', '?')}
```

## 参数
| 参数 | 值 |
|------|-----|
| 模型 | {config.get('model', '?')} |
| 分辨率 | {config.get('height', '?')} × {config.get('width', '?')} |
| 帧数 | {config.get('num_frames', '?')} |
| 去噪步数 | {config.get('num_steps', '?')} |
| Prompt 数 | {config.get('num_prompts', '?')} |
| 采样 Query 数 | {config.get('sample_queries', '?')} |
| Latent shape | {results.get('latent_shape', '?')} |
| Video tokens | {results.get('total_video_tokens', '?')} |
| 随机种子 | {config.get('seed', '?')} |
| 总耗时 | {results.get('total_time_seconds', '?')}s |

## 结果

### 性质 ① 块对角线主导

"""
    p1 = analysis.get('property_1')
    if p1:
        report += f"""**结论: {p1['verdict']}**

- 平均比率: **{p1['overall_mean_ratio']}x** (同帧平均注意力 / 跨帧平均注意力)
- 中位数: {p1['overall_median_ratio']}x
- 预期: {p1['expected']}

每头比率:
| Head | Ratio |
|------|-------|
"""
        for h, ratio in sorted(p1['per_head_mean'].items(), key=lambda x: int(x[0])):
            report += f"| {h} | {ratio} |\n"
    else:
        report += "无数据\n"

    report += "\n### 性质 ② 时间局部性衰减\n\n"
    p2 = analysis.get('property_2')
    if p2:
        report += f"""**结论: {p2['verdict']}**

- 单调递减: {'是' if p2['is_monotonic_decay'] else '否'}
- dist=0 注意力占比: **{p2['dist0_concentration_pct']}%**
- 预期: {p2['expected']}

衰减曲线:
| 帧距离 | 平均注意力 |
|--------|-----------|
"""
        for d, m in sorted(p2['decay_curve'].items(), key=lambda x: int(x[0])):
            report += f"| {d} | {m} |\n"
    else:
        report += "无数据\n"

    report += "\n### 性质 ③ 跨 Prompt 模式不变性\n\n"
    p3 = analysis.get('property_3')
    if p3:
        report += f"""**结论: {p3['verdict']}**

- 方法: {p3.get('method', 'N/A')}
- 平均余弦相似度: **{p3.get('mean_cosine_sim', 'N/A')}**
- 中位数: {p3.get('median_cosine_sim', 'N/A')}
- 比较次数: {p3.get('num_comparisons', 0)}
- 预期: {p3.get('expected', '')}
"""
    else:
        report += "无数据\n"

    report += "\n### 性质 ④ 去噪步间稳定性（U 形曲线）\n\n"
    p4 = analysis.get('property_4')
    if p4:
        report += f"""**结论: {p4['verdict']}**

- U 形: {'是' if p4.get('is_u_shape') else '否'}
"""
        if 'u_shape_ratio' in p4:
            report += f"- U 形比率 (边缘/中间): **{p4['u_shape_ratio']}x**\n"
        if 'early_mean_change' in p4:
            report += f"- 前期变化: {p4['early_mean_change']}\n"
        if 'mid_mean_change' in p4:
            report += f"- 中期变化: {p4['mid_mean_change']}\n"
        if 'late_mean_change' in p4:
            report += f"- 后期变化: {p4['late_mean_change']}\n"
        report += f"- 预期: {p4.get('expected', '')}\n"
    else:
        report += "无数据\n"

    report += "\n### 性质 ⑤ Head 功能特化\n\n"
    p5 = analysis.get('property_5')
    if p5:
        report += f"""**结论: {p5['verdict']}**

分类统计:
| 类型 | 数量 |
|------|------|
"""
        for t, c in sorted(p5.get('classification_counts', {}).items()):
            report += f"| {t} | {c} |\n"
        report += f"\n总头数: {p5.get('total_heads', 0)}\n"
        report += f"预期: {p5.get('expected', '')}\n"
    else:
        report += "无数据\n"

    report += "\n### 性质 ⑦ 注意力熵分布\n\n"
    p7 = analysis.get('property_7')
    if p7:
        report += f"""**结论: {p7['verdict']}**

| 指标 | 值 |
|------|-----|
| 平均熵 | {p7.get('mean_entropy', 'N/A')} |
| 标准差 | {p7.get('std_entropy', 'N/A')} |
| 最小熵 | {p7.get('min_entropy', 'N/A')} |
| 最大熵 | {p7.get('max_entropy', 'N/A')} |
| 低熵头占比 (<0.5) | {p7.get('low_entropy_heads_pct', 'N/A')}% |
| 高熵头占比 (>0.8) | {p7.get('high_entropy_heads_pct', 'N/A')}% |

预期: {p7.get('expected', '')}
"""
    else:
        report += "无数据\n"

    report += "\n### 注意力集中度\n\n"
    conc = analysis.get('concentration', {})
    if conc:
        report += "| Top-k% tokens | 覆盖注意力 |\n|------|------|\n"
        for k, v in sorted(conc.items()):
            report += f"| {k} | {v['mean_mass']:.4f} ± {v['std']:.4f} |\n"

    # ── Go/No-Go Decision ──
    report += "\n## Go/No-Go 决策\n\n"

    go_criteria = {
        '① 块对角主导': p1 and p1['verdict'] == 'CONFIRMED',
        '② 时间衰减': p2 and p2['verdict'] == 'CONFIRMED',
        '③ 跨 prompt 不变性': p3 and p3['verdict'] == 'CONFIRMED',
        '④ U 形稳定性': p4 and p4['verdict'] == 'CONFIRMED',
        '⑤ 头特化': p5 and p5['verdict'] in ('CONFIRMED', 'PARTIAL'),
        '⑦ 熵分布': p7 and p7['verdict'] in ('CONFIRMED', 'PARTIAL'),
    }

    report += "| 性质 | 状态 |\n|------|------|\n"
    for name, passed in go_criteria.items():
        status = 'PASS' if passed else 'FAIL'
        report += f"| {name} | {status} |\n"

    n_pass = sum(1 for v in go_criteria.values() if v)
    n_total = len(go_criteria)
    report += f"\n**通过 {n_pass}/{n_total}**\n\n"

    if n_pass >= 5:
        report += "**决策：GO** — 进入 Phase 1（增量计算可行性验证）\n"
    elif n_pass >= 3:
        report += "**决策：CONDITIONAL GO** — 部分性质未确认，需调整方案后继续\n"
    else:
        report += "**决策：NO GO** — 核心假设不成立，需重新评估方向\n"

    report += "\n## 下一步\n\n"
    report += "基于以上结果，建议的下一步行动...\n"

    with open(output_path, 'w') as f:
        f.write(report)

    return report


def main():
    parser = argparse.ArgumentParser(description="Analyze Phase 0 profiling results")
    parser.add_argument("stats_file", help="Path to attention_stats JSON file")
    parser.add_argument("--report_dir", default="experiments",
                        help="Directory to save experiment report")
    args = parser.parse_args()

    print(f"Loading {args.stats_file}...")
    results = load_stats(args.stats_file)
    stats = results['stats']
    config = results['config']

    print(f"Analyzing {len(stats)} prompts...")

    analysis = {}
    print("\n=== Property ① Block Diagonal Dominance ===")
    analysis['property_1'] = analyze_property_1(stats, config)
    print(json.dumps(analysis['property_1'], indent=2))

    print("\n=== Property ② Temporal Locality Decay ===")
    analysis['property_2'] = analyze_property_2(stats, config)
    print(json.dumps(analysis['property_2'], indent=2))

    print("\n=== Property ③ Cross-Prompt Invariance ===")
    analysis['property_3'] = analyze_property_3(stats, config)
    print(json.dumps(analysis['property_3'], indent=2))

    print("\n=== Property ④ Step-to-Step Stability ===")
    analysis['property_4'] = analyze_property_4(stats, config)
    if analysis['property_4']:
        # Print without per_step details (too verbose)
        p4_summary = {k: v for k, v in analysis['property_4'].items()
                      if k not in ('per_step_mean_change', 'per_step_mean_cosim')}
        print(json.dumps(p4_summary, indent=2, default=str))

    print("\n=== Property ⑤ Head Specialization ===")
    analysis['property_5'] = analyze_property_5(stats, config)
    if analysis['property_5']:
        p5_summary = {k: v for k, v in analysis['property_5'].items()
                      if k != 'head_types_sample'}
        print(json.dumps(p5_summary, indent=2))

    print("\n=== Property ⑦ Entropy Distribution ===")
    analysis['property_7'] = analyze_property_7(stats, config)
    print(json.dumps(analysis['property_7'], indent=2))

    print("\n=== Attention Concentration ===")
    analysis['concentration'] = analyze_attention_concentration(stats, config)
    print(json.dumps(analysis['concentration'], indent=2))

    # Generate report
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    report_path = report_dir / f"phase0_attention_profiling_{timestamp}.md"

    report = generate_report(results, analysis, report_path)
    print(f"\nReport saved to {report_path}")

    # Also save raw analysis as JSON
    analysis_path = report_path.with_suffix('.json')
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"Analysis JSON saved to {analysis_path}")


if __name__ == "__main__":
    main()
