#!/usr/bin/env python3
"""Unlimited-OCR 报告配图：基准排名横向柱状图 + 子项能力雷达图"""
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/noto-serif-sc/NotoSerifSC-Regular.ttf')
fm.fontManager.addfont('/usr/share/fonts/truetype/noto-serif-sc/NotoSerifSC-Bold.ttf')
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Noto Serif SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ACCENT = '#2932E1'      # 主强调色（Unlimited OCR 2）
GRAY = '#B9BFD4'        # 其余模型
EDGE = '#1A1A2E'
OUT = '/home/z/my-project/download/charts'

import os
os.makedirs(OUT, exist_ok=True)

# ---------- 图 1：两大基准整体得分 ----------
mpdoc = [('Unlimited OCR 2', 84.59), ('Qwen3.8-Max', 78.30), ('Qwen-3.8-Flash-Next', 77.36),
         ('Gemini-3.8-Flash', 76.25), ('Gemini-3.1-Pro', 71.79), ('GLM-5.3-Flash', 68.02),
         ('DeepSeek-4.1-Flash', 66.25), ('Kimi-K3', 60.77)]
marathon = [('Unlimited OCR 2', 77.06), ('Kimi-K3', 33.74), ('Gemini-3.8-Flash', 28.61),
            ('Qwen-3.8-Flash-Next', 27.83), ('Qwen3.8-Max', 22.93), ('MiniMax-M3', 17.97),
            ('DeepSeek-4.1-Flash', 12.34), ('GLM-5.3-Flash', 7.51)]

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
for ax, data, title in zip(axes, [mpdoc, marathon],
                           ['MPDocBench（数十页级）综合得分', 'MarathonDocBench（百页级）综合得分']):
    names = [d[0] for d in data][::-1]
    vals = [d[1] for d in data][::-1]
    colors = [ACCENT if n == 'Unlimited OCR 2' else GRAY for n in names]
    bars = ax.barh(names, vals, color=colors, edgecolor=EDGE, linewidth=0.4, height=0.62)
    for b, v in zip(bars, vals):
        ax.text(v + 1.2, b.get_y() + b.get_height() / 2, f'{v:.1f}',
                va='center', ha='left', fontsize=9, color=EDGE)
    ax.set_title(title, fontsize=12, fontweight='bold', color=EDGE, pad=10)
    ax.set_xlim(0, 100)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='y', labelsize=9)
    ax.tick_params(axis='x', labelsize=8)
    ax.set_axisbelow(True)
    ax.grid(axis='x', color='#E3E6F0', linewidth=0.7)
fig.suptitle('Unlimited OCR 2 vs 主流模型：长文档解析基准得分（百度官方自报，2026-10）',
             fontsize=13, fontweight='bold', color=EDGE)
fig.savefig(f'{OUT}/benchmark_ranking.png', dpi=200, facecolor='white')
plt.close(fig)

# ---------- 图 2：子项能力雷达图 ----------
labels = ['文本内容', '公式还原', '阅读顺序', '表格结构', '标题层级']
mp = [94.8, 89.4, 92.3, 87.8, 60.1]
ma = [89.6, 87.9, 85.3, 74.8, 48.2]

angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]
mp += mp[:1]
ma += ma[:1]

fig = plt.figure(figsize=(7.2, 6.2), constrained_layout=True)
ax = fig.add_subplot(111, polar=True)
ax.plot(angles, mp, color=ACCENT, linewidth=2.2, label='MPDocBench（数十页级）')
ax.fill(angles, mp, color=ACCENT, alpha=0.18)
ax.plot(angles, ma, color='#E8590C', linewidth=2.2, linestyle='--', label='MarathonDocBench（百页级）')
ax.fill(angles, ma, color='#E8590C', alpha=0.12)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=11, color=EDGE)
ax.set_ylim(0, 115)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=8, color='#7A7F92')
ax.grid(color='#D8DCE8', linewidth=0.8)
ax.spines['polar'].set_color('#D8DCE8')
for ang, v in zip(angles[:-1], mp[:-1]):
    ax.text(ang, v + 9, f'{v:.1f}', ha='center', fontsize=8.5, color=ACCENT, fontweight='bold')
for ang, v in zip(angles[:-1], ma[:-1]):
    ax.text(ang, v - 11, f'{v:.1f}', ha='center', fontsize=8.5, color='#E8590C')
ax.set_title('Unlimited OCR 2 五维能力画像（标题层级为公开短板）',
             fontsize=12.5, fontweight='bold', color=EDGE, pad=22)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=10, frameon=False)
fig.savefig(f'{OUT}/radar_subscores.png', dpi=200, facecolor='white')
plt.close(fig)

print('charts done:', os.listdir(OUT))
