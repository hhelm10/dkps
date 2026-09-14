"""Sync the paper artifacts into artifacts/ under their paper-facing
names (per HH artifact flow: 1 hero table, 2 radars, 3 PKPS, 4 MAE
protocol grid, 5 cost, 6 rubric analysis).

Scripts keep writing to figures/; run this after any re-render so
artifacts/ stays the single uncrowded source of truth for the paper.
"""
import shutil

MANIFEST = {
    'figures/hero_table.png': 'artifacts/table1_hero.png',
    'figures/hero_table.md': 'artifacts/table1_hero.md',
    'notes/tables_hero.tex': 'artifacts/table1_hero.tex',
    'figures/radar_all.png': 'artifacts/fig1_radar.png',
    'figures/fig2_nm.png': 'artifacts/fig2_pkps.png',
    'figures/fig2_protocols.png': 'artifacts/fig3_protocols.png',
    'figures/fig_cost.png': 'artifacts/fig4_cost.png',
    'figures/fig_sensitivity_paper.png': 'artifacts/fig5_sensitivity.png',
}

for src, dst in MANIFEST.items():
    # copyfile, not copy2: a fresh mtime so the sync time is visible
    shutil.copyfile(src, dst)
    print(f'{src} -> {dst}')
