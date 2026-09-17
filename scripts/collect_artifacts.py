"""Sync the paper artifacts into artifacts/ under their paper-facing
names (per HH artifact flow: 1 hero table, 2 radars, 3 PKPS, 4 MAE
protocol grid, 5 cost, 6 rubric analysis).

Scripts keep writing to figures/; run this after any re-render so
artifacts/ stays the single uncrowded source of truth for the paper.
"""
import shutil

MANIFEST = {
    'figures/hero_table.md': 'artifacts/table1_hero.md',
    'notes/tables_hero.tex': 'artifacts/table1_hero.tex',
}
for src, dst in (
        ('figures/hero_table', 'artifacts/table1_hero'),
        ('figures/radar_all', 'artifacts/fig1_radar'),
        ('figures/fig2_nm', 'artifacts/fig2_pkps'),
        ('figures/fig2_protocols', 'artifacts/fig3_protocols'),
        ('figures/fig_ablations', 'artifacts/fig4_ablations'),
        ('figures/fig_cost', 'artifacts/fig5_cost'),
        ('figures/fig_sensitivity_paper', 'artifacts/fig6_sensitivity')):
    for ext in ('.png', '.pdf'):
        MANIFEST[src + ext] = dst + ext

for src, dst in MANIFEST.items():
    # copyfile, not copy2: a fresh mtime so the sync time is visible
    shutil.copyfile(src, dst)
    print(f'{src} -> {dst}')
