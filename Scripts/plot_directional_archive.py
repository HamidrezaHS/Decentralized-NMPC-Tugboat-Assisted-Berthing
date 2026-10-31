"""Regenerate directional summary plots in results/ without altering archived raw data."""
from __future__ import annotations
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ENV_DIRECTION_REPORT import report

src=ROOT/'Data/RawRuns/environment_4x4_v31'
dst=ROOT/'results/archived_directional_report'
dst.mkdir(parents=True,exist_ok=True)
for name in ('summary.json','configuration.json','summary.csv'):
    if (src/name).exists():shutil.copy2(src/name,dst/name)
for source in src.glob('*/metrics.json'):
    target=dst/source.parent.name/'metrics.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)
report(dst)
print('Output:',dst)
