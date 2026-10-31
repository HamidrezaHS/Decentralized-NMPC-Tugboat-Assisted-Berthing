import json
from pathlib import Path
import pytest
from ENV_DIRECTION_REPORT import report


def test_report_requires_real_summary(tmp_path):
    with pytest.raises(FileNotFoundError):
        report(tmp_path)


def test_report_keeps_unexecuted_direction_cells(tmp_path):
    (tmp_path/'configuration.json').write_text(json.dumps({'wind_angles':[0,90], 'current_angles':[0,90]}),encoding='utf8')
    (tmp_path/'summary.json').write_text(json.dumps([dict(psi0=20,psi0_deg=20,wind_mps=8,wind_to_deg=90,
                                        current_mps=.25,current_to_deg=90,status='SUCCESS')]),encoding='utf8')
    counts=report(tmp_path)
    assert counts['SUCCESS']==1
    matrix=next(tmp_path.glob('angle_matrix_*.csv'))
    content=matrix.read_text(encoding='utf8')
    assert content.count('NOT TESTED')==3
    assert content.count('SUCCESS')==1
