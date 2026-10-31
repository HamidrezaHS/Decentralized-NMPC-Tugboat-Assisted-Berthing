"""Verify that end-of-line tolerance never masks modified scientific results."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Scripts'))
from verify_repository import verify_archive_bytes


def test_reference_exact_bytes():
    data = b'time_s,force_N\r\n0,12.5\r\n1,15.0\r\n'
    ref = hashlib.sha256(data).hexdigest()
    assert verify_archive_bytes(data, len(data), ref, '.csv') == (True, 'byte-exact')


def test_git_converted_lf_is_accepted_only_for_csv():
    original = b'time_s,force_N\r\n0,12.5\r\n1,15.0\r\n'
    checked_out = original.replace(b'\r\n', b'\n')
    ref = hashlib.sha256(original).hexdigest()
    assert verify_archive_bytes(checked_out, len(original), ref, '.csv')[0]
    assert not verify_archive_bytes(checked_out, len(original), ref, '.json')[0]


def test_real_value_change_is_rejected():
    original = b'time_s,force_N\r\n0,12.5\r\n1,15.0\r\n'
    ref = hashlib.sha256(original).hexdigest()
    altered = original.replace(b'15.0', b'15.1').replace(b'\r\n', b'\n')
    assert not verify_archive_bytes(altered, len(original), ref, '.csv')[0]


def test_unusual_carriage_return_is_rejected():
    original = b'a,b\r\n1,2\r\n'
    ref = hashlib.sha256(original).hexdigest()
    altered = b'a,b\r1,2\n'
    assert not verify_archive_bytes(altered, len(original), ref, '.csv')[0]
