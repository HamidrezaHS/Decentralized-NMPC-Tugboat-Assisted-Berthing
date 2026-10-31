"""Validate the research data archive using only the Python standard library.

Archived SHA-256 values refer to the original bytes of each file. Some Git
configurations change CSV line endings (CRLF <-> LF) during checkout, although
all fields and numeric values remain identical. For CSV files only, this tool
also permits a *verified line-ending-only equivalent*: one of the two uniform
line-ending representations must match BOTH the original byte count and SHA-256.
Any content change, missing file, or modified numerical value still fails.

For new Git repositories, .gitattributes marks archived CSV files as binary to
avoid line-ending conversions. The extra check supports older checkouts that
were already normalized by Git before that attribute was introduced.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'Data'


def _matches_reference(data: bytes, expected_bytes: int, expected_sha256: str) -> bool:
    return (len(data) == expected_bytes and
            hashlib.sha256(data).hexdigest() == expected_sha256)


def verify_archive_bytes(data: bytes, expected_bytes: int, expected_sha256: str,
                         suffix: str) -> tuple[bool, str]:
    """Return (valid, comparison_mode) without changing any file on disk.

    Only CSV files may use line-ending equivalence; all other files require
    exact byte-for-byte equality with the published manifest. We never accept
    a changed numerical value or ignore an unknown byte difference.
    """
    if _matches_reference(data, expected_bytes, expected_sha256):
        return True, 'byte-exact'

    if suffix.lower() != '.csv':
        return False, 'mismatch'

    # Check that all CR bytes, if any, are part of CRLF sequences. Avoid
    # silently normalizing unusual or corrupt line-ending conventions.
    if b'\r' in data and data.count(b'\r') != data.count(b'\r\n'):
        return False, 'mismatch'

    # Normalizing to LF and then to CRLF covers common Windows/Unix Git
    # transformations without accepting arbitrary whitespace modifications.
    lf = data.replace(b'\r\n', b'\n')
    variants = (('LF', lf), ('CRLF', lf.replace(b'\n', b'\r\n')))
    for mode, candidate in variants:
        if candidate != data and _matches_reference(candidate, expected_bytes, expected_sha256):
            return True, f'line-ending-only ({mode} archive)'
    return False, 'mismatch'


def main() -> None:
    errors: list[str] = []
    manifest = DATA / 'SHA256_MANIFEST.csv'
    with manifest.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    line_endings_only = 0
    for row in rows:
        rel_path = row['path']
        p = ROOT / rel_path
        if not p.is_file():
            errors.append(f'MISSING {rel_path}')
            continue
        valid, mode = verify_archive_bytes(p.read_bytes(), int(row['bytes']),
                                           row['sha256'], p.suffix)
        if not valid:
            errors.append(f'HASH MISMATCH {rel_path}')
        elif mode != 'byte-exact':
            line_endings_only += 1

    groups = {'environment_4x4_v31': 16, 'v31_regression_suite': 6,
              'v31_heading40_combo': 1}
    with (DATA / 'RECORDED_CASES.csv').open(newline='', encoding='utf-8') as f:
        cases = list(csv.DictReader(f))
    if len(cases) != sum(groups.values()):
        errors.append(f'Expected {sum(groups.values())} case rows but found {len(cases)}')
    for group, n in groups.items():
        files = list((DATA / 'RawRuns' / group).glob('*/metrics.json'))
        if len(files) != n:
            errors.append(f'{group}: expected {n} metrics files, got {len(files)}')
        for mpath in files:
            metrics = json.loads(mpath.read_text(encoding='utf-8'))
            if metrics.get('status') != 'SUCCESS':
                errors.append(f'Unexpected status in {mpath}: {metrics.get("status")}')
            if not (mpath.parent / 'trajectory.csv').is_file():
                errors.append(f'Missing trajectory beside {mpath}')
        summary = DATA / 'RawRuns' / group / 'summary.json'
        if not summary.exists() or len(json.loads(summary.read_text(encoding='utf-8'))) != n:
            errors.append(f'Incorrect summary for {group}')

    print(f'Integrity: {len(rows)} archived data files checked; {len(cases)} cases indexed.')
    if line_endings_only:
        print(f'INFO: {line_endings_only} CSV files match the archived SHA-256 '
              'after strictly verified CRLF/LF line-ending conversion.')
    if errors:
        for error in errors:
            print('ERROR:', error)
        raise SystemExit(1)
    print('PASS: all archived files match their recorded content, and all '
          '23 simulation cases are present.')


if __name__ == '__main__':
    main()
