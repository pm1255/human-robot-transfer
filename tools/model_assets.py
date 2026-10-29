"""List, verify, or download the versioned assets used by the annotation code.

Downloading requires huggingface_hub. Listing and local verification use only
the standard library. This checks file integrity, not model accuracy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def check_asset(root, asset):
    path = root / asset['local_path']
    if not path.is_file():
        return 'manual_download_required' if asset['delivery'] == 'manual' else 'missing'
    if path.stat().st_size != asset['bytes']:
        return 'size_mismatch'
    return 'verified' if sha256(path) == asset['sha256'] else 'hash_mismatch'


def selected_assets(manifest, components):
    known = {row['component'] for row in manifest['assets']}
    unknown = set(components) - known
    if unknown:
        raise ValueError('Unknown components: ' + ', '.join(sorted(unknown)))
    return [row for row in manifest['assets'] if not components or row['component'] in components]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['list', 'check', 'download'])
    parser.add_argument('--manifest', type=Path, default=REPO / 'model_assets/manifest.json')
    parser.add_argument('--root', type=Path, default=REPO)
    parser.add_argument('--component', action='append', default=[])
    parser.add_argument('--hub-repo', default=None)
    parser.add_argument('--revision', default='main', help='Pin a published Hub commit for reproducibility')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows = selected_assets(manifest, args.component)
    root = args.root.expanduser().resolve()
    results = []
    for row in rows:
        if args.command == 'list':
            results.append(row)
            continue
        status = check_asset(root, row)
        if args.command == 'download' and status != 'verified' and row['delivery'] == 'mirror':
            from huggingface_hub import hf_hub_download
            import shutil
            cached = hf_hub_download(args.hub_repo or manifest['hub_repository'],
                                     row['hub_path'], revision=args.revision)
            if Path(cached).stat().st_size != row['bytes'] or sha256(cached) != row['sha256']:
                raise ValueError('Downloaded asset differs from manifest: ' + row['hub_path'])
            target = root / row['local_path']
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_name(target.name + '.download-partial')
            shutil.copyfile(cached, temp)
            temp.replace(target)
            status = check_asset(root, row)
        results.append(dict(component=row['component'], path=row['local_path'],
                            status=status, upstream=row['upstream']))
    output = dict(command=args.command, root=str(root), files=results,
                  scope='File integrity only; not an inference or accuracy test',
                  external_components=manifest['external_components'])
    text = json.dumps(output, ensure_ascii=False, indent=2) + '\n'
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)
    print(text, end='')
    if args.command != 'list' and any(row['status'] != 'verified' for row in results):
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
