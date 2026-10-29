"""Audit or publish the manifest's exact model files to the companion Hub repo.

Default mode is read-only: compare existing Hub objects with the manifest.
--publish uploads missing/mismatched objects after verifying every local file.
--make-public explicitly changes visibility after the upload has succeeded.
Authentication uses HF_TOKEN or the standard Hugging Face login store.
"""
import argparse
import json
from pathlib import Path

from model_assets import REPO, sha256


def source_path(asset, root, sam_root):
    if asset['component'] == 'sam31' and sam_root:
        return sam_root / Path(asset['local_path']).name
    return root / asset['local_path']


def remote_matches(asset, remote):
    if remote is None or remote.size != asset['bytes']:
        return False
    lfs = getattr(remote, 'lfs', None)
    digest = lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)
    return digest == asset['sha256']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=REPO, help='Existing model files installed here')
    parser.add_argument('--sam-root', type=Path, help='Optional existing SAM 3.1 directory')
    parser.add_argument('--repo-id', default='erwtwert/embodied-data-loop-models')
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--make-public', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.make_public and not args.publish:
        parser.error('--make-public requires --publish')
    from huggingface_hub import HfApi, CommitOperationAdd, get_token
    token = get_token()
    if not token:
        raise RuntimeError('Hugging Face login is required; configure HF_TOKEN or run hf auth login. Do not put a token in this script.')
    api = HfApi(token=token)
    # Verify the named existing repository before any mutation. Do not silently
    # create a new repository when a name or authorization is incorrect.
    info = api.model_info(args.repo_id, files_metadata=True)
    existing = {item.rfilename: item for item in info.siblings}
    manifest = json.loads((REPO / 'model_assets/manifest.json').read_text())
    manifest['hub_repository'] = args.repo_id
    assets = [row for row in manifest['assets'] if row['delivery'] == 'mirror']
    rows = []
    operations = []
    root = args.root.expanduser().resolve()
    sam_root = args.sam_root.expanduser().resolve() if args.sam_root else None
    for asset in assets:
        item = existing.get(asset['hub_path'])
        matches = remote_matches(asset, item)
        # Small Git objects have no LFS hash; compare their real content.
        if item is not None and not getattr(item, 'lfs', None) and item.size == asset['bytes']:
            from huggingface_hub import hf_hub_download
            cached = hf_hub_download(args.repo_id, asset['hub_path'], revision=info.sha, token=token)
            matches = sha256(cached) == asset['sha256']
        rows.append(dict(path=asset['hub_path'], remote_status='verified' if matches else 'missing_or_different'))
        if args.publish:
            source = source_path(asset, root, sam_root)
            if not source.is_file() or source.stat().st_size != asset['bytes'] or sha256(source) != asset['sha256']:
                raise ValueError('Missing or changed local model asset: ' + str(source))
            if not matches:
                operations.append(CommitOperationAdd(path_in_repo=asset['hub_path'], path_or_fileobj=source))
    result = dict(repository=args.repo_id, private_before=info.private, revision_before=info.sha,
                  required_files=rows, existing_files=sorted(existing), excluded=['MANO_RIGHT.pkl'],
                  published=False)
    # Save the observed state before uploading so interrupted releases can resume.
    def report():
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    report()
    if args.publish:
        card = (REPO / 'model_assets/MODEL_CARD.md').read_bytes()
        if 'README.md' in existing:
            from huggingface_hub import hf_hub_download
            old = Path(hf_hub_download(args.repo_id, 'README.md', revision=info.sha, token=token)).read_bytes()
            if old != card:
                operations.append(CommitOperationAdd(path_in_repo=f'archive/README-before-{info.sha}.md', path_or_fileobj=old))
        operations.extend([
            CommitOperationAdd(path_in_repo='README.md', path_or_fileobj=card),
            CommitOperationAdd(path_in_repo='manifest.json', path_or_fileobj=(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode()),
        ])
        for path in sorted((REPO / 'model_assets/licenses').glob('*.txt')):
            operations.append(CommitOperationAdd(path_in_repo='licenses/' + path.name, path_or_fileobj=path))
        commit = api.create_commit(repo_id=args.repo_id, operations=operations,
                                   parent_commit=info.sha,
                                   commit_message='Publish annotation model assets with hashes, licenses, and GitHub script links')
        result['revision_after'] = commit.oid
        result['published'] = True
        report()
        # Check the committed object metadata before widening visibility.
        after = api.model_info(args.repo_id, revision=commit.oid, files_metadata=True)
        indexed = {item.rfilename: item for item in after.siblings}
        for asset in assets:
            item = indexed.get(asset['hub_path'])
            if item is None or item.size != asset['bytes']:
                raise RuntimeError('Upload verification failed: ' + asset['hub_path'])
            if getattr(item, 'lfs', None) and not remote_matches(asset, item):
                raise RuntimeError('Upload hash mismatch: ' + asset['hub_path'])
        if args.make_public:
            api.update_repo_settings(repo_id=args.repo_id, private=False)
        final = api.model_info(args.repo_id)
        result['private_after'] = final.private
        if args.make_public:
            public = HfApi(token=False).model_info(args.repo_id)
            if public.private:
                raise RuntimeError('Anonymous public verification failed')
            result['anonymous_access_verified'] = True
        report()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
