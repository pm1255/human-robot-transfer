"""Deterministic pilot selection; never modifies source datasets."""
import json, random
from pathlib import Path
import pyarrow.parquet as pq

OUT = Path('/user/panmiao/human-robot-transfer/outputs/unified_30_20261021')
OUT.mkdir(parents=True, exist_ok=True)
def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))

ego = []
inventory = []
for part, count in [(4,3),(5,3),(6,2),(7,2)]:
    root = Path(f'/user/yangqinglin/egosuite_part_{part:02d}/lerobot_v3')
    info = json.loads((root/'meta/info.json').read_text())
    candidates = []
    for path in sorted((root/'meta/episodes').rglob('*.parquet')):
        columns = [c for c in pq.ParquetFile(path).schema_arrow.names if not c.startswith('stats/')]
        candidates.extend(pq.read_table(path, columns=columns).to_pylist())
    eligible = [r for r in candidates if 240 <= r['length'] <= 900]
    random.Random(20261021 + part).shuffle(eligible)
    tasks = set()
    picked = []
    for row in eligible:
        task = row['source_key'].rsplit('/',2)[-2]
        if task in tasks: continue
        if row['episode_index'] in [r['episode_index'] for r in picked]: continue
        tasks.add(task);picked.append(row)
        if len(picked) == count: break
    assert len(picked)==count
    inventory.append(dict(part=part, total_episodes=len(candidates), eligible_short_episodes=len(eligible), exported=info['total_episodes']))
    for row in picked:
        sid=f"egosuite_p{part:02d}_ep_{row['episode_index']:06d}"
        ego.append(dict(sample_id=sid,root=str(root),episode=row,info=info,task_name=row['source_key'].rsplit('/',2)[-2]))
        print(sid,row['length'],row['source_key'].rsplit('/',2)[-2],row.get('tasks'),flush=True)
dump(OUT/'ego_selection.json',ego)
dump(OUT/'ego_inventory.json',inventory)

pool=Path('/user/panmiao/workspace/robodojo-data-windtunnel/outputs/abc130k_less6000_20261014/delivery/clean_selected_500/episodes.jsonl')
rows=[json.loads(line) for line in pool.read_text().splitlines() if line.strip()]
print('ABC_KEYS',list(rows[0]),flush=True)
dump(OUT/'abc_pool.json',rows)
print('SELECTED_EGO',len(ego),flush=True)
