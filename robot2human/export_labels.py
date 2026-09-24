import pathlib,json
r=pathlib.Path('/user/panmiao/workspace/robot2human-vace-20260924');v=pathlib.Path('/user/panmiao/workspace/human2robot-wan-v4-20260924')
m=json.loads((v/'manifests/bridge.json').read_text())['train']
for clip in json.loads((r/'inputs/manifest.json').read_text()):
 rows=[x for x in m if x['episode']==clip['id'] and x['t']/clip['fps']<clip['duration']]
 data={'episode':clip['id'],'provenance':'Unmodified v4 Bridge manifest rows; source robot labels','semantics':'Realized TCP motion in local frame, not native command actions; synthetic human video is not geometrically verified','robot_labels_unchanged':True,'human_alignment_verified':False,'source_fps':clip['fps'],'clip_duration':clip['duration'],'samples':rows}
 (r/'inputs'/clip['id']/'robot_labels.json').write_text(json.dumps(data,indent=2));print(clip['id'],len(rows),flush=True)
