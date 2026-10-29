"""Check score contracts, provenance, abstention and preserved source layers."""
import hashlib
import json
import shutil
import urllib.request
from collections import Counter
from pathlib import Path

BASE=Path('/Users/panmiao/Documents/ChatGPT/paper')
ROOT=BASE/'outputs/annotation_confidence_20261021'
SITE=BASE/'outputs/hand_mesh_repair_20261020'
MODEL=BASE/'outputs/unified_multimodal_20261020'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def valid_score(x):return x is None or isinstance(x,(int,float)) and 0<=x<=100

def main():
    manifest=json.loads((SITE/'assets/multi_manifest.json').read_text())
    before=json.loads((ROOT/'manifest_before_confidence.json').read_text())
    assert {c['id']:c['layers']for c in manifest['clips']}=={c['id']:c['layers']for c in before['clips']}
    total_frames=total_instances=unknown=0;statuses=Counter();audit=[]
    for c in manifest['clips']:
        if not c.get('confidence'):continue
        data=json.loads((SITE/c['confidence']).read_text());rows=data['frames']
        assert data['sample']==c['id'] and len(rows)==c['frames']
        assert not data['calibrated_probability'] and not data['training_eligible']
        assert data['source_sha256']==sha(SITE/'unified'/c['id']/'cam_0.mp4')
        assert data['conflict_record_sha256']==sha(MODEL/'visual_review.json')
        for f,h in data['input_hashes'].items():assert h==sha(MODEL/'samples'/c['id']/'multimodal'/f)
        originals=json.loads((MODEL/'samples'/c['id']/'multimodal/objects.json').read_text())['frames']
        for i,(r,original)in enumerate(zip(rows,originals)):
            assert r['frame_index']==i and abs(r['timestamp_s']-i/c['fps'])<1e-6
            assert len(r['instances'])==len(original['instances'])
            assert {o['class_id']for o in r['missing_classes']}==set(original['missing_classes'])
            for o in r['instances']:
                assert valid_score(o['temporal_score']) and o['probability'] is None
                assert not o['pseudo_label_candidate']
                if o['known_issue_codes']:assert o['status']=='known_issue'
                if o['temporal_score'] is None:unknown+=1
                statuses[o['status']]+=1;total_instances+=1
            for o in r['missing_classes']:assert o['score']is None and o['status']=='visibility_unknown'
            assert valid_score(r['depth']['stability_score']) and valid_score(r['flow_roundtrip_score'])
        assert rows[0]['depth']['stability_score']is None and rows[0]['flow_roundtrip_score']is None
        with urllib.request.urlopen('http://127.0.0.1:53419/'+c['confidence']) as response:
            assert response.status==200 and json.load(response)['sample']==c['id']
        total_frames+=len(rows);audit.append(dict(sample=c['id'],frames=len(rows),source_hash_checked=True,http_available=True))
    assert total_frames==2481 and total_instances==9757
    report=dict(status='passed_score_contracts_not_accuracy',frames=total_frames,instances=total_instances,
                unscored_instance_frames=unknown,statuses=dict(statuses),samples=audit,original_layers_unchanged=True,
                accuracy_verified=False,calibrated_probability=False)
    (ROOT/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    shutil.copy2(ROOT/'validation.json',SITE/'confidence/validation.json')
    print(json.dumps({k:v for k,v in report.items()if k!='samples'},ensure_ascii=False))

if __name__=='__main__':main()
