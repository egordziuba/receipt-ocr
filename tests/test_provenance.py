from pathlib import Path
import hashlib,json
import evaluate

def test_provenance_rejects_wrong_source_hash_and_mixed_run(tmp_path):
    assert hasattr(evaluate,'verify_provenance'),'Provenance validation missing'
    source=tmp_path/'input';source.mkdir();(source/'sample.png').write_bytes(b'actual-image-content')
    outputs=tmp_path/'out';outputs.mkdir()
    output={'source_files':['sample.png'],'input_sha256':'wrong','run_fingerprint':'other','words':[]}
    (outputs/'one.ocr.json').write_text(json.dumps(output),encoding='utf8')
    (outputs/'manifest.json').write_text(json.dumps({'unique_images':1,'run_fingerprint':'expected','results':[{'sha256':'wrong','files':['sample.png'],'result':'one.ocr.json','status':'unverified_ocr'}]}),encoding='utf8')
    p=evaluate.verify_provenance([{'file':'sample.png'}],outputs,source)
    assert not p['verified']
    assert any('hash' in e.lower() for e in p['errors'])
    assert any('fingerprint' in e.lower() for e in p['errors'])

def test_provenance_requires_finished_manifest(tmp_path):
    assert hasattr(evaluate,'verify_provenance')
    result=evaluate.verify_provenance([],tmp_path,tmp_path)
    assert not result['verified']
