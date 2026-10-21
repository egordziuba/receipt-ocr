from pathlib import Path
import subprocess,sys,json
ROOT=Path(__file__).resolve().parents[1]

def test_accuracy_gate_returns_failure_for_bad_recognition(tmp_path):
    annotations=tmp_path/'gold';annotations.mkdir();results=tmp_path/'results';results.mkdir()
    (annotations/'test.json').write_text(json.dumps({'images':[{'file':'test.png','category':'synthetic','observations':[{'id':'x','box':[0,0,100,100],'text':'Unknown','kind':'text','legible':True}]}]}),encoding='utf8')
    (results/'one.ocr.json').write_text(json.dumps({'source_files':['test.png'],'words':[]}),encoding='utf8')
    r=subprocess.run([sys.executable,str(ROOT/'evaluate.py'),'--annotations',str(annotations),'--results',str(results),'--output',str(tmp_path/'report.json'),'--require-character-accuracy','95'],capture_output=True,text=True)
    assert r.returncode==1,(r.returncode,r.stderr)
    report=json.loads((tmp_path/'report.json').read_text('utf8'))
    assert report['acceptance']['passed']is False
