from pathlib import Path
import subprocess,sys,json
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]

def test_cli_duplicate_images_and_new_name_do_not_duplicate_results(tmp_path):
    assert (ROOT/'ocr.py').exists(),'No standalone command'
    source=tmp_path/'in';source.mkdir();out=tmp_path/'out'
    im=Image.new('RGB',(800,220),'white');ImageDraw.Draw(im).text((35,70),'Test 3194.82',font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',42),fill='black')
    im.save(source/'one.png');(source/'renamed.png').write_bytes((source/'one.png').read_bytes())
    cmd=[sys.executable,str(ROOT/'ocr.py'),'--input',str(source),'--output',str(out),'--mode','quick','--models',str(ROOT/'models')]
    r=subprocess.run(cmd,capture_output=True,text=True);assert r.returncode==0,r.stderr
    manifest=json.loads((out/'manifest.json').read_text('utf8'))
    assert manifest['unique_images']==1 and manifest['input_files']==2
    assert len(list(out.glob('*.ocr.json')))==1
    r=subprocess.run(cmd,capture_output=True,text=True);assert r.returncode==0,r.stderr
    assert len(list(out.glob('*.ocr.json')))==1
