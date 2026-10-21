from pathlib import Path
import subprocess,sys,json,shutil,secrets
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]

def _font(size):
    for candidate in ['C:/Windows/Fonts/arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','/Library/Fonts/Arial.ttf']:
        if Path(candidate).exists():return ImageFont.truetype(candidate,size)
    return ImageFont.load_default(size)

def test_runtime_in_isolated_folder_without_annotations_on_random_new_text(tmp_path):
    isolated=tmp_path/'standalone';isolated.mkdir()
    for name in ['autotess.py','ocr.py']:shutil.copy2(ROOT/name,isolated/name)
    token=''.join(str(secrets.randbelow(10))for _ in range(8))
    image=Image.new('RGB',(1100,260),'white');d=ImageDraw.Draw(image);font=_font(44)
    d.text((40,60),'NEW RANDOM '+token,font=font,fill='black')
    src=tmp_path/'never_seen_random.png';image.save(src)
    result=subprocess.run([sys.executable,'-E',str(isolated/'ocr.py'),'--input',str(src),'--output',str(tmp_path/'out'),'--models',str(ROOT/'models'),'--mode','accurate'],cwd=isolated,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    files=list((tmp_path/'out').glob('*.ocr.json'));assert len(files)==1
    data=json.loads(files[0].read_text('utf8'))
    assert token in data['text'],data['text']
    assert data['reference_data_used']is False
    assert not (isolated/'annotations').exists()
    for name in ['autotess.py','ocr.py']:
        code=(isolated/name).read_text('utf8')
        assert 'reviewed_samples'not in code and 'annotations/'not in code and 'photo_1_'not in code
