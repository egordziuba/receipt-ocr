from pathlib import Path
import importlib.util
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]

def test_runtime_reads_unseen_printed_text_without_reference(tmp_path):
    spec=importlib.util.find_spec('autotess')
    assert spec is not None, 'Autonomous runtime does not exist yet'
    from autotess import recognize
    im=Image.new('RGB',(1000,260),'white'); d=ImageDraw.Draw(im)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',44)
    d.text((50,55),'Arbitrary sample XQ-729',font=font,fill='black')
    d.text((50,135),'8462.37  2031-11-24',font=font,fill='black')
    inp=tmp_path/'never_seen.png';im.save(inp)
    r=recognize(inp,ROOT/'models',mode='quick')
    text=r['text']
    assert '8462.37' in text,text
    assert 'XQ-729' in text,text
    assert r['engine'].startswith('tesseract')
    assert r['words'] and all('box' in x for x in r['words'])
    assert r['input_sha256']
