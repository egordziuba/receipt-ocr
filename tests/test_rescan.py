from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import autotess
ROOT=Path(__file__).resolve().parents[1]

def test_automatic_line_mosaic_recognizes_pixels_not_seed_text():
    assert hasattr(autotess,'rescan_groups'),'Automatic group rescan missing'
    im=Image.new('L',(1100,320),'white');d=ImageDraw.Draw(im);f=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',30)
    d.text((30,30),'8592.74',font=f,fill=0);d.text((30,130),'Arbitrary ZM-426',font=f,fill=0)
    # Seed has only geometry. Deliberately wrong text must never be used as OCR answer.
    seed=[{'box':[28,28,170,70],'alternatives':[]},{'box':[28,128,350,175],'alternatives':[]}]
    words=autotess.rescan_groups(np.array(im),seed,ROOT/'models')
    text=' '.join(w['text']for w in words)
    assert '8592.74' in text,text
    assert 'ZM-426' in text,text
    assert any(w['box'][1]>100 for w in words)
