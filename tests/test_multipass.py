from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]

def test_accurate_mode_uses_multiple_independent_image_passes(tmp_path):
    from autotess import recognize
    im=Image.new('RGB',(900,330),'white');d=ImageDraw.Draw(im);f=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',40)
    for i,t in enumerate(['Unknown invoice QX-813','Opaque reference 5638.19','Date 2032-04-17']):d.text((35,30+i*80),t,font=f,fill='black')
    p=tmp_path/'novel.png';im.save(p);r=recognize(p,ROOT/'models',mode='accurate')
    assert len(r.get('passes',[]))>=4,'No multi-pass OCR evidence'
    assert '5638.19' in r['text']
    assert r.get('regions')
    assert r.get('reference_data_used') is False
