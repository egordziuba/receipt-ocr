from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]

def test_default_profile_does_not_enable_rejected_experiments(tmp_path):
    from autotess import recognize
    im=Image.new('RGB',(900,250),'white');d=ImageDraw.Draw(im)
    d.text((40,70),'Independent trial QV-538 7314.26',font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',38),fill='black')
    p=tmp_path/'fresh.png';im.save(p)
    r=recognize(p,ROOT/'models',mode='accurate')
    assert not any(x['variant'].endswith('_sauvola')for x in r['passes'])
    assert r['layout_order']=='fusion_geometry'
