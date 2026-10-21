from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]

def test_accurate_pipeline_records_deskew_and_keeps_raw_candidate(tmp_path):
    from autotess import recognize
    im=Image.new('RGB',(900,360),'white');d=ImageDraw.Draw(im)
    f=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',34)
    for i in range(4):d.text((40,30+i*75),f'Unseen entry {i}: 7593.24',font=f,fill='black')
    im=im.rotate(5,expand=True,fillcolor='white');p=tmp_path/'tilt.png';im.save(p)
    r=recognize(p,ROOT/'models',mode='experimental')
    assert all('deskew_degrees' in x for x in r['regions'])
    assert any(x['variant']=='raw_gray' for x in r['passes'])
    assert '7593.24' in r['text']
    assert r.get('local_evidence'), 'No local multi-pass evidence in pipeline'
    assert any(x['variant'].startswith('auto_lines_') for x in r['passes'])
    assert any(x['variant']=='raw_gray_sauvola' for x in r['passes'])
    assert r.get('layout_order')=='upright_geometry'
    assert all(x.get('coordinate_space')=='original' for x in r['local_evidence'])
    assert all(0<=x['box'][0]<x['box'][2]<=im.width+1 and 0<=x['box'][1]<x['box'][3]<=im.height+1 for x in r['local_evidence'])
