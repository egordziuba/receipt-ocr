from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]

def test_rotated_input_detected_and_boxes_return_original_coordinates(tmp_path):
    from autotess import recognize
    image=Image.new('RGB',(1100,500),'white');d=ImageDraw.Draw(image)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',38)
    for i in range(6):d.text((40,30+i*68),f'Independent document line {i} 7284.63',font=font,fill='black')
    image=image.rotate(90,expand=True);p=tmp_path/'rotated.png';image.save(p)
    r=recognize(p,ROOT/'models',mode='accurate')
    assert '7284.63' in r['text'],r['text']
    assert r.get('rotation') in (90,270),r.get('rotation')
    for w in r['words']:
        x1,y1,x2,y2=w['box']
        assert 0<=x1<x2<=image.width+1
        assert 0<=y1<y2<=image.height+1
