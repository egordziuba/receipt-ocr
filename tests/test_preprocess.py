import numpy as np
from PIL import Image,ImageDraw,ImageFont
import autotess

def test_generic_preprocessing_suppresses_screen_stripes_and_preserves_text():
    assert hasattr(autotess,'preprocess'),'Preprocessing missing'
    image=Image.new('RGB',(700,260),'white');d=ImageDraw.Draw(image)
    d.text((30,90),'Document 9276.48',font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',34),fill='black')
    a=np.asarray(image).astype(float)
    a-=np.tile((np.arange(700)%3==0)*55,(260,1))[:,:,None]
    a=np.clip(a,0,255).astype('uint8')
    variants=autotess.preprocess(a)
    assert len(variants)>=2
    assert all(im.ndim==2 for name,im,scale in variants)
    gray=variants[0][1]
    assert gray.shape[0]>260
    assert np.percentile(gray,10)>np.min(gray)+30


def test_broken_stroke_variant_is_available_without_text_templates():
    image=np.full((250,600,3),255,np.uint8)
    image[100:104,60:110]=0;image[108:112,60:110]=0
    variants=autotess.preprocess(image)
    assert any(name=='joined_strokes' for name,im,scale in variants)
