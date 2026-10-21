from pathlib import Path
import inspect
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import autotess
ROOT=Path(__file__).resolve().parents[1]

def test_sauvola_thresholding_is_real_tesseract_option():
    assert 'thresholding' in inspect.signature(autotess.run_tesseract).parameters
    im=Image.new('L',(950,250),210);ImageDraw.Draw(im).text((40,70),'Unknown text 7648.32',font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',42),fill=40)
    words=autotess.run_tesseract(np.array(im),ROOT/'models',psm=6,thresholding=2)
    assert '7648.32' in ' '.join(w['text']for w in words)
