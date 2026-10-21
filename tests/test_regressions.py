import numpy as np
import cv2
import autotess

def test_detector_keeps_large_irregular_sheet_beside_small_rectangular_sheet():
    image=np.full((900,1100,3),55,dtype=np.uint8)
    cv2.rectangle(image,(60,350),(350,800),(230,230,230),-1)
    pts=np.array([[530,40],[1030,60],[1000,500],[1040,830],[790,810],[520,850],[500,400]],np.int32)
    cv2.fillPoly(image,[pts],(240,240,240))
    assert len(autotess.page_regions(image))==2,'Dropping non-quadrilateral paper loses text'

def test_two_real_receipts_are_both_detected():
    from PIL import Image
    from pathlib import Path
    import pytest
    import os
    p=Path(os.environ.get('OCR_FIXTURE_DIR','missing'))/'two_receipts_1.jpg'
    if not p.exists():pytest.skip('External photographic fixture unavailable (set OCR_FIXTURE_DIR)')
    assert len(autotess.page_regions(np.array(Image.open(p).convert('RGB'))))==2


def test_pass_quality_does_not_reward_hundreds_of_noise_fragments():
    good=[{'text':'Independent','confidence':91},{'text':'sample','confidence':85},{'text':'8347.63','confidence':96}]
    noise=[{'text':'i','confidence':80},{'text':'|','confidence':70},{'text':'hy','confidence':30}]*150
    assert autotess.quality(good)>autotess.quality(noise)
