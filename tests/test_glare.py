import cv2,numpy as np
import autotess

def test_attached_bright_glare_does_not_expand_document_crop():
    image=np.full((1100,1000,3),40,np.uint8)
    cv2.rectangle(image,(200,50),(520,1050),(235,235,235),-1)
    cv2.ellipse(image,(580,260),(230,110),0,0,360,(190,190,190),-1)
    # Dark generic text-like strokes ensure the detector cannot rely on a blank card.
    for y in range(100,1000,50):cv2.putText(image,'Unknown document',(225,y),cv2.FONT_HERSHEY_SIMPLEX,.5,(30,30,30),1)
    regions=autotess.page_regions(image)
    largest=max(regions,key=lambda x:x['image'].shape[0])
    assert largest['image'].shape[1]<400,largest['image'].shape
    assert largest['image'].shape[0]>900


def test_connected_secondary_paper_is_not_discarded_as_glare():
    image=np.full((1100,1100,3),40,np.uint8)
    cv2.rectangle(image,(500,30),(850,1060),(235,235,235),-1)
    cv2.rectangle(image,(150,350),(520,720),(235,235,235),-1)
    for x,y in [(180,390),(530,80)]:
        for i in range(7):cv2.putText(image,'Independent text',(x,y+i*40),cv2.FONT_HERSHEY_SIMPLEX,.7,(20,20,20),1)
    regions=autotess.page_regions(image)
    assert len(regions)>=2,'Secondary connected paper was cropped away'


def test_real_touching_receipts_are_not_lost():
    from pathlib import Path
    from PIL import Image
    import pytest
    import os
    for n in [15,19]:
        path=Path(os.environ.get('OCR_FIXTURE_DIR','missing'))/f'touching_receipts_{n}.jpg'
        if not path.exists():pytest.skip('External photographic fixtures unavailable (set OCR_FIXTURE_DIR)')
        assert len(autotess.page_regions(np.array(Image.open(path).convert('RGB'))))>=2,n
