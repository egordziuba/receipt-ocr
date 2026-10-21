import numpy as np
import cv2
import autotess

def test_separate_paper_regions_without_document_template():
    assert hasattr(autotess,'page_regions'),'Automatic region detection missing'
    image=np.full((900,1100,3),55,dtype=np.uint8)
    cv2.rectangle(image,(60,300),(400,800),(230,230,230),-1)
    cv2.rectangle(image,(520,50),(1020,820),(240,240,240),-1)
    for x,y in [(80,340),(540,100)]:
        for row in range(8):cv2.putText(image,'Alpha beta 789',(x,y+row*42),cv2.FONT_HERSHEY_SIMPLEX,.6,(20,20,20),1)
    regions=autotess.page_regions(image)
    assert len(regions)==2,len(regions)
    assert all(region['image'].shape[0]>300 for region in regions)
    assert all(region['matrix'].shape==(3,3)for region in regions)
