import cv2,numpy as np
import autotess

def test_deskew_rotated_grid_without_keywords():
    assert hasattr(autotess,'deskew'),'Deskew missing'
    image=np.full((700,1000,3),255,dtype=np.uint8)
    for y in range(100,620,65):cv2.line(image,(100,y),(900,y),(0,0,0),2)
    for x in range(100,950,160):cv2.line(image,(x,100),(x,620),(0,0,0),2)
    m=cv2.getRotationMatrix2D((500,350),7,1)
    image=cv2.warpAffine(image,m,(1000,700),borderValue=(255,255,255))
    output,matrix,angle=autotess.deskew(image)
    assert abs(angle+7)<1,angle
    assert output.shape[0]>=image.shape[0]
    assert matrix.shape==(3,3)
