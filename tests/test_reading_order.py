import numpy as np
import autotess

def test_reading_order_uses_final_geometry_after_fusion():
    assert hasattr(autotess,'reorder_words'),'Final reading-order reconciliation missing'
    words=[{'text':'right','box':[60,10,150,40],'confidence':95,'line':[0,1,1,1],'region':0}, {'text':'left','box':[10,17,45,33],'confidence':95,'line':[0,1,1,2],'region':0}]
    result=autotess.reorder_words(words,[{'index':0,'matrix':np.eye(3).tolist()}])
    assert [w['text']for w in result]==['left','right']
    assert result[0]['line']==result[1]['line']
