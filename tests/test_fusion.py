import autotess

def token(text,conf,x,y,w=100,h=20):return {'text':text,'confidence':conf,'box':[x,y,x+w,y+h],'line':[1,1,int(y//30)+1]}

def test_select_best_local_regions_not_one_winner_for_whole_page():
    assert hasattr(autotess,'fuse_passes'),'Local ensemble missing'
    first={'name':'a','psm':6,'scale':1.,'words':[token('Alpha',95,10,10),token('Bxta',35,10,60)],'quality':200}
    second={'name':'b','psm':11,'scale':1.,'words':[token('Alpna',35,10,10),token('Beta',96,10,60)],'quality':190}
    words,evidence=autotess.fuse_passes([first,second])
    assert [w['text']for w in words]==['Alpha','Beta']
    assert evidence and all('alternatives' in x for x in evidence)
    assert all(w['text']in ['Alpha','Beta','Alpna','Bxta']for w in words)


def test_fusion_keeps_missing_region_when_other_passes_agree():
    first={'name':'a','psm':6,'scale':1.,'words':[token('Alpha',99,10,10)],'quality':300}
    second={'name':'b','psm':11,'scale':1.,'words':[token('Alpha',95,10,10),token('Gamma',94,10,90)],'quality':200}
    third={'name':'c','psm':6,'scale':1.,'words':[token('Gamma',93,11,90)],'quality':100}
    words,_=autotess.fuse_passes([first,second,third])
    assert 'Gamma' in [w['text']for w in words]


def test_rescan_is_not_allowed_to_remove_original_page_coverage():
    original={'name':'raw_gray','psm':6,'scale':1.,'words':[token('Alpha',95,10,10),token('Beta',71,10,90)],'quality':200}
    rescan={'name':'auto_lines_raw_gray','psm':6,'scale':1.,'words':[token('Alpha',99,10,10)],'quality':400}
    words,_=autotess.fuse_passes([original,rescan])
    assert 'Beta' in [w['text']for w in words]


def test_supported_reading_beats_single_overconfident_disagreement():
    a={'name':'gray','psm':6,'scale':1.,'words':[token('Alpha',90,10,10)],'quality':200}
    b={'name':'raw_gray','psm':11,'scale':1.,'words':[token('Alpha',90,10,10)],'quality':200}
    c={'name':'binary','psm':6,'scale':1.,'words':[token('Alpna',99,10,10)],'quality':300}
    words,_=autotess.fuse_passes([a,b,c])
    assert [x['text']for x in words]==['Alpha']


def test_high_confidence_partial_reading_does_not_drop_leading_character():
    a={'name':'gray','psm':6,'scale':1.,'words':[token('Z',85,10,10,w=10),token('ebra',92,22,10,w=65)],'quality':300}
    b={'name':'binary','psm':11,'scale':1.,'words':[token('ebra',97,22,10,w=65)],'quality':200}
    words,_=autotess.fuse_passes([a,b])
    assert ''.join(x['text']for x in words)=='Zebra'
