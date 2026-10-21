import autotess

def w(text,box,conf=90):return {'text':text,'box':box,'confidence':conf,'line':[1,1,1]}

def test_deduplicate_exact_same_pixel_token_and_contained_suffix():
    assert hasattr(autotess,'deduplicate_tokens'),'Pixel deduplication missing'
    a=w('8537.46',[10,10,110,30]);b=w('46',[85,10,110,30],95)
    words=autotess.deduplicate_tokens([a,dict(a),b,w('46',[150,10,175,30])])
    assert [x['text']for x in words]==['8537.46','46']
    assert words[-1]['box']==[150,10,175,30]

def test_different_rows_or_adjacent_identical_tokens_are_preserved():
    assert hasattr(autotess,'deduplicate_tokens')
    words=[w('250',[10,10,40,30]),w('250',[50,10,80,30]),w('250',[10,35,40,55])]
    assert len(autotess.deduplicate_tokens(words))==3
