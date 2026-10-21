import importlib.util

def test_metrics_count_substitution_and_missing_field_as_errors():
    assert importlib.util.find_spec('evaluate') is not None,'Evaluator missing'
    from evaluate import score_observation
    a=score_observation('12 345,67','12 345.67','money')
    assert a['exact'] and a['errors']==0
    b=score_observation('12345.67','12345.87','money')
    assert not b['exact'] and b['errors']==1
    c=score_observation('12345.67','','money')
    assert not c['exact'] and c['errors']==c['reference_chars']
    d=score_observation('сейф','ceйф','text')
    assert not d['exact'], 'Must not silently replace Latin with Cyrillic'
