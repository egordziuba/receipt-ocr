"""External evaluator only. NEVER imported by autotess.py or ocr.py.
Gold boxes/text only select and score previously generated OCR; no OCR is performed here.
"""
from pathlib import Path
import json,re,unicodedata,argparse,collections,sys,hashlib
from rapidfuzz.distance import Levenshtein


def normalize(text,kind):
    text=unicodedata.normalize('NFKC',text).casefold()
    text=re.sub(r'\s+','',text)
    if kind in ('money','number'):
        text=text.replace(',','.').replace('’','').replace("'",'')
    return text


def score_observation(expected,actual,kind):
    a=normalize(expected,kind);b=normalize(actual,kind)
    return {'expected_normalized':a,'actual_normalized':b,'reference_chars':len(a),'errors':Levenshtein.distance(a,b),'exact':a==b,'recognized':bool(b)}


def extract(words,box):
    x1,y1,x2,y2=box
    return ' '.join(w['text']for w in words if x1<=(w['box'][0]+w['box'][2])/2<=x2 and y1<=(w['box'][1]+w['box'][3])/2<=y2)


def summary(rows):
    n=len(rows);chars=sum(r['reference_chars']for r in rows);errors=sum(r['errors']for r in rows)
    return {'observations':n,'reference_chars':chars,'edit_errors':errors,'character_accuracy_pct':round(max(0,1-errors/chars)*100,3)if chars else None,'exact_observations':sum(r['exact']for r in rows),'exact_pct':round(sum(r['exact']for r in rows)*100/n,3)if n else None,'nonempty':sum(r['recognized']for r in rows)}


def verify_provenance(images,results,source):
    errors=[];manifest_path=results/'manifest.json'
    if not manifest_path.is_file():return {'verified':False,'errors':['Finished manifest missing']}
    manifest=json.loads(manifest_path.read_text('utf8'));entries=manifest.get('results',[])
    if len(entries)!=manifest.get('unique_images'):errors.append('Manifest incomplete')
    if any(x.get('status')=='ERROR' for x in entries):errors.append('Manifest contains failed inputs')
    if source is None:return {'verified':False,'errors':errors+['Source image directory required']}
    index={};files=list(results.glob('*.ocr.json'));expected=manifest.get('run_fingerprint')
    if len(files)!=manifest.get('unique_images'):errors.append('Result count differs from manifest')
    for path in files:
        data=json.loads(path.read_text('utf8'))
        if not expected or data.get('run_fingerprint')!=expected:errors.append('Run fingerprint mismatch: '+path.name)
        for alias in data.get('source_files',[]):
            name=Path(alias).name
            if name in index:errors.append('Ambiguous basename: '+name)
            index[name]=data
    checked=set()
    for image in images:
        name=image['file'];path=source/name;data=index.get(name)
        if Path(name).name!=name:errors.append('Unsafe source name');continue
        if not path.is_file() or data is None:errors.append('Missing source/result: '+name);continue
        digest=hashlib.sha256(path.read_bytes()).hexdigest();checked.add(digest)
        if digest!=data.get('input_sha256'):errors.append('Source hash mismatch: '+name)
        if not any(x.get('sha256')==digest for x in entries):errors.append('Source hash absent from manifest: '+name)
    if len(checked)!=len(images):errors.append('Missing or duplicated annotated images')
    return {'verified':not errors,'errors':errors,'checked_unique_images':len(checked),'run_fingerprint':expected}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--annotations',type=Path,required=True);ap.add_argument('--results',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--require-character-accuracy',type=float);ap.add_argument('--source-dir',type=Path);ap.add_argument('--strict-provenance',action='store_true');args=ap.parse_args()
    images=[]
    for p in sorted(args.annotations.glob('*.json')):
        data=json.loads(p.read_text('utf8'));images.extend(data.get('images',[]))
    provenance=verify_provenance(images,args.results,args.source_dir)
    if args.strict_provenance and not provenance['verified']:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps({'provenance':provenance,'acceptance':{'passed':False}},ensure_ascii=False,indent=2),encoding='utf8')
        print(json.dumps(provenance,ensure_ascii=False));return 2
    by_name={}
    for p in args.results.glob('*.ocr.json'):
        data=json.loads(p.read_text('utf8'))
        for name in data.get('source_files',[]):by_name[Path(name).name]=data
    rows=[];unscored=[]
    for image in images:
        result=by_name.get(image['file'],{'words':[]})
        for obs in image['observations']:
            if obs.get('legible',True)is False:
                unscored.append({'file':image['file'],**obs});continue
            actual=extract(result['words'],obs['box'])
            rows.append({'file':image['file'],'category':image['category'],**obs,'actual':actual,**score_observation(obs['text'],actual,obs['kind'])})
    groups=sorted(set(r['category']for r in rows))
    report={'scope':'Sampled annotated regions; not full-page accuracy and not proof on unseen documents. Blank/missing OCR is counted as error. Human-illegible regions are reported separately.','normalization':'Unicode NFKC, casefold, remove whitespace; money: decimal comma -> dot, strip grouping apostrophes. No digit, date, spelling or alphabet corrections.', 'unique_images':len(set(i['file']for i in images)),'all':summary(rows),'by_category':{g:summary([r for r in rows if r['category']==g])for g in groups},'money':summary([r for r in rows if r['kind']=='money']),'by_image':{f:summary([r for r in rows if r['file']==f])for f in sorted(set(r['file']for r in rows))},'unscored_illegible':unscored,'observations':rows}
    report['provenance']=provenance
    score=report['all']['character_accuracy_pct']
    passed=bool(rows) and score is not None and (args.require_character_accuracy is None or score>=args.require_character_accuracy)
    report['acceptance']={'required_character_accuracy_pct':args.require_character_accuracy,'passed':passed,'scope':'annotated_regions_only','full_document_95_proven':False}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items()if k not in ['observations','unscored_illegible','by_image']},ensure_ascii=False,indent=2));print('UNSCORED_ILLEGIBLE',len(unscored));print('REPORT',args.output)
    return 0 if passed else 1
if __name__=='__main__':sys.exit(main())
