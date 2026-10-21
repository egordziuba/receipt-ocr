"""Standalone OCR CLI. Reads images and Tesseract models, never annotations or workbooks."""
from pathlib import Path
import argparse,json,hashlib,time,sys,os
from concurrent.futures import ThreadPoolExecutor,as_completed
from autotess import recognize

def save_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8');os.replace(temp,path)

def main():
    ap=argparse.ArgumentParser(description='Autonomous Tesseract OCR: no templates, expected sums or reference answers.')
    ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--pattern',default='*');ap.add_argument('--models',type=Path,default=Path(__file__).parent/'models')
    ap.add_argument('--mode',choices=['quick','accurate','experimental'],default='accurate');ap.add_argument('--workers',type=int,default=2)
    args=ap.parse_args()
    if not args.input.exists():ap.error('Input does not exist')
    files=sorted(p for p in ([args.input]if args.input.is_file()else args.input.glob(args.pattern))if p.is_file()and p.suffix.lower()in {'.jpg','.jpeg','.png','.tif','.tiff','.bmp','.webp'})
    if not files:ap.error('No supported images')
    if args.workers<1 or args.workers>4:ap.error('workers must be 1..4')
    args.output.mkdir(parents=True,exist_ok=True)
    grouped={}
    for p in files:grouped.setdefault(hashlib.sha256(p.read_bytes()).hexdigest(),[]).append(p)
    engine_revision=hashlib.sha256(Path(__file__).with_name('autotess.py').read_bytes()).hexdigest()
    model_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted(args.models.glob('*.traineddata'))}
    fingerprint=hashlib.sha256(json.dumps([engine_revision,model_hashes,args.mode],sort_keys=True).encode()).hexdigest()
    def work(item):
        digest,paths=item;dest=args.output/(digest+'.ocr.json');t=time.monotonic()
        if dest.exists():
            previous=json.loads(dest.read_text('utf8'))
            if previous.get('run_fingerprint')==fingerprint:return {'sha256':digest,'files':[str(p)for p in paths],'result':str(dest),'cached':True,'status':previous['status']}
        try:
            result=recognize(paths[0],args.models,mode=args.mode)
            result['source_files']=[str(p)for p in paths];result['run_fingerprint']=fingerprint;result['elapsed_seconds']=round(time.monotonic()-t,3)
            save_json(dest,result)
            dest.with_suffix('.txt').write_text(result['text'],encoding='utf8')
            return {'sha256':digest,'files':[str(p)for p in paths],'result':str(dest),'cached':False,'status':result['status'],'seconds':result['elapsed_seconds'],'word_count':len(result['words'])}
        except Exception as e:return {'sha256':digest,'files':[str(p)for p in paths],'status':'ERROR','error':str(e)}
    results=[]
    with ThreadPoolExecutor(max_workers=args.workers)as ex:
        for f in as_completed([ex.submit(work,item)for item in grouped.items()]):
            r=f.result();results.append(r);print(json.dumps(r,ensure_ascii=False),flush=True)
    manifest={'input_files':len(files),'unique_images':len(grouped),'run_fingerprint':fingerprint,'engine_revision':engine_revision,'model_hashes':model_hashes,'mode':args.mode,'reference_data_used':False,'results':results}
    save_json(args.output/'manifest.json',manifest)
    return 1 if any(r['status']=='ERROR'for r in results)else 0
if __name__=='__main__':sys.exit(main())
