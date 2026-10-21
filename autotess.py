"""Content-agnostic local OCR. No document templates, financial rules or reference answers."""
from pathlib import Path
import os, shutil, subprocess, tempfile, csv, io, hashlib
import numpy as np
from PIL import Image, ImageOps
import cv2
from concurrent.futures import ThreadPoolExecutor


def quality(words):
    # Content-independent: confidence and recognized character coverage, no dictionary of expected fields.
    meaningful=[w for w in words if sum(c.isalnum() for c in w['text'])>=2]
    if not meaningful:return 0.
    weights=np.array([min(16,len(w['text'])) for w in meaningful],float)
    conf=np.array([max(0,w['confidence']) for w in meaningful],float)
    mean=float(np.average(conf,weights=weights))
    useful=sum(weights[conf>=55])
    fragmentation=len(meaningful)/max(1,len(words))
    return mean**2*np.log1p(useful)*fragmentation


def rotation_matrix(k,width,height):
    return [np.eye(3),np.array([[0,1,0],[-1,0,width],[0,0,1]],float),np.array([[-1,0,width],[0,-1,height],[0,0,1]],float),np.array([[0,-1,height],[1,0,0],[0,0,1]],float)][k]


def map_words(words,inverse,width,height):
    out=[]
    for w in words:
        x1,y1,x2,y2=w['box'];pts=np.array([[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]],dtype=np.float32)
        pts=cv2.perspectiveTransform(pts,inverse)[0]
        box=[max(0,float(pts[:,0].min())),max(0,float(pts[:,1].min())),min(width,float(pts[:,0].max())),min(height,float(pts[:,1].max()))]
        if box[2]>box[0] and box[3]>box[1]:out.append(dict(w,box=[round(x,2)for x in box]))
    return out


def orient(image,models):
    scale=min(1.,1400/max(image.shape[:2]));small=cv2.resize(image,None,fx=scale,fy=scale) if scale<1 else image
    def probe(k):return quality(run_tesseract(np.ascontiguousarray(np.rot90(small,k)),models,psm=11))
    with ThreadPoolExecutor(max_workers=4)as ex:scores=list(ex.map(probe,range(4)))
    k=max(range(4),key=lambda i:scores[i])
    return np.ascontiguousarray(np.rot90(image,k)),rotation_matrix(k,image.shape[1],image.shape[0]),k*90,scores


def robust_sheet_quad(contour,shape):
    """Fit dominant long sheet edges, rejecting short bright side protrusions."""
    points=cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
    s=points.sum(axis=1);d=np.diff(points,axis=1).ravel()
    quad=np.float32([points[s.argmin()],points[d.argmin()],points[s.argmax()],points[d.argmax()]])
    if len(np.unique(quad,axis=0))<4:return points
    width=max(2,int(np.linalg.norm(quad[1]-quad[0])));height=max(2,int(np.linalg.norm(quad[3]-quad[0])))
    transform=cv2.getPerspectiveTransform(quad,np.float32([[0,0],[width-1,0],[width-1,height-1],[0,height-1]]))
    mask=np.zeros(shape[:2],np.uint8);cv2.drawContours(mask,[contour],-1,255,-1)
    mask=cv2.warpPerspective(mask,transform,(width,height),flags=cv2.INTER_NEAREST)
    transposed=width>height
    if transposed:mask=mask.T
    long,short=mask.shape
    if long<short*1.3:return points
    measurements=[]
    for y in range(2,long-2):
        xs=np.flatnonzero(mask[y])
        if len(xs)>short*.2:measurements.append((y,int(xs[0]),int(xs[-1])))
    if len(measurements)<long*.65:return points
    data=np.asarray(measurements,float);widths=data[:,2]-data[:,1];median=np.median(widths)
    good=data[(widths>=median*.8)&(widths<=median*1.12)]
    if len(good)<len(data)*.55 or np.ptp(good[:,0])<long*.65:return points
    left=np.polyfit(good[:,0],good[:,1],1);right=np.polyfit(good[:,0],good[:,2],1)
    residual=np.median(np.abs(np.polyval(left,good[:,0])-good[:,1])+np.abs(np.polyval(right,good[:,0])-good[:,2]))
    if residual>max(5,median*.05):return points
    y0=float(data[:,0].min());y1=float(data[:,0].max())
    refined=np.float32([[np.polyval(left,y0),y0],[np.polyval(right,y0),y0],[np.polyval(right,y1),y1],[np.polyval(left,y1),y1]])
    if transposed:refined=refined[:,::-1].copy()
    return cv2.perspectiveTransform(refined.reshape(1,-1,2),np.linalg.inv(transform))[0]


def page_regions(image):
    """Find bright quadrilateral sheets using geometry only; no text labels/coordinates."""
    h,w=image.shape[:2];gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    smooth=cv2.GaussianBlur(gray,(9,9),0)
    threshold,_=cv2.threshold(smooth,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask=np.uint8(smooth>max(100,threshold))*255
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((13,13),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    regions=[]
    pending=list(sorted(contours,key=cv2.contourArea,reverse=True))
    for contour in pending:
        area=cv2.contourArea(contour)
        if area<h*w*.055:continue
        poly=cv2.approxPolyDP(contour,.025*cv2.arcLength(contour,True),True)
        if len(poly)!=4 or not cv2.isContourConvex(poly):
            poly=robust_sheet_quad(contour,image.shape).reshape(-1,1,2)
            remainder=np.zeros((h,w),np.uint8);cv2.drawContours(remainder,[contour],-1,255,-1)
            cv2.fillPoly(remainder,[np.round(poly).astype(np.int32)],0)
            fragments,_=cv2.findContours(remainder,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for fragment in fragments:
                if cv2.contourArea(fragment)<h*w*.04:continue
                outline=cv2.approxPolyDP(fragment,.035*cv2.arcLength(fragment,True),True)
                if len(outline)==4 and cv2.isContourConvex(outline) and len(pending)<20:pending.append(fragment)
                elif len(pending)<20:
                    rect=cv2.minAreaRect(fragment);rw,rh=rect[1]
                    if min(rw,rh)>140 and cv2.contourArea(fragment)/max(1,rw*rh)>.75:
                        pending.append(np.round(cv2.boxPoints(rect)).astype(np.int32).reshape(-1,1,2))
        pts=poly[:,0,:].astype(np.float32);sums=pts.sum(axis=1);diff=np.diff(pts,axis=1).ravel()
        ordered=np.array([pts[np.argmin(sums)],pts[np.argmin(diff)],pts[np.argmax(sums)],pts[np.argmax(diff)]])
        if len(np.unique(ordered,axis=0))<4:continue
        tl,tr,br,bl=ordered
        width=int(max(np.linalg.norm(tr-tl),np.linalg.norm(br-bl)))
        height=int(max(np.linalg.norm(bl-tl),np.linalg.norm(br-tr)))
        if min(width,height)<140 or area/(width*height)<.45:continue
        dst=np.float32([[0,0],[width,0],[width,height],[0,height]])
        matrix=cv2.getPerspectiveTransform(ordered,dst)
        crop=cv2.warpPerspective(image,matrix,(width,height),borderValue=(255,255,255))
        regions.append({'image':crop,'matrix':matrix,'area':area})
    if not regions:return [{'image':image,'matrix':np.eye(3),'area':w*h}]
    return sorted(regions[:6],key=lambda r:cv2.perspectiveTransform(np.float32([[[0,0]]]),np.linalg.inv(r['matrix']))[0,0,0])


def deskew(image):
    h,w=image.shape[:2];gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)if image.ndim==3 else image
    edges=cv2.Canny(cv2.GaussianBlur(gray,(5,5),.7),50,140)
    lines=cv2.HoughLinesP(edges,1,np.pi/720,threshold=60,minLineLength=max(70,w//6),maxLineGap=15)
    angles=[]
    if lines is not None:
        for x1,y1,x2,y2 in lines.reshape(-1,4):
            angle=np.degrees(np.arctan2(y2-y1,x2-x1))
            if abs(angle)<=12:angles.append((angle,np.hypot(x2-x1,y2-y1)))
    if len(angles)<3:return image,np.eye(3),0.
    angles.sort();total=sum(v for a,v in angles);acc=0.;angle=0.
    for a,v in angles:
        acc+=v
        if acc>=total/2:angle=float(a);break
    if abs(angle)<.15:return image,np.eye(3),0.
    affine=cv2.getRotationMatrix2D((w/2,h/2),angle,1)
    nw=int(h*abs(affine[0,1])+w*abs(affine[0,0]));nh=int(h*abs(affine[0,0])+w*abs(affine[0,1]))
    affine[0,2]+=nw/2-w/2;affine[1,2]+=nh/2-h/2
    out=cv2.warpAffine(image,affine,(nw,nh),flags=cv2.INTER_CUBIC,borderValue=(255,255,255))
    return out,np.vstack([affine,[0,0,1]]),angle


def preprocess(image):
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) if image.ndim==3 else image
    gray=cv2.GaussianBlur(gray,(0,0),.65)
    background=cv2.GaussianBlur(gray,(0,0),19)
    normalized=cv2.divide(gray,np.maximum(background,1),scale=230)
    scale=min(3.,3600/max(gray.shape));scale=max(1.,scale)
    up=cv2.resize(normalized,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
    binary=cv2.adaptiveThreshold(up,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,13)
    ink=255-binary
    horizontal=cv2.morphologyEx(ink,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(max(65,int(up.shape[1]*.08)),1)))
    vertical=cv2.morphologyEx(ink,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(1,max(65,int(up.shape[0]*.08)))))
    mask=cv2.bitwise_or(horizontal,vertical)
    mask=cv2.dilate(mask,np.ones((3,3),np.uint8))
    clean=cv2.inpaint(up,mask,3,cv2.INPAINT_TELEA)
    raw=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)if image.ndim==3 else image
    raw=cv2.resize(raw,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
    joined=255-cv2.morphologyEx(255-raw,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    return [('gray',up,scale),('raw_gray',raw,scale),('joined_strokes',joined,scale),('no_grid',clean,scale),('binary',cv2.bitwise_or(binary,mask),scale)]


def deduplicate_tokens(words):
    """Suppress repeated readings of identical pixels, not repeated values on a page."""
    retained=[]
    for index,word in enumerate(words):
        a=word['box'];wa=max(1,(a[2]-a[0])*(a[3]-a[1]));text=''.join(word['text'].casefold().split());drop=False
        for j,other in enumerate(words):
            if j==index:continue
            b=other['box'];othertext=''.join(other['text'].casefold().split())
            if not text or text not in othertext:continue
            if len(othertext)<len(text) or len(othertext)==len(text)and j>index:continue
            if abs((a[1]+a[3]-b[1]-b[3])/2)>.4*min(a[3]-a[1],b[3]-b[1]):continue
            intersect=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
            if intersect/wa>=.85 and (other['confidence']>=45 or text==othertext):drop=True;break
        if not drop:retained.append(word)
    return retained


def fuse_passes(results):
    """Choose complete OCR word groups by local geometric overlap; never synthesize characters."""
    full_page=[r for r in results if not r['name'].startswith('auto_lines_')]
    best=max(full_page or results,key=lambda r:r['quality'])
    seed=[w for w in best['words'] if any(c.isalnum()for c in w['text'])]
    def overlap(a,b):
        aa=a['box'];bb=b['box'];intersection=max(0,min(aa[2],bb[2])-max(aa[0],bb[0]))*max(0,min(aa[3],bb[3])-max(aa[1],bb[1]))
        return intersection/max(1,min((aa[2]-aa[0])*(aa[3]-aa[1]),(bb[2]-bb[0])*(bb[3]-bb[1])))
    proposals=[(i,w)for i,r in enumerate(results)for w in r['words']if w['confidence']>=75 and sum(c.isalnum()for c in w['text'])>=2]
    for source,word in sorted(proposals,key=lambda pair:-pair[1]['confidence']):
        if any(overlap(word,existing)>.3 for existing in seed):continue
        agree=any(i!=source and other['text'].casefold()==word['text'].casefold()and overlap(word,other)>.7 for i,other in proposals)
        if agree:seed.append(word)
    if not seed:return [],[]
    rows=[]
    for word in sorted(seed,key=lambda w:(w['box'][1]+w['box'][3])/2):
        box=word['box'];cy=(box[1]+box[3])/2;h=box[3]-box[1]
        if rows and abs(cy-rows[-1]['cy'])<max(4,.5*min(h,rows[-1]['height'])):
            rows[-1]['words'].append(word)
            rows[-1]['cy']=float(np.median([(w['box'][1]+w['box'][3])/2 for w in rows[-1]['words']]))
            rows[-1]['height']=float(np.median([w['box'][3]-w['box'][1]for w in rows[-1]['words']]))
        else:rows.append({'cy':cy,'height':h,'words':[word]})
    selected=[];evidence=[]
    for row_index,row in enumerate(rows):
        ordered=sorted(row['words'],key=lambda w:w['box'][0]);groups=[]
        for w in ordered:
            if not groups or w['box'][0]-groups[-1][-1]['box'][2]>2.2*row['height']:groups.append([])
            groups[-1].append(w)
        for group in groups:
            x1=min(w['box'][0]for w in group);x2=max(w['box'][2]for w in group)
            y1=min(w['box'][1]for w in group);y2=max(w['box'][3]for w in group)
            pad=max(3,row['height']*.35);options=[]
            for index,r in enumerate(results):
                ww=[w for w in r['words']if x1-pad<=(w['box'][0]+w['box'][2])/2<=x2+pad and abs((w['box'][1]+w['box'][3])/2-row['cy'])<=max(row['height']*.6,4)]
                ww=sorted(ww,key=lambda w:w['box'][0])
                if ww:options.append({'pass':index,'words':ww,'text':' '.join(w['text']for w in ww)})
            if not options:continue
            for option in options:
                normalized=''.join(option['text'].casefold().split())
                support=sum(''.join(x['text'].casefold().split())==normalized for x in options)
                ww=option['words']
                meaningful=[w for w in ww if any(c.isalnum()for c in w['text'])]
                lengths=[max(1,min(16,len(w['text'])))for w in meaningful]
                confidence=float(np.average([max(0,w['confidence'])for w in meaningful],weights=lengths))if lengths else 0.
                left=min(w['box'][0]for w in ww);right=max(w['box'][2]for w in ww)
                coverage=min(1.,max(0,min(x2,right)-max(x1,left))/max(1,x2-x1))
                option['score']=confidence**2*np.log1p(sum(lengths))*coverage**2*(1+.6*(support-1))
            winner=max(options,key=lambda x:x['score'])
            for w in winner['words']:selected.append(dict(w,line=[1,1,row_index+1],selected_pass=winner['pass']))
            evidence.append({'box':[x1,y1,x2,y2],'selected_pass':winner['pass'],'alternatives':[{'pass':x['pass'],'text':x['text'],'score':round(x['score'],3)}for x in options]})
    return selected,evidence


def rescan_groups(image,evidence,models):
    """Pack automatically detected regions into large, separated lines; OCR the pixels anew."""
    tiles=[];h,w=image.shape[:2]
    for item in evidence:
        x1,y1,x2,y2=item['box'];height=y2-y1
        if height<5 or x2-x1<5:continue
        pad=max(4,int(height*.25))
        left=max(0,int(x1)-pad);top=max(0,int(y1)-pad);right=min(w,int(x2)+pad);bottom=min(h,int(y2)+pad)
        crop=image[top:bottom,left:right]
        scale=min(3.,max(.8,48/max(height,1)),2200/max(1,right-left))
        crop=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
        tiles.append({'image':crop,'left':left,'top':top,'scale':scale})
    chunks=[];current=[];used=30
    for tile in tiles:
        hh=tile['image'].shape[0]+50
        if current and used+hh>12000:chunks.append(current);current=[];used=30
        current.append(tile);used+=hh
    if current:chunks.append(current)
    out=[]
    for chunk in chunks:
        width=max(t['image'].shape[1]for t in chunk)+60;height=sum(t['image'].shape[0]+50 for t in chunk)+30
        canvas=np.full((height,width),255,np.uint8);y=25
        for tile in chunk:
            tile['canvas_y']=y;ih,iw=tile['image'].shape[:2];canvas[y:y+ih,30:30+iw]=tile['image'];y+=ih+50
        words=run_tesseract(canvas,models,psm=6,timeout=120)
        for word in words:
            x1,y1,x2,y2=word['box'];cy=(y1+y2)/2
            tile=next((t for t in chunk if t['canvas_y']<=cy<=t['canvas_y']+t['image'].shape[0]),None)
            if tile is None:continue
            s=tile['scale'];left=tile['left'];top=tile['top'];oy=tile['canvas_y']
            box=[left+(x1-30)/s,top+(y1-oy)/s,left+(x2-30)/s,top+(y2-oy)/s]
            out.append(dict(word,box=box))
    return out


def executable():
    candidates=[os.environ.get('TESSERACT_CMD'),shutil.which('tesseract'),r'C:\Program Files\Tesseract-OCR\tesseract.exe']
    for p in candidates:
        if p and Path(p).is_file():return str(p)
    raise RuntimeError('Tesseract executable not found; set TESSERACT_CMD')


def run_tesseract(image,models,psm=6,language='ukr+rus+eng',timeout=75,thresholding=0):
    with tempfile.TemporaryDirectory(prefix='autotess_') as td:
        p=Path(td)/'page.png';Image.fromarray(image).save(p)
        cmd=[executable(),str(p),'stdout','--tessdata-dir',str(models),'-l',language,'--oem','1','--psm',str(psm),'--dpi','300','-c','tessedit_create_tsv=1','-c',f'thresholding_method={thresholding}']
        env=dict(os.environ);env['OMP_THREAD_LIMIT']='1'
        r=subprocess.run(cmd,capture_output=True,encoding='utf-8',errors='replace',timeout=timeout,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if r.returncode:raise RuntimeError('Tesseract: '+r.stderr[-1500:])
        words=[]
        for row in csv.DictReader(io.StringIO(r.stdout),delimiter='\t',quoting=csv.QUOTE_NONE):
            if row.get('level')!='5' or not (row.get('text')or'').strip():continue
            try:
                x,y,w,h=(int(row[k])for k in ['left','top','width','height'])
                words.append({'text':row['text'].strip(),'confidence':float(row['conf']),'box':[x,y,x+w,y+h], 'line':[int(row[k])for k in ['block_num','par_num','line_num']]})
            except (ValueError,TypeError):continue
        return words


def reorder_words(words,regions):
    """Reconcile line order using final word geometry in upright coordinates."""
    output=[]
    for region in regions:
        matrix=np.asarray(region['matrix'],float);local=[]
        for word in words:
            if word.get('region',0)!=region['index']:continue
            x1,y1,x2,y2=word['box'];pts=np.float32([[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]])
            pts=cv2.perspectiveTransform(pts,matrix)[0]
            b=[float(pts[:,0].min()),float(pts[:,1].min()),float(pts[:,0].max()),float(pts[:,1].max())]
            local.append((word,b))
        rows=[]
        for word,b in sorted(local,key=lambda pair:(pair[1][1]+pair[1][3])/2):
            cy=(b[1]+b[3])/2;h=b[3]-b[1]
            if rows and abs(cy-rows[-1]['cy'])<=max(3,.6*max(h,rows[-1]['height'])):
                rows[-1]['items'].append((word,b))
                rows[-1]['cy']=float(np.median([(a[1]+a[3])/2 for w,a in rows[-1]['items']]))
                rows[-1]['height']=float(np.median([a[3]-a[1]for w,a in rows[-1]['items']]))
            else:rows.append({'cy':cy,'height':h,'items':[(word,b)]})
        for i,row in enumerate(rows):
            for word,b in sorted(row['items'],key=lambda pair:pair[1][0]):output.append(dict(word,line=[region['index'],1,1,i+1]))
    return output


def text_from_words(words):
    out=[];last=None
    for w in words:
        key=w.get('line')
        if key!=last:out.append([]);last=key
        if not out:out.append([])
        out[-1].append(w['text'])
    return '\n'.join(' '.join(line)for line in out)


def recognize(path,models,mode='accurate'):
    path=Path(path);models=Path(models)
    image=np.array(ImageOps.exif_transpose(Image.open(path)).convert('RGB'))
    all_words=[];passes=[];region_info=[];local_evidence=[]
    regions=page_regions(image) if mode!='quick' else [{'image':image,'matrix':np.eye(3),'area':image.shape[0]*image.shape[1]}]
    for region_index,region in enumerate(regions):
        processed,rotation_transform,rotation,scores=orient(region['image'],models) if mode!='quick' else (region['image'],np.eye(3),0,[])
        processed,deskew_transform,skew=deskew(processed) if mode!='quick' else (processed,np.eye(3),0.)
        base_matrix=deskew_transform@rotation_transform@region['matrix']
        variants=preprocess(processed) if mode!='quick' else [('original',processed,1.)]
        jobs=[(name,im,scale,psm,0)for name,im,scale in variants for psm in ((6,11)if mode!='quick'else(6,))]
        if mode=='experimental':jobs.extend((name+'_sauvola',im,scale,psm,2)for name,im,scale in variants if name=='raw_gray' for psm in (6,11))
        def work(job):
            name,im,scale,psm,thresholding=job
            words=run_tesseract(im,models,psm=psm,thresholding=thresholding)
            return {'name':name,'psm':psm,'scale':scale,'words':words,'quality':quality(words),'thresholding':thresholding}
        with ThreadPoolExecutor(max_workers=3)as ex:results=list(ex.map(work,jobs))
        if mode!='quick':
            _,seed_evidence=fuse_passes(results)
            for name,im,scale in variants:
                if name not in ('raw_gray','no_grid'):continue
                rescanned=rescan_groups(im,seed_evidence,models)
                results.append({'name':'auto_lines_'+name,'psm':6,'scale':scale,'words':rescanned,'quality':quality(rescanned)})
        best=max(results,key=lambda x:x['quality'])
        for entry in results:
            passes.append({'region':region_index,'variant':entry['name'],'psm':entry['psm'],'quality':entry['quality'],'word_count':len(entry['words']),'selected':entry is best,'scale':entry['scale'],'words':entry['words']})
        matrix=np.diag([best['scale'],best['scale'],1.])@base_matrix
        chosen,evidence=fuse_passes(results) if mode!='quick' else (best['words'],[])
        for item in evidence:
            item['region']=region_index
            item['processed_box']=item['box']
            original=map_words([item],np.linalg.inv(matrix),image.shape[1],image.shape[0])
            if original:
                item['box']=original[0]['box'];item['coordinate_space']='original'
                local_evidence.append(item)
        words=map_words(chosen,np.linalg.inv(matrix),image.shape[1],image.shape[0])
        for word in words:word['line']=[region_index]+word['line'];word['region']=region_index
        all_words.extend(words)
        region_info.append({'index':region_index,'rotation':rotation,'deskew_degrees':skew,'orientation_scores':scores,'matrix':base_matrix.tolist(),'selected_variant':best['name'],'selected_psm':best['psm']})
    all_words=deduplicate_tokens(all_words)
    if mode=='experimental':all_words=reorder_words(all_words,region_info)
    layout={'quick':'tesseract','accurate':'fusion_geometry','experimental':'upright_geometry'}[mode]
    return {'schema_version':1,'engine':'tesseract-lstm','input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'input_size':[image.shape[1],image.shape[0]],'words':all_words,'text':text_from_words(all_words),'status':'unverified_ocr','mode':mode,'rotation':region_info[0]['rotation'],'regions':region_info,'passes':passes,'local_evidence':local_evidence,'reference_data_used':False,'layout_order':layout}
