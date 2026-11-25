"""總控制器：依 Config 執行 Stage1/Stage2/Stage1+2、記錄所有選樣與輸出。"""
from __future__ import annotations
from datetime import datetime
import csv
import json
import logging
import os
from pathlib import Path
import shutil
import time
import uuid
import numpy as np
from .configuration import resolved_path
from .datasets import ImageItem,discover_data,discover_parts
from .features import FeatureStore
from .greedy import select_stage1
from .stage2 import select_stage2,SubspacePCA
from .visualization import distribution,ranking,iteration,patch_error_heatmaps
from .evaluation import evaluate

LOGGER=logging.getLogger(__name__)


def save_csv(path:Path,records:list[dict],columns:list[str]|None=None):
    if not records and not columns:
        columns=['part','image_name','image_path','score','selected']
    fields=list(dict.fromkeys((columns or [])+[k for r in records for k in r.keys()]))
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        w.writeheader();w.writerows(records)


def _image_copy(src:Path,target:Path,source_label:str):
    """不覆寫同名不同內容的原始圖片，衝突時使用固定路徑雜湊。"""
    import hashlib
    target.mkdir(parents=True,exist_ok=True)
    dest=target/src.name
    if dest.exists():
        digest=hashlib.sha256(str(src.resolve()).encode()).hexdigest()[:10]
        dest=target/f'{src.stem}__{source_label}_{digest}{src.suffix}'
    shutil.copy2(src,dest)
    return str(dest)


def _write_pictures(partdir:Path,train:list[ImageItem],selected:list[ImageItem],conf:dict):
    out=conf['output']
    records=[]
    if out['copy_selected']:
        for im in selected:
            path=_image_copy(im.path,partdir/'Selected_Only','selected')
            records.append({'group':'selected','source_path':str(im.path),'output_path':path})
    if out['copy_merged']:
        for im in train:
            path=_image_copy(im.path,partdir/'Merged_Training_Set','original')
            records.append({'group':'original','source_path':str(im.path),'output_path':path})
        for im in selected:
            path=_image_copy(im.path,partdir/'Merged_Training_Set','selected')
            records.append({'group':'selected_merged','source_path':str(im.path),'output_path':path})
    return records


def _logfile(out:Path):
    fmt='%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(level=logging.INFO,format=fmt,handlers=[logging.StreamHandler(),
                         logging.FileHandler(out/'run.log',encoding='utf-8')],force=True)


def run(conf:dict,cwd:Path|None=None)->dict:
    cwd=(cwd or Path.cwd()).resolve()
    root=resolved_path(conf['dataset']['root'],cwd)
    if not root.is_dir():
        raise FileNotFoundError(f'Dataset Root 不存在：{root}')
    base=resolved_path(conf['output']['root'],cwd)
    base.mkdir(parents=True,exist_ok=True)
    run_id=datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
    out=base/run_id
    out.mkdir()
    _logfile(out)
    started=time.perf_counter()
    LOGGER.info('=== JDP Image Selection 開始 ===')
    mode=conf['selection']['mode'];strategy=conf['stage2']['strategy'];k=int(conf['selection']['k'])
    LOGGER.info('Mode=%s / Stage2=%s / K=%d / Backend=%s',mode,strategy,k,conf['feature_extractor']['backend'])
    LOGGER.info('Dataset=%s; Candidate=%s; Output=%s',root,conf['dataset']['candidate_rel'],out)
    (out/'run_config.json').write_text(json.dumps(conf,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    store=FeatureStore(conf,cwd)
    summary={'run_id':run_id,'mode':mode,'stage2_strategy':strategy,'backend':conf['feature_extractor']['backend'],
             'k':k,'shortlist_target':2*k if mode=='stage1_stage2' else k if mode=='stage1' else 0,
             'dataset_root':str(root),'output_root':str(out),'parts':{},'errors':[]}
    try:
        parts=discover_parts(conf,root)
        if not parts:
            raise ValueError('沒有找到任何 Part，請檢查 dataset.parts、root 與 train_rel。')
        for part in parts:
            partdir=out/part
            partdir.mkdir(exist_ok=True)
            plots=partdir/'plots';plots.mkdir(exist_ok=True)
            info={'status':'error','selected_count':0,'original_count':0,'candidate_count':0,'invalid_count':0}
            summary['parts'][part]=info
            LOGGER.info('--- Part：%s ---',part)
            try:
                train,candidates,skipped=discover_data(conf,root,part)
                info.update(original_count=len(train),candidate_count=len(candidates),invalid_count=len(skipped),
                            train_dir=str(root/part/conf['dataset']['train_rel']),
                            candidate_dir=str(root/part/conf['dataset']['candidate_rel']))
                # Random Only 不使用 DINO/PCA。若也關閉特徵圖與評估，就不必浪費時間萃取。
                needs_features=(mode!='stage2' or strategy!='random' or
                                conf['visualization']['feature_plot']!='none' or
                                conf['evaluation']['enabled'])
                items=store.ensure(train+candidates) if needs_features else train+candidates
                valid_ids={i.key for i in items}
                train=[im for im in train if im.key in valid_ids]
                candidates=[im for im in candidates if im.key in valid_ids]
                invalid=skipped+store.invalid
                store.invalid=[]
                info['invalid_count']=len(invalid)
                info['valid_images']=len(train)+len(candidates)
                save_csv(partdir/'invalid_images.csv',invalid,['part','source','path','reason'])
                if not train:
                    raise ValueError('有效原始 Training Set 為空，停止此產品。')
                if not candidates:
                    LOGGER.warning('%s 候選池沒有有效圖片，該產品選樣 0 張。',part)
                shortlist=[];stage1_records=[];stage2_records=[];selection=[];initial_pca=None
                if mode in {'stage1','stage1_stage2'}:
                    budget=k if mode=='stage1' else 2*k
                    shortlist,stage1_records,stage1_reason=select_stage1(train,candidates,store,conf,budget,part)
                    info.update(shortlist_count=len(shortlist),stage1_stop=stage1_reason)
                if mode=='stage1':
                    selection=shortlist
                    info['stop_reason']=info['stage1_stop']
                else:
                    source=candidates if mode=='stage2' else shortlist
                    selection,stage2_records,initial_pca,stop,tau=select_stage2(train,source,store,conf,k,part)
                    info.update(stage2_stop=stop,stop_reason=stop,high_error_tau=tau,
                                initial_pca_dim=int(initial_pca.params['k']) if initial_pca is not None else None)
                info.update(status='ok',selected_count=len(selection),final_train_count=len(train)+len(selection))
                LOGGER.info('%s：原始 %d、候選 %d、Stage1 預選 %d、最終新增 %d/%d、停止原因 %s',
                            part,len(train),len(candidates),len(shortlist),len(selection),k,info['stop_reason'])
                by1={r['image_path']:r for r in stage1_records if r.get('decision')=='accepted'}
                by2={r['image_path']:r for r in stage2_records}
                selected_ids={i.key for i in selection};short_ids={i.key for i in shortlist}
                candidate_records=[]
                for im in candidates:
                    # 選樣前若圖片未列入 Shortlist，其 Stage2 Score 一律留空，不偽造分數。
                    x1=by1.get(str(im.path),{})
                    x2=by2.get(str(im.path),{})
                    candidate_records.append({'part':part,'image_name':im.name,'image_path':str(im.path),
                        'mode':mode,'stage2_strategy':strategy if mode!='stage1' else None,
                        'stage1_shortlisted':im.key in short_ids,'stage1_round':x1.get('round'),
                        'stage1_gain':x1.get('gain'),'stage1_self_gain':x1.get('self_gain'),
                        'stage1_other_gain':x1.get('other_gain'),
                        'stage2_score':x2.get('score'),'stage2_rank':x2.get('rank'),
                        'stage2_threshold_tau':x2.get('threshold_tau'),
                        'selected':im.key in selected_ids,
                        'decision':'selected' if im.key in selected_ids else 'not_selected'})
                if conf['output']['save_csv']:
                    save_csv(partdir/'candidate_scores.csv',candidate_records)
                    save_csv(partdir/'stage1_ranking.csv',stage1_records,['part','round','rank','image_name','image_path','gain','decision'])
                    save_csv(partdir/'stage2_ranking.csv',stage2_records,['part','image_name','image_path','strategy','rank','score','selected'])
                    save_csv(partdir/'selection_decisions.csv',
                             [{'part':part,'selected_order':j+1,'image_name':im.name,
                               'image_path':str(im.path),'mode':mode,'strategy':strategy if mode!='stage1' else 'coverage'}
                              for j,im in enumerate(selection)])
                copy_records=_write_pictures(partdir,train,selection,conf)
                if conf['output']['save_csv']:
                    save_csv(partdir/'copied_files.csv',copy_records,['group','source_path','output_path'])
                viz=conf['visualization']
                if viz['feature_plot']!='none':
                    distribution(train,candidates,selection,store,viz['feature_plot'],plots,
                                 int(conf['selection']['seed']),int(viz['plot_max_images']),shortlist=shortlist)
                if viz['save_iteration'] and stage1_records:
                    iteration(stage1_records,plots/'selection_iteration.png')
                if viz['save_ranking']:
                    if stage2_records:
                        ranking(stage2_records,'score',plots/'stage2_score_ranking.png')
                    if stage1_records:
                        first=[r for r in stage1_records if r['round']==1]
                        ranking(first,'gain',plots/'stage1_first_round_ranking.png')
                if viz['save_heatmap'] and mode!='stage1' and strategy!='random' and initial_pca is not None:
                    patch_error_heatmaps(selection,store,initial_pca,plots,int(viz['max_heatmaps']))
                if conf['evaluation']['enabled']:
                    try:
                        info['evaluation']=evaluate(conf,root,part,train,selection,store)
                    except Exception as exc:
                        info['evaluation']={'status':'error','reason':str(exc)}
                (partdir/'metrics.json').write_text(json.dumps(info,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
            except Exception as exc:
                info.update(status='error',error=f'{type(exc).__name__}: {exc}',stop_reason='error')
                summary['errors'].append({'part':part,'error':info['error']})
                LOGGER.exception('產品 %s 失敗：%s',part,exc)
        if conf['output']['save_csv']:
            save_csv(out/'feature_manifest.csv',store.manifest,['part','source','image','cache','cache_status','patch_count','feature_dim','model_signature'])
        summary.update(time_seconds=round(time.perf_counter()-started,3),cache_hits=store.cache_hits,
                       extracted_images=store.cache_misses,
                       image_success=sum(p.get('valid_images',0) for p in summary['parts'].values()),
                       image_failed=sum(p.get('invalid_count',0) for p in summary['parts'].values()))
        summary['status']='completed' if not summary['errors'] else 'partial_failure'
    except Exception as exc:
        summary['status']='error'
        summary['errors'].append({'part':'*','error':str(exc)})
        LOGGER.exception('無法完成整體選樣：%s',exc)
    finally:
        summary['time_seconds']=round(time.perf_counter()-started,3)
        (out/'run_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
        print('\n================= JDP SELECT SUMMARY =================')
        print(f"Run: {summary['run_id']} | Status: {summary['status']}")
        print(f"Mode: {mode} | Strategy: {strategy} | Backend: {summary['backend']}")
        print(f"K: {k} | Stage1 shortlist target: {summary['shortlist_target']}")
        print(f"Dataset: {root}\nOutput: {out}")
        for part,info in summary['parts'].items():
            print(f"{part}: status={info['status']} | original={info.get('original_count',0)} | "
                  f"candidates={info.get('candidate_count',0)} | shortlist={info.get('shortlist_count',0)} | "
                  f"selected={info.get('selected_count',0)}/{k} | PCA dim={info.get('initial_pca_dim') or 'N/A'} | stop={info.get('stop_reason','unknown')}")
        print(f"Successful images: {summary.get('image_success',0)} | Invalid images: {summary.get('image_failed',0)}")
        print(f"Feature cache hits: {store.cache_hits} | New feature extractions: {store.cache_misses}")
        print(f"Time: {summary['time_seconds']:.1f}s | Errors: {len(summary['errors'])}")
        print('======================================================\n')
    return summary
