"""選樣完成後的可選獨立評估。測試資料絕不參與選樣或門檻校準。"""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
from PIL import Image
from .datasets import ImageItem,find_images
from .features import FeatureStore
from .stage2 import SubspacePCA

LOGGER=logging.getLogger(__name__)


def _image_score(pca:SubspacePCA,features:np.ndarray,frac:float)->float:
    errors=pca.patch_errors(features)
    n=max(1,int(np.ceil(len(errors)*frac)))
    return float(np.mean(np.partition(errors,-n)[-n:]))


def evaluate(conf:dict,root:Path,part:str,train:list[ImageItem],selected:list[ImageItem],store:FeatureStore):
    ev=conf['evaluation']
    test_dir=root/part/ev['test_rel']
    if not test_dir.is_dir():
        return {'status':'skipped','reason':f'找不到獨立測試資料夾 {test_dir}'}
    test=[]
    for cls_dir in sorted(test_dir.iterdir()):
        if cls_dir.is_dir():
            for path in find_images(cls_dir,True):
                test.append((ImageItem(path,part,'test'),0 if cls_dir.name.lower() in {'good','normal'} else 1))
    if not test:
        return {'status':'skipped','reason':'Test Set 沒有圖片'}
    val_dir=root/part/ev['val_rel']
    val_good=[ImageItem(p,part,'val') for p in find_images(val_dir/'good',True)]
    # 所有 test 與 val 必須獨立，且不參與 DINO 之外的選樣步驟。
    usable=store.ensure([v[0] for v in test]+val_good)
    usable_ids={v.key for v in usable}
    test=[v for v in test if v[0].key in usable_ids]
    val_good=[v for v in val_good if v.key in usable_ids]
    if not test:
        return {'status':'skipped','reason':'Test Set 影像無法萃取特徵'}
    result={'status':'ok','test_count':len(test),'val_good_count':len(val_good)}
    for label,items in [('initial',train),('final',train+selected)]:
        pca=SubspacePCA(conf,store.cwd).fit(items,store)
        score=np.asarray([_image_score(pca,store.get(im),float(ev['top_patch_fraction'])) for im,_ in test])
        y=np.asarray([lbl for _,lbl in test],dtype=int)
        threshold=ev['threshold']
        if threshold is None and val_good:
            val_scores=[_image_score(pca,store.get(im),float(ev['top_patch_fraction'])) for im in val_good]
            threshold=float(np.quantile(val_scores,1-float(ev['target_fpr'])))
        record={'pca_dim':int(pca.params['k']),'threshold':threshold}
        if len(np.unique(y))==2:
            from sklearn.metrics import roc_auc_score
            record['image_auroc']=float(roc_auc_score(y,score))
        if threshold is not None:
            pred=score>=float(threshold)
            normal=y==0; anomaly=y==1
            record['fpr']=float(np.mean(pred[normal])) if np.any(normal) else None
            record['recall']=float(np.mean(pred[anomaly])) if np.any(anomaly) else None
        else:
            record['fpr']=None;record['recall']=None
            record['note']='無固定閾值與 val/good，FPR/Recall 不計算；未用測試集選閾值。'
        result[label]=record
    return result
