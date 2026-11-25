"""DINO 原始 Patch Features 快取；Stage 1 視圖與 Stage 2 PCA 共用輸出。"""
from __future__ import annotations
import hashlib
import json
import logging
from pathlib import Path
import numpy as np
from PIL import Image
from .configuration import resolved_path
from .datasets import ImageItem

LOGGER = logging.getLogger(__name__)


def _sha(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,default=str).encode('utf-8')).hexdigest()


def _file_fingerprint(path: Path) -> dict:
    p = Path(path)
    if p.exists():
        st=p.stat()
        return {'path':str(p.resolve()),'size':st.st_size,'mtime_ns':st.st_mtime_ns}
    return {'path':str(p),'missing':True}


class FeatureStore:
    def __init__(self, conf: dict, cwd: Path, extractor_factory= None):
        self.conf = conf
        self.cwd = cwd
        ext = conf['feature_extractor']
        backend=ext['backend']
        settings = {'backend':backend,'resolution':ext['image_res'],'clahe':ext['use_clahe']}
        if backend=='company_anomalyvfm':
            c=ext['company']
            settings.update({'variant':c['variant'],'repo':str(resolved_path(c['subspacead_repo_path'],cwd))})
            repo=resolved_path(c['subspacead_repo_path'],cwd)
            root=resolved_path(c['anomalyvfm_root'],cwd)
            settings['adapter_source']=_file_fingerprint(repo/'src/subspacead/core/anomalyvfm_extractor.py')
            settings['offline_wrapper']=_file_fingerprint(root/'models/dinov2_offline.py')
            for k in ['anomalyvfm_root','anomalyvfm_ckpt','dino_repo_path','dino_weight_path']:
                settings[k]=_file_fingerprint(resolved_path(c[k],cwd))
        else:
            c=ext['public']
            settings.update({'model_id':c['model_id'],'local_model_dir':str(c['local_model_dir']),
                             'local_fingerprint':_file_fingerprint(resolved_path(c['local_model_dir'],cwd))
                             if c['local_model_dir'] else None})
        self.model_signature=_sha(settings)[:20]
        self.cache_dir=resolved_path(ext['cache_dir'],cwd)/self.model_signature
        self.cache_dir.mkdir(parents=True,exist_ok=True)
        self._extractor=None
        self._factory=extractor_factory
        self.cache_hits=0
        self.cache_misses=0
        self.invalid=[]
        self.manifest=[]

    def cache_path(self, item: ImageItem) -> Path:
        # 圖片內容雜湊 + 模型版本，避免新舊權重或圖片更換時誤用 Features。
        stat=item.path.stat()
        # sha256 圖片成本遠低於 DINO 推論，確保內容改變就失效。
        h=hashlib.sha256()
        with item.path.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):
                h.update(b)
        key=_sha({'path':str(item.path),'sha256':h.hexdigest(),'size':stat.st_size})
        return self.cache_dir/f'{key}.npy'

    def ensure(self, items: list[ImageItem]) -> list[ImageItem]:
        """批次萃取未命中快取的圖片，失敗者回報、從有效清單剔除。"""
        ready=[]
        missing=[]
        for item in items:
            path=self.cache_path(item)
            if path.is_file():
                try:
                    x=np.load(path,mmap_mode='r')
                    if x.ndim!=2 or x.shape[0]==0 or x.shape[1]<1:
                        raise ValueError('Feature shape invalid')
                    self.cache_hits+=1
                    self.manifest.append({'part':item.part,'source':item.source,'image':str(item.path),
                                          'cache':str(path),'cache_status':'hit','patch_count':x.shape[0],
                                          'feature_dim':x.shape[1],'model_signature':self.model_signature})
                    ready.append(item)
                    continue
                except Exception:
                    path.unlink(missing_ok=True)
            missing.append((item,path))
        if not missing:
            return ready
        if self._extractor is None:
            if self._factory is not None:
                self._extractor=self._factory()
            else:
                from .extractors import build_extractor
                self._extractor=build_extractor(self.conf,self.cwd)
        bs=int(self.conf['feature_extractor']['batch_size'])
        for i in range(0,len(missing),bs):
            group=missing[i:i+bs]
            try:
                imgs=[]
                for item,_ in group:
                    with Image.open(item.path) as im:
                        imgs.append(im.convert('RGB').copy())
                batches=self._extractor.extract(imgs)
                if batches.shape[0]!=len(group):
                    raise ValueError(f'Extractor 回傳 Batch Size 不符：{batches.shape}')
            except Exception as exc:
                # GPU OOM 等 Batch 故障，改以單張找出實際失敗的檔案。
                LOGGER.warning('圖片 Batch 失敗，改用單張處理：%s',exc)
                batches=None
            for j,(item,path) in enumerate(group):
                try:
                    if batches is None:
                        with Image.open(item.path) as im:
                            f=self._extractor.extract([im.convert('RGB')])[0]
                    else:
                        f=batches[j]
                    f=np.asarray(f,dtype=np.float32).reshape(-1,f.shape[-1])
                    if not np.isfinite(f).all() or f.shape[0]<1:
                        raise ValueError('Feature 含 NaN/Inf 或沒有有效 Patch')
                    # 先寫入 tmp，再原子搬移，避免中途中斷留下損壞的 .npy。
                    tmp=path.with_suffix('.tmp.npy')
                    np.save(tmp,f)
                    tmp.replace(path)
                    self.cache_misses+=1
                    ready.append(item)
                    self.manifest.append({'part':item.part,'source':item.source,'image':str(item.path),
                                           'cache':str(path),'cache_status':'new','patch_count':f.shape[0],
                                           'feature_dim':f.shape[1],'model_signature':self.model_signature})
                except Exception as exc:
                    self.invalid.append({'part':item.part,'source':item.source,'path':str(item.path),
                                         'reason':f'Feature Extraction: {type(exc).__name__}: {exc}'})
                    LOGGER.error('特徵萃取失敗 %s：%s',item.path,exc)
        return ready

    def get(self,item: ImageItem, mmap: bool=True) -> np.ndarray:
        return np.load(self.cache_path(item),mmap_mode='r' if mmap else None)

    def normalized(self,item: ImageItem) -> np.ndarray:
        f=np.asarray(self.get(item),dtype=np.float32)
        norm=np.linalg.norm(f,axis=1,keepdims=True)
        keep=(norm[:,0]>1e-12)&np.isfinite(norm[:,0])
        if not np.any(keep):
            raise ValueError(f'{item.path}: 全部 Patch Feature 都是零向量')
        return f[keep]/norm[keep]
