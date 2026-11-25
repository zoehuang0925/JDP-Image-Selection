"""Stage 2：與三篩 SubspaceAD 相同的 PCA 建模 + 四種批次選樣策略。"""
from __future__ import annotations
import logging
import math
import sys
from pathlib import Path
import numpy as np
from .configuration import resolved_path
from .datasets import ImageItem
from .features import FeatureStore

LOGGER=logging.getLogger(__name__)


class SubspacePCA:
    """呼叫原三篩 PCAModel：two-pass streaming covariance / eigen decomposition。"""
    def __init__(self,config:dict,cwd:Path):
        self.config=config
        self.cwd=cwd
        self.params=None

    def fit(self,items:list[ImageItem],store:FeatureStore):
        if not items:
            raise ValueError('PCA 至少需要一張原始正常訓練圖片。')
        # 若選樣是獨立 Repo 下的子模組，也可由 Config 尋找三篩根目錄。
        root=resolved_path(self.config['feature_extractor']['company']['subspacead_repo_path'],self.cwd)
        if not (root/'src/subspacead/core/pca.py').is_file():
            root=self.cwd
        if str(root) not in sys.path:
            sys.path.insert(0,str(root))
        try:
            from src.subspacead.core.pca import PCAModel
        except Exception as exc:
            raise ImportError('無法匯入三篩 SubspaceAD PCAModel，請確認程式放在原 Repo 或修正 subspacead_repo_path。') from exc
        counts=[len(store.get(im)) for im in items]
        total=sum(counts)
        dim=store.get(items[0]).shape[1]
        if total<2:
            raise ValueError('PCA 至少需要兩個 Patch 特徵。')
        if any(store.get(im).shape[1]!=dim for im in items):
            raise ValueError('同一產品的 Feature Dimension 不一致，請檢查快取與模型。')
        p=self.config['pca']
        if p['pca_dim'] is not None and int(p['pca_dim'])>min(dim,total-1):
            raise ValueError(f"pca_dim={p['pca_dim']} 超過樣本數可支援的主成分上限 {min(dim,total-1)}")
        def gen():
            for im in items:
                yield np.asarray(store.get(im),dtype=np.float32)
        model=PCAModel(k=p['pca_dim'],ev=p['pca_ev'],whiten=bool(p['whiten']))
        self.params=model.fit(gen,feature_dim=dim,total_tokens=total,num_batches=len(items))
        self.params['k']=int(self.params['k'])
        LOGGER.info('SubspaceAD PCA 訓練：%d 張／%d Patches；實際維度 %d',len(items),total,self.params['k'])
        return self

    def patch_errors(self,features:np.ndarray, batch:int=8192) -> np.ndarray:
        """與三篩 scoring.py 的 reconstruction 相同：每個 Patch 的平方 L2 重建殘差。"""
        if self.params is None:
            raise ValueError('尚未 fit PCA。')
        p=self.params
        mu=p['mu']
        C=p['components'][:,:p['k']]
        drop=int(self.config['pca']['drop_k'])
        res=np.empty(len(features),dtype=np.float64)
        for i in range(0,len(features),batch):
            x=np.asarray(features[i:i+batch],dtype=np.float64)
            z=(x-mu)@C
            if drop:
                z[:,:min(drop,z.shape[1])]=0.0
            recon=z@C.T+mu
            res[i:i+len(x)]=np.sum((x-recon)**2,axis=1)
        return res


def summarize_error(errors:np.ndarray,strategy:str,tau:float|None=None) -> float:
    if len(errors)==0:
        raise ValueError('沒有 Patch Errors。')
    if strategy=='mean':
        return float(np.mean(errors))
    if strategy=='p90':
        return float(np.percentile(errors,90))
    if strategy=='high_error_ratio':
        if tau is None:
            raise ValueError('High-error Ratio 必須提供 τ。')
        return float(np.mean(errors>tau))
    raise ValueError(f'不支援的 Error Aggregation Strategy：{strategy}')


def threshold_from_train(conf:dict,train:list[ImageItem],store:FeatureStore,pca:SubspacePCA) -> float:
    st=conf['stage2']
    if st['high_error_threshold_mode']=='fixed':
        return float(st['high_error_threshold'])
    errors=np.concatenate([pca.patch_errors(store.get(im)) for im in train])
    LOGGER.warning('High-error Ratio τ 使用原始 Train Errors 的 in-sample Q%.3f，正式研究建議另驗證校準偏差。',st['high_error_quantile'])
    return float(np.quantile(errors,float(st['high_error_quantile'])))


def select_stage2(train:list[ImageItem],candidates:list[ImageItem],store:FeatureStore,
                  conf:dict,k:int,part:str) -> tuple[list[ImageItem],list[dict],SubspacePCA|None,str,float|None]:
    strategy=conf['stage2']['strategy']
    if not candidates:
        return [],[],None,'no_candidates',None
    if strategy=='random':
        rng=np.random.default_rng(int(conf['selection']['seed']))
        perm=rng.permutation(len(candidates))
        ordered=[candidates[i] for i in perm]
        chosen=ordered[:k]
        records=[]
        for rank,im in enumerate(ordered,1):
            records.append({'part':part,'image_name':im.name,'image_path':str(im.path),
                            'strategy':strategy,'rank':rank,'score':None,'threshold_tau':None,
                            'selected':rank<=k})
        return chosen,records,None,'max_budget' if len(chosen)>=k else 'no_candidates',None
    pca=SubspacePCA(conf,store.cwd).fit(train,store)
    tau=threshold_from_train(conf,train,store,pca) if strategy=='high_error_ratio' else None
    scores=[]
    for im in candidates:
        errors=pca.patch_errors(store.get(im))
        score=summarize_error(errors,strategy,tau)
        scores.append((im,score))
    scores.sort(key=lambda x:(-x[1],str(x[0].path)))
    chosen=[p[0] for p in scores[:k]]
    records=[{'part':part,'image_name':im.name,'image_path':str(im.path),
              'strategy':strategy,'rank':rank,'score':score,'threshold_tau':tau,
              'selected':rank<=k} for rank,(im,score) in enumerate(scores,1)]
    return chosen,records,pca,'max_budget' if len(chosen)>=k else 'no_candidates',tau
