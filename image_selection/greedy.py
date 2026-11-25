"""Stage 1：文件定義的完整 Patch Cosine Coverage + 逐張 Greedy。"""
from __future__ import annotations
import hashlib
import json
import logging
from pathlib import Path
import numpy as np
import torch
from numpy.lib.format import open_memmap
from .configuration import resolved_path
from .datasets import ImageItem
from .features import FeatureStore

LOGGER=logging.getLogger(__name__)


def _device(name: str) -> torch.device:
    if name=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('stage1.similarity_device=cuda，但目前 PyTorch 找不到 CUDA。')
    return torch.device('cuda' if name=='cuda' or (name=='auto' and torch.cuda.is_available()) else 'cpu')


@torch.no_grad()
def max_similarities(query: np.ndarray, reference: np.ndarray, query_chunk: int=256,
                     reference_chunk: int=512, device: str='cpu') -> np.ndarray:
    """對每個 Query Patch，計算與 reference 所有 Patch 最大 (1+cos)/2；分塊避免 O(N*M) 記憶體。"""
    if len(reference)==0:
        return np.zeros(len(query),dtype=np.float32)
    dev=_device(device)
    result=np.zeros(len(query),dtype=np.float32)
    # query/reference 均已經逐列 L2 Normalized，且已剔除零向量。
    for st in range(0,len(query),query_chunk):
        q=torch.as_tensor(np.asarray(query[st:st+query_chunk]).copy(),device=dev,dtype=torch.float32)
        best=torch.full((q.shape[0],),-1.0,device=dev)
        for j in range(0,len(reference),reference_chunk):
            r=torch.as_tensor(np.asarray(reference[j:j+reference_chunk]).copy(),device=dev,dtype=torch.float32)
            best=torch.maximum(best,(q@r.T).amax(dim=1))
        result[st:st+len(q)]=((best+1.0)*0.5).clamp(0.0,1.0).cpu().numpy()
    return result


def candidate_coverages(pool: list[np.ndarray], reference: np.ndarray, query_chunk: int,
                        reference_chunk: int, device: str) -> np.ndarray:
    """計算來源影像所有 patches 對整個 U 的最大相似度；U 以各 image patch 累接。"""
    return np.concatenate([max_similarities(q,reference,query_chunk,reference_chunk,device)
                           for q in pool]).astype(np.float32)


def averaged_by_image(values: np.ndarray, offsets: list[int]) -> float:
    """先平均每張圖片內 Gain，再平均整個候選圖片池，而不是 patch 數量加權。"""
    per=[float(np.mean(values[offsets[i]:offsets[i+1]])) for i in range(len(offsets)-1)]
    return float(np.mean(per)) if per else 0.0


def gain_and_details(baseline: np.ndarray, coverage: np.ndarray, offsets: list[int], self_index: int) -> dict:
    inc=np.maximum(coverage-baseline,0.0)
    all_gains=[float(np.mean(inc[offsets[i]:offsets[i+1]])) for i in range(len(offsets)-1)]
    base_cov=averaged_by_image(baseline,offsets)
    after_cov=averaged_by_image(np.maximum(baseline,coverage),offsets)
    return {'gain':float(np.mean(all_gains)),'self_gain':all_gains[self_index],
            'other_gain':float(np.mean([v for i,v in enumerate(all_gains) if i!=self_index])) if len(all_gains)>1 else 0.0,
            'baseline_coverage':base_cov,'post_coverage':after_cov}


def select_stage1(train: list[ImageItem], candidates: list[ImageItem], store: FeatureStore,
                  config: dict, budget: int, part: str) -> tuple[list[ImageItem],list[dict],str]:
    """預算 budget 可以是 K 或 2*K；已預選者暫時更新 Coverage，正式加訓練集由 Controller 決定。"""
    if not candidates:
        return [],[], 'no_candidates'
    opts=config['stage1']
    qchunk=int(opts['query_chunk']);rchunk=int(opts['reference_chunk']); dev=opts['similarity_device']
    pool=[store.normalized(im) for im in candidates]
    offsets=[0]
    for f in pool:
        offsets.append(offsets[-1]+len(f))
    size=offsets[-1]
    # 計算完整 U 每個 Patch 的 baseline m_A(p)。
    baseline=np.zeros(size,dtype=np.float32)
    for item in train:
        initial=candidate_coverages(pool,store.normalized(item),qchunk,rchunk,dev)
        baseline=np.maximum(baseline,initial)
    # 對每張候選影像只計算一次「它對整個 U 的 Coverage」，後續輪次直接重用。
    # 不先抽樣/降維/平均 Features：保留完整 Patch Feature 集合。
    model_info={'part':part,'model_signature':store.model_signature,
                'items':[str(i.path)+':'+store.cache_path(i).name for i in candidates],
                'qchunk':qchunk,'rchunk':rchunk}
    key=hashlib.sha256(json.dumps(model_info,sort_keys=True).encode()).hexdigest()[:24]
    cache_root=resolved_path(opts['coverage_cache_dir'],store.cwd)/part
    cache_root.mkdir(parents=True,exist_ok=True)
    cache_path=cache_root/f'{key}.npy'
    expected_bytes=len(candidates)*size*4
    allowed=float(opts['max_coverage_cache_gb'])*1024**3
    if expected_bytes > allowed:
        raise RuntimeError(f'Stage1 Coverage 快取預估 {expected_bytes/1024**3:.1f} GB，超過上限 {allowed/1024**3:.1f} GB。')
    # 預算檢查之外，也要檢查當前磁碟剩餘空間，避免寫滿公司磁碟。
    import shutil
    if not cache_path.is_file() and shutil.disk_usage(cache_root).free < int(expected_bytes*1.10):
        raise RuntimeError(f'Stage1 Coverage 快取預估需要 {expected_bytes/1024**3:.1f} GB，實際磁碟空間不足。')
    cached=False
    if cache_path.is_file():
        try:
            m=np.load(cache_path,mmap_mode='r')
            cached=(m.shape==(len(candidates),size) and np.isfinite(m[0,:min(size,100)]).all())
        except Exception:
            pass
    if cached:
        coverage=np.load(cache_path,mmap_mode='r')
        LOGGER.info('Stage1 命中 Coverage 快取：%s',cache_path)
    else:
        LOGGER.info('Stage1 建立 Coverage 矩陣：%.2f GB；首次計算可能耗時。',expected_bytes/1024**3)
        temp=cache_path.with_suffix('.tmp.npy')
        coverage=open_memmap(temp,mode='w+',dtype=np.float32,shape=(len(candidates),size))
        for i,f in enumerate(pool):
            coverage[i,:]=candidate_coverages(pool,f,qchunk,rchunk,dev)
            coverage.flush()
            if i%5==0 or i==len(pool)-1:
                LOGGER.info('Stage1 Coverage 已計算 %d/%d 張',i+1,len(pool))
        del coverage
        temp.replace(cache_path)
        coverage=np.load(cache_path,mmap_mode='r')
    eligible=set(range(len(candidates)))
    selected=[]
    log=[]
    epsilon=float(opts['epsilon'])
    reason='max_budget'
    for round_no in range(1,min(budget,len(candidates))+1):
        if not eligible:
            reason='no_candidates';break
        per=[]
        for i in eligible:
            d=gain_and_details(baseline,np.asarray(coverage[i]),offsets,i)
            per.append((i,d))
        per.sort(key=lambda x:(-x[1]['gain'],str(candidates[x[0]].path)))
        for rank,(i,details) in enumerate(per,1):
            log.append({'part':part,'round':round_no,'rank':rank,'image_name':candidates[i].name,
                        'image_path':str(candidates[i].path),**details,
                        'decision':'accepted' if rank==1 and details['gain']>epsilon else 'not_selected'})
        winner,details=per[0]
        if details['gain']<=epsilon:
            reason='gain_threshold';break
        selected.append(candidates[winner])
        eligible.remove(winner)
        baseline=np.maximum(baseline,np.asarray(coverage[winner]))
        LOGGER.info('Stage1 %s 第%d輪：%s Gain=%.8f',part,round_no,candidates[winner].name,details['gain'])
        if len(selected)>=budget:
            reason='max_budget';break
        if not eligible:
            reason='no_candidates';break
    return selected,log,reason
