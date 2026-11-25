"""不同模式可切換的 2D 特徵分布、Ranking、Greedy Iteration 與 Patch Error Heatmap。"""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
from .datasets import ImageItem
from .features import FeatureStore
from .stage2 import SubspacePCA

LOGGER=logging.getLogger(__name__)


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def distribution(train:list[ImageItem],candidates:list[ImageItem],selected:list[ImageItem],
                 store:FeatureStore,mode:str,save_to:Path,seed:int,max_images:int,
                 shortlist:list[ImageItem]|None=None) -> list[Path]:
    """只為畫圖做 mean patch aggregation；選樣本身始終使用所有 Patch Features。"""
    if mode=='none':
        return []
    rng=np.random.default_rng(seed)
    # 使用固定 seed 從候選池抽樣，若未達上限則全部顯示；務必包含最終選中者。
    subset=list(candidates)
    if len(subset)>max_images:
        selkeys={im.key for im in selected}
        keep=[im for im in subset if im.key in selkeys]
        others=[im for im in subset if im.key not in selkeys]
        slots=max(0,max_images-len(keep))
        if len(others)>slots:
            others=[others[i] for i in sorted(rng.choice(len(others),slots,replace=False))]
        subset=keep+others
    all_items=list(train)+subset
    if len(all_items)<3:
        LOGGER.warning('PCA 2D / UMAP 圖片總數小於 3，略過分布圖。')
        return []
    x=np.stack([np.mean(store.get(im),axis=0) for im in all_items]).astype(np.float32)
    def plot(z:np.ndarray,label:str) -> Path:
        plt=_plt()
        fig,ax=plt.subplots(figsize=(8,6))
        selected_set={j.key for j in selected}
        preselected_set={j.key for j in (shortlist or [])}
        cats=['initial' if im.source=='train' else 'selected' if im.key in selected_set
              else 'shortlisted_not_selected' if im.key in preselected_set else 'candidate' for im in all_items]
        for cat,color,mark in [('candidate','#b9bcc7','o'),('initial','#4e7ac7','^'),
                               ('shortlisted_not_selected','#946ebd','x'),('selected','#e88648','*')]:
            idx=[i for i,c in enumerate(cats) if c==cat]
            if idx:
                ax.scatter(z[idx,0],z[idx,1],c=color,marker=mark,alpha=.82,
                           s=130 if cat=='selected' else 40,label=cat)
        ax.set_title(f'{label} image feature distribution (visualization only)')
        ax.set_xlabel('component 1');ax.set_ylabel('component 2')
        ax.legend(loc='best');fig.tight_layout()
        out=save_to/f'feature_distribution_{label.lower()}.png'
        fig.savefig(out,dpi=160);plt.close(fig)
        return out
    outputs=[]
    if mode in {'pca','both'}:
        from sklearn.decomposition import PCA
        outputs.append(plot(PCA(n_components=2,random_state=seed).fit_transform(x),'PCA'))
    if mode in {'umap','both'}:
        try:
            import umap
        except ImportError as exc:
            raise ImportError('feature_plot=umap/both 需要安裝 umap-learn。') from exc
        n_neighbors=max(2,min(15,len(all_items)-1))
        z=umap.UMAP(n_components=2,random_state=seed,n_neighbors=n_neighbors,init='random').fit_transform(x)
        outputs.append(plot(z,'UMAP'))
    return outputs


def ranking(records:list[dict],value_field:str,out:Path,top:int=30):
    if not records:
        return
    plt=_plt()
    records=sorted(records,key=lambda r:(int(r.get('rank',999999)),str(r.get('image_name',''))))[:top]
    random_rank=value_field=='score' and all(x.get('score') is None for x in records)
    scores=[float(len(records)-int(x['rank'])+1) if random_rank
            else float(x.get(value_field) or 0) for x in records]
    labels=[r.get('image_name','')[:28] for r in records]
    fig,ax=plt.subplots(figsize=(9,max(4,len(scores)*.3)))
    ax.barh(range(len(scores)),scores,color='#6595be')
    ax.set_yticks(range(len(labels)),labels,fontsize=8)
    ax.invert_yaxis();ax.set_xlabel('Random draw order (higher = earlier)' if random_rank else value_field)
    ax.set_title('Random candidate order (seeded)' if random_rank else 'Candidate score ranking')
    fig.tight_layout();fig.savefig(out,dpi=160);plt.close(fig)


def iteration(records:list[dict],out:Path):
    picked=sorted((r for r in records if r.get('decision')=='accepted'),key=lambda r:int(r['round']))
    if not picked:
        return
    plt=_plt();fig,ax=plt.subplots(figsize=(8,4))
    ax.plot([r['round'] for r in picked],[r['gain'] for r in picked],marker='o')
    ax.set_title('Stage1 accepted coverage gain by iteration')
    ax.set_xlabel('Greedy round');ax.set_ylabel('Global coverage gain')
    ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out,dpi=160);plt.close(fig)


def patch_error_heatmaps(images:list[ImageItem],store:FeatureStore,pca:SubspacePCA,
                         output_dir:Path,max_count:int):
    if pca is None:
        return []
    from PIL import Image
    plt=_plt()
    outs=[]
    for im in images[:max_count]:
        f=store.get(im)
        errors=pca.patch_errors(f)
        side=int(np.sqrt(len(errors)))
        if side*side!=len(errors):
            LOGGER.warning('Patch 數不是正方形，無法輸出 Heatmap：%s',im.path)
            continue
        with Image.open(im.path) as pil:
            img=np.asarray(pil.convert('RGB'))
        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,4))
        ax1.imshow(img);ax1.set_title('Original');ax1.axis('off')
        heat=ax2.imshow(errors.reshape(side,side),cmap='inferno')
        ax2.set_title('Initial PCA raw patch squared error');ax2.axis('off')
        fig.colorbar(heat,ax=ax2,fraction=.04,pad=.03)
        fig.tight_layout()
        import hashlib
        name=hashlib.sha256(str(im.path).encode()).hexdigest()[:8]
        path=output_dir/f'{im.path.stem}_{name}_pca_error.png'
        fig.savefig(path,dpi=160);plt.close(fig);outs.append(path)
    return outs
