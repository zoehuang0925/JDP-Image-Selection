"""Fake DINO + Fake SubspaceAD PCA 下的端到端輸出測試（不需要模型權重）。"""
from pathlib import Path
import sys
import types
import numpy as np
import yaml
from PIL import Image
import pytest
import image_selection.pipeline as pipeline
from image_selection.features import FeatureStore
from image_selection.datasets import ImageItem


class FakeExtractor:
    def extract(self,images):
        # 每張 4 Patches × 4D。用圖片 RGB 平均產生可重現特徵。
        outputs=[]
        for image in images:
            arr=np.asarray(image.convert('RGB'),dtype=float)
            s=arr.mean(axis=(0,1))/255.0
            x=np.array([[s[0]+.1,s[1]+.1,s[2]+.1,0.05],
                        [s[0]+.1,s[1]+.15,s[2]+.1,.5],
                        [s[0]+.2,s[1]+.25,s[2]+.2,1.0],
                        [s[0]+.3,s[1]+.35,s[2]+.3,.15]],dtype=np.float32)
            outputs.append(x.reshape(2,2,4))
        return np.asarray(outputs)


class FakePCAModel:
    def __init__(self,k=None,ev=.99,whiten=False):
        self.k=k;self.ev=ev;self.whiten=whiten
    def fit(self,gen,feature_dim,total_tokens,num_batches):
        feats=np.vstack(list(gen())).astype(float)
        mu=np.mean(feats,axis=0)
        cov=np.cov((feats-mu).T)
        vals,vec=np.linalg.eigh(cov)
        vals=vals[::-1];vec=vec[:,::-1]
        if self.k is None:
            c=np.cumsum(np.maximum(vals,0)); k=int(np.searchsorted(c/c[-1],self.ev))+1
        else:k=self.k
        return {'mu':mu,'components':vec[:,:k],'k':k,'eigvals':vals[:k]}


def _make_config(tmp_path,mode,strategy='p90'):
    c=yaml.safe_load((Path(__file__).parents[1]/'config.yaml').read_text())
    c['dataset']['root']=str(tmp_path/'dataset')
    c['dataset']['parts']=['bottle']
    c['feature_extractor']['cache_dir']=str(tmp_path/'feature_cache')
    c['stage1']['coverage_cache_dir']=str(tmp_path/'coverage_cache')
    c['stage1']['similarity_device']='cpu'
    c['stage1']['query_chunk']=2;c['stage1']['reference_chunk']=2
    c['selection']['mode']=mode;c['selection']['k']=2
    c['stage2']['strategy']=strategy
    c['pca']['pca_dim']=2;c['pca']['pca_ev']=None
    c['visualization']['feature_plot']='pca'
    c['visualization']['max_heatmaps']=2
    c['output']['root']=str(tmp_path/'results')
    # 此測試用 fake PCA，不需要真正存在的 Repo
    return c


def _data(tmp_path):
    root=tmp_path/'dataset'/'bottle'
    for i in range(2):
        p=root/'train'/'good'/f't{i}.png';p.parent.mkdir(parents=True,exist_ok=True)
        Image.new('RGB',(14,14),(30+i*10,60+i*15,20)).save(p)
    for i in range(4):
        p=root/'candidates'/'good'/f'c{i}.png';p.parent.mkdir(parents=True,exist_ok=True)
        Image.new('RGB',(14,14),(20+i*40,40+i*20,10+i*30)).save(p)
    # 非圖片檔應自動忽略；損壞圖片記錄失敗。
    (root/'candidates'/'good'/'bad.png').write_bytes(b'broken')


@pytest.mark.parametrize('mode,strategy',[
    ('stage1','p90'),('stage2','random'),('stage2','mean'),
    ('stage2','p90'),('stage2','high_error_ratio'),('stage1_stage2','p90')])
def test_pipeline_smoke(tmp_path,monkeypatch,mode,strategy):
    _data(tmp_path)
    # 依賴反轉：只替換 DINO / PCA 模型，pipeline、CSV、圖表、copy、快取都是真正的程式。
    monkeypatch.setattr(pipeline,'FeatureStore',lambda conf,cwd:FeatureStore(conf,cwd,extractor_factory=FakeExtractor))
    mod=types.ModuleType('src.subspacead.core.pca');mod.PCAModel=FakePCAModel
    for name in ['src','src.subspacead','src.subspacead.core']:
        monkeypatch.setitem(sys.modules,name,types.ModuleType(name))
    monkeypatch.setitem(sys.modules,'src.subspacead.core.pca',mod)
    c=_make_config(tmp_path,mode,strategy)
    result=pipeline.run(c,tmp_path)
    assert result['status']=='completed',result['errors']
    info=result['parts']['bottle']
    assert info['status']=='ok'
    assert info['original_count']==2 and info['candidate_count']==4
    assert info['invalid_count']==1
    assert 0<=info['selected_count']<=2
    assert result['image_success']==6 and result['image_failed']==1
    root=Path(result['output_root'])
    for x in ['run_summary.json','run_config.json','feature_manifest.csv','run.log',
              'bottle/candidate_scores.csv','bottle/invalid_images.csv',
              'bottle/selection_decisions.csv','bottle/metrics.json',
              'bottle/plots/feature_distribution_pca.png']:
        assert (root/x).is_file(),f'missing {x}'
    copied=list((root/'bottle'/'Merged_Training_Set').glob('*'))
    assert len(copied)==2+info['selected_count']
    assert len(list((root/'bottle'/'Selected_Only').glob('*')))==info['selected_count']
    if mode=='stage1_stage2':
        assert info['shortlist_count']<=4
    if strategy!='random' and mode!='stage1' and info['selected_count']:
        assert list((root/'bottle'/'plots').glob('*_pca_error.png'))
