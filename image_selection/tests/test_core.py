"""快速邏輯測試：不需要下載任何 DINO 權重。"""
from pathlib import Path
import numpy as np
import pytest
import yaml
from image_selection.configuration import load_config
from image_selection.greedy import max_similarities, gain_and_details, select_stage1
from image_selection.stage2 import summarize_error, select_stage2
from image_selection.datasets import ImageItem


class ToyStore:
    def __init__(self,tmp_path,data):
        self.cwd=tmp_path
        self.model_signature='toy-v1'
        self.cache=tmp_path/'cache'; self.cache.mkdir(exist_ok=True)
        self.data=data
        self.mapping={}
        for i,(path,a) in enumerate(data.items()):
            f=self.cache/f'{i}.npy'; np.save(f,np.asarray(a,dtype=np.float32)); self.mapping[path]=f
    def get(self,item,mmap=True):
        return np.load(self.mapping[item.key])
    def normalized(self,item):
        f=self.get(item)
        n=np.linalg.norm(f,axis=1,keepdims=True)
        return f[n[:,0]>1e-12]/n[n[:,0]>1e-12]
    def cache_path(self,item):
        return self.mapping[item.key]


def cfg(tmp_path):
    p=Path(__file__).parents[1]/'config.yaml'
    c=yaml.safe_load(p.read_text())
    c['stage1']['coverage_cache_dir']=str(tmp_path/'coverage')
    c['stage1']['similarity_device']='cpu'
    c['stage1']['query_chunk']=2
    c['stage1']['reference_chunk']=2
    c['visualization']['feature_plot']='none'
    return c


def test_config_modes(tmp_path):
    conf=cfg(tmp_path)
    p=tmp_path/'conf.yaml';p.write_text(yaml.safe_dump(conf))
    assert load_config(p)['selection']['mode']=='stage1_stage2'
    conf['pca']['pca_dim']=10
    p.write_text(yaml.safe_dump(conf))
    with pytest.raises(ValueError,match='只能選一種'):
        load_config(p)


def test_cosine_and_gain():
    q=np.array([[1,0],[0,1]],dtype=np.float32)
    ref=np.array([[1,0]],dtype=np.float32)
    got=max_similarities(q,ref,query_chunk=1,reference_chunk=1,device='cpu')
    assert np.allclose(got,[1.0,0.5])
    d=gain_and_details(np.array([.9,.3,.6,.4],dtype=np.float32),
                       np.array([1.,1.,.65,.8],dtype=np.float32),[0,2,4],0)
    assert d['gain']==pytest.approx(.3125)
    assert d['self_gain']==pytest.approx(.4)
    assert d['other_gain']==pytest.approx(.225)


def test_stage1_iterative(tmp_path):
    a=ImageItem(tmp_path/'train.png','toy','train')
    b=ImageItem(tmp_path/'b.png','toy','candidate')
    c=ImageItem(tmp_path/'c.png','toy','candidate')
    for x in [a,b,c]:x.path.write_bytes(b'x')
    data={a.key:[[1,0],[1,0]],b.key:[[0,1],[0,1]],c.key:[[-1,0],[-1,0]]}
    store=ToyStore(tmp_path,data)
    out,logs,reason=select_stage1([a],[b,c],store,cfg(tmp_path),2,'toy')
    assert len(out)==2
    assert {im.key for im in out}=={b.key,c.key}
    assert reason=='max_budget'
    assert sorted({x['round'] for x in logs})==[1,2]
    # 同一組相同資料重跑，應重用 Coverage 快取；結果一致。
    out2,_,_=select_stage1([a],[b,c],store,cfg(tmp_path),2,'toy')
    assert [x.key for x in out]==[x.key for x in out2]


def test_stage2_error_aggregations():
    a=np.array([1,2,3,9,15],dtype=float)
    assert summarize_error(a,'mean')==pytest.approx(6)
    assert summarize_error(a,'p90')==pytest.approx(12.6)
    assert summarize_error(a,'high_error_ratio',8)==pytest.approx(.4)


def test_random_k(tmp_path):
    data={};items=[]
    for j in range(10):
        p=tmp_path/f'c{j}.png';p.write_bytes(b'x')
        data[str(p)]=np.array([[j,1],[1,j]])
        items.append(ImageItem(p,'toy','candidate'))
    t=ImageItem(tmp_path/'t.png','toy','train')
    t.path.write_bytes(b'x');data[t.key]=np.array([[1,0],[0,1]])
    st=ToyStore(tmp_path,data)
    c=cfg(tmp_path);c['stage2']['strategy']='random'
    chosen,records,model,reason,_=select_stage2([t],items,st,c,4,'toy')
    assert len(chosen)==4 and len(records)==10 and model is None and reason=='max_budget'
    chosen2,_,_,_,_=select_stage2([t],items,st,c,4,'toy')
    assert chosen==chosen2


def test_public_backend_download_flag_and_token_slicing(tmp_path,monkeypatch):
    """用假 Transformers 模型檢查首次下載參數與 CLS/Register Token 移除。"""
    import sys
    import types
    import torch
    from PIL import Image
    from image_selection.extractors import PublicDINOv2
    calls=[]
    class DummyProcessor:
        @classmethod
        def from_pretrained(cls,model_id,**kwargs):
            calls.append(('processor',model_id,kwargs))
            return cls()
        def __call__(self,images,**kw):
            return {'pixel_values':torch.zeros(len(images),3,56,56)}
    class DummyModel:
        config=types.SimpleNamespace(patch_size=14,num_register_tokens=4,hidden_size=4)
        @classmethod
        def from_pretrained(cls,model_id,**kwargs):
            calls.append(('model',model_id,kwargs))
            return cls()
        def eval(self): return self
        def to(self,device):return self
        def __call__(self,**kwargs):
            b=kwargs['pixel_values'].shape[0]
            # 1 CLS + 4 Register + 16 Patch；Patch 值必須從索引 5 開始。
            tokens=torch.arange(21,dtype=torch.float32)[None,:,None].repeat(b,1,4)
            return types.SimpleNamespace(last_hidden_state=tokens)
    monkeypatch.setitem(sys.modules,'transformers',types.SimpleNamespace(
        AutoImageProcessor=DummyProcessor,AutoModel=DummyModel))
    c=cfg(tmp_path)
    c['feature_extractor']['image_res']=56
    c['feature_extractor']['backend']='public_dinov2'
    c['feature_extractor']['public']['auto_download']=True
    ext=PublicDINOv2(c,tmp_path)
    result=ext.extract([Image.new('RGB',(56,56))])
    assert result.shape==(1,4,4,4)
    assert result[0,0,0,0]==5.0 and result[0,-1,-1,0]==20.0
    assert all(x[2]['local_files_only'] is False for x in calls)
