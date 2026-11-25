"""兩個可互換的 Frozen DINO2/AnomalyVFM Feature Extractor。"""
from __future__ import annotations
import logging
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .configuration import resolved_path

LOGGER = logging.getLogger(__name__)


def _clahe_rgb(img: Image.Image) -> Image.Image:
    import cv2
    rgb = np.asarray(img.convert('RGB'))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L,a,b = cv2.split(lab)
    L = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(L)
    return Image.fromarray(cv2.cvtColor(cv2.merge((L,a,b)),cv2.COLOR_LAB2RGB))


class CompanyAnomalyVFM:
    """直接 import 原三篩的 Adapter，避免重新實作權重或前處理。"""
    def __init__(self, conf: dict, cwd: Path):
        ext = conf['feature_extractor']
        c = ext['company']
        repo = resolved_path(c['subspacead_repo_path'],cwd)
        if not (repo/'src/subspacead/core/anomalyvfm_extractor.py').is_file():
            raise FileNotFoundError(f'找不到三篩 AnomalyVFM Adapter：{repo}')
        # 給獨立模組提供原三篩 package；原有主程式不需要執行。
        if str(repo) not in sys.path:
            sys.path.insert(0,str(repo))
        from src.subspacead.core.anomalyvfm_extractor import AnomalyVFMFeatureExtractor
        self.impl = AnomalyVFMFeatureExtractor(
            anomalyvfm_root=str(resolved_path(c['anomalyvfm_root'],cwd)),
            anomalyvfm_ckpt=str(resolved_path(c['anomalyvfm_ckpt'],cwd)),
            dino_repo_path=str(resolved_path(c['dino_repo_path'],cwd)),
            dino_weight_path=str(resolved_path(c['dino_weight_path'],cwd)),
            variant=c['variant'],
        )
        self.res = int(ext['image_res'])
        self.clahe = bool(ext['use_clahe'])

    def extract(self, images: list[Image.Image]) -> np.ndarray:
        # a1 最終層特徵，layers/agg 不參與 a1 決策；API 參數仍需傳入。
        tokens,grid,_ = self.impl.extract_tokens(
            images,res=self.res,layers=[-12,-13,-14,-15,-16,-17,-18],
            agg_method='mean',docrop=False,use_clahe=self.clahe,
        )
        if tuple(grid)!=(48,48):
            raise ValueError(f'公司 DINO Patch Grid 不符預期：{grid}')
        return tokens.astype(np.float32,copy=False)


class PublicDINOv2:
    """官方公開 DINOv2；第一次執行允許下載並由 Hugging Face 快取。"""
    def __init__(self, conf: dict, cwd: Path):
        from transformers import AutoModel, AutoImageProcessor
        ext = conf['feature_extractor']
        c = ext['public']
        self.res = int(ext['image_res'])
        self.clahe = bool(ext['use_clahe'])
        auto = bool(c['auto_download'])
        local = resolved_path(c.get('local_model_dir'),cwd)
        model_id = str(local) if local else c['model_id']
        cache_dir = resolved_path(c.get('hf_cache_dir'),cwd)
        kw = {'local_files_only':not auto}
        if cache_dir is not None:
            kw['cache_dir'] = str(cache_dir)
        LOGGER.info('載入公開 DINOv2：%s (auto_download=%s)',model_id,auto)
        self.processor = AutoImageProcessor.from_pretrained(model_id,**kw)
        self.model = AutoModel.from_pretrained(model_id,**kw)
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        cfg = self.model.config
        self.patch_size = int(cfg.patch_size)
        self.num_reg = int(getattr(cfg,'num_register_tokens',0))
        self.dim = int(cfg.hidden_size)
        if self.res % self.patch_size:
            raise ValueError('影像尺寸不能被 patch_size 整除。')

    @torch.no_grad()
    def extract(self, images: list[Image.Image]) -> np.ndarray:
        images = [_clahe_rgb(im) if self.clahe else im.convert('RGB') for im in images]
        processor = self.processor(
            images=images,return_tensors='pt',do_resize=True,
            size={'height':self.res,'width':self.res},
            do_center_crop=False,
        )
        processor = {k:v.to(self.device) for k,v in processor.items()}
        seq = self.model(**processor).last_hidden_state
        n = self.res//self.patch_size
        count=n*n
        # CLS 與 Register Token 在序列前端，與三篩 H0 FeatureExtractor 取法相同。
        patch = seq[:,1+self.num_reg:1+self.num_reg+count,:]
        if patch.shape[1]!=count:
            raise ValueError(f'公開 DINOv2 回傳 Patch 數不符：{patch.shape}')
        return patch.detach().float().reshape(len(images),n,n,self.dim).cpu().numpy()


def build_extractor(conf: dict, cwd: Path):
    if conf['feature_extractor']['backend']=='company_anomalyvfm':
        return CompanyAnomalyVFM(conf,cwd)
    return PublicDINOv2(conf,cwd)
