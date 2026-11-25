"""YAML Config 讀取與一致性檢查；保留完整的原始設定以供重現。"""
from __future__ import annotations
from pathlib import Path
import copy
import yaml

MODES = {'stage1', 'stage2', 'stage1_stage2'}
STRATEGIES = {'random', 'mean', 'p90', 'high_error_ratio'}
PLOTS = {'none', 'pca', 'umap', 'both'}
BACKENDS = {'company_anomalyvfm', 'public_dinov2'}


def load_config(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    with path.open('r', encoding='utf-8') as fh:
        conf = yaml.safe_load(fh)
    if not isinstance(conf, dict):
        raise ValueError('設定檔必須是 YAML 物件。')
    conf = copy.deepcopy(conf)
    conf['_config_file'] = str(path)
    conf['_config_dir'] = str(path.parent)
    sel, ds, ext, stage1, stage2, pca, viz = (conf[x] for x in
        ['selection', 'dataset', 'feature_extractor', 'stage1', 'stage2', 'pca', 'visualization'])
    if sel['mode'] not in MODES:
        raise ValueError(f"selection.mode 必須是 {sorted(MODES)}")
    if not isinstance(sel['k'], int) or sel['k'] <= 0:
        raise ValueError('selection.k 必須是正整數（新增影像上限）。')
    if stage2['strategy'] not in STRATEGIES:
        raise ValueError(f"stage2.strategy 必須是 {sorted(STRATEGIES)}")
    if ext['backend'] not in BACKENDS:
        raise ValueError(f"feature_extractor.backend 必須是 {sorted(BACKENDS)}")
    if viz['feature_plot'] not in PLOTS:
        raise ValueError(f"visualization.feature_plot 必須是 {sorted(PLOTS)}")
    if ds['parts'] != 'all' and (not isinstance(ds['parts'], list) or not ds['parts']):
        raise ValueError('dataset.parts 請使用 all 或非空 list，例如 [bottle]。')
    if float(stage1['epsilon']) < 0:
        raise ValueError('stage1.epsilon 不可為負。')
    if any(int(stage1[k]) <= 0 for k in ['query_chunk','reference_chunk']):
        raise ValueError('stage1.query_chunk/reference_chunk 必須為正整數。')
    if stage1['similarity_device'] not in {'auto','cpu','cuda'}:
        raise ValueError('stage1.similarity_device 必須是 auto/cpu/cuda。')
    if stage2['high_error_threshold_mode'] not in {'fixed','train_quantile'}:
        raise ValueError('stage2.high_error_threshold_mode 必須是 fixed/train_quantile。')
    if not 0 < float(stage2['high_error_quantile']) < 1:
        raise ValueError('stage2.high_error_quantile 必須位於 (0,1)。')
    if stage2['high_error_threshold_mode'] == 'fixed' and stage2['high_error_threshold'] is None:
        raise ValueError('fixed 模式必須設定 stage2.high_error_threshold。')
    if pca['pca_dim'] is not None and pca['pca_ev'] is not None:
        raise ValueError('pca.pca_dim 與 pca.pca_ev 只能選一種，另一個必須設 null。')
    if pca['pca_dim'] is None and pca['pca_ev'] is None:
        raise ValueError('pca.pca_dim 與 pca.pca_ev 必須擇一設定。')
    if pca['pca_dim'] is not None and int(pca['pca_dim']) <= 0:
        raise ValueError('pca.pca_dim 必須 > 0。')
    if pca['pca_ev'] is not None and not 0 < float(pca['pca_ev']) <= 1:
        raise ValueError('pca.pca_ev 必須介於 (0,1]。')
    if ext['backend'] == 'company_anomalyvfm' and int(ext['image_res']) != 672:
        raise ValueError('公司 AnomalyVFM H1 的 image_res 必須是 672。')
    if int(ext['image_res']) % 14:
        raise ValueError('image_res 須可被 DINOv2 patch_size=14 整除。')
    if int(ext['batch_size']) <= 0:
        raise ValueError('feature_extractor.batch_size 必須為正整數。')
    if not 0 < float(conf['evaluation']['top_patch_fraction']) <= 1:
        raise ValueError('evaluation.top_patch_fraction 必須介於 (0,1]。')
    return conf


def resolved_path(raw: str | None, base: Path) -> Path | None:
    """相對路徑以「執行時目前工作目錄」為基準，符合 Repo 中 CLI 的使用習慣。"""
    if raw is None:
        return None
    return (Path(raw).expanduser() if Path(raw).expanduser().is_absolute()
            else (base / raw)).resolve()
