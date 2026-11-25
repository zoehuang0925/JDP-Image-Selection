# JDP Image Selection — Stage 1 / Stage 2 / Stage 1+2 (v1)

**獨立的正常影像選樣專案**。本 Repository 放置新的選樣程式，不複製、修改或提交朋友的 [`yuting8902/SubspaceAD`](https://github.com/yuting8902/SubspaceAD) 原始碼。它在執行時會**唯讀引用**原 Repo 的 `AnomalyVFMFeatureExtractor`（公司模式）及 `PCAModel`（Stage 2／評估模式）。

## 快速開始

1. 取得本 Repo 與原 SubspaceAD Repo 的原始碼；預設假設兩個 Repo 放在同一上層目錄（`../SubspaceAD`）。如果公司原本已有 SubspaceAD，可直接指定該路徑，**不用 Clone，也不要改朋友 Repo**。
2. 在自己的 Repo 根目錄安裝：`python -m pip install -r requirements.txt`，且確保現有 PyTorch 運算環境相容。公開 DINOv2 模式再安裝 `python -m pip install -r requirements-public.txt`。
3. 修改 `image_selection/config.yaml`：
   - `dataset.root`、`dataset.parts`、`dataset.candidate_rel`、`selection.mode`、`selection.k`、`stage2.strategy`。
   - **公司 AnomalyVFM：** `feature_extractor.backend: company_anomalyvfm`，確認 `feature_extractor.company.subspacead_repo_path`（原 Repo 的原始碼目錄）、`anomalyvfm_root`、`anomalyvfm_ckpt`、`dino_repo_path`、`dino_weight_path`，並與三篩使用的 A1/CLAHE 設定一致。
   - **本地公開 DINO：** `feature_extractor.backend: public_dinov2` 且 `feature_extractor.public.auto_download: true`；第一次推論時自動從 Hugging Face 下載公開權重。**但若執行 Stage 2 非 Random 策略，仍須能讀取原 Repo 的 `pca.py`。**
4. 執行：

```bash
python -m image_selection --config image_selection/config.yaml
```

## 模式與選樣規則

| 模式 | 運作方式 | 最終最多新增 |
| --- | --- | --- |
| `stage1` | Greedy Patch Coverage | K |
| `stage2` | `random` / `mean` / `p90` / `high_error_ratio` | K |
| `stage1_stage2` | Stage 1 自動預選最多 2K → Stage 2 排名取最多 K | K |

`K` 只代表**新加入**的正常影像數量，不包含原本 `train/good/`。兩版 DINO 提取出的 Feature 不等價，必須分開比較；候選樣本也需人工驗證確為可信正常資料。

## 資料格式

```text
datasets/mvtec-ad/
└── bottle/
    ├── train/good/         # 原始訓練正常圖，不修改
    └── candidates/good/    # 待選候選圖，與原始訓練集分開
```

## 測試

```bash
python -m pip install pytest
python -m pytest image_selection/tests -q
```

目前已執行模擬/Mock 測試；**公司 AnomalyVFM 真實權重與 Hugging Face 首次實際下載仍需在對應環境驗證**。

## 文件

- [完整使用說明（每個 Config 與模式）](image_selection/README.md)
- [GitHub 建立與安裝說明](GITHUB_SETUP.md)
- [測試報告](TEST_REPORT.md)

> 公司資料或實際產線影像請勿隨程式提交 GitHub。若涉及產學合作機密，Repository 請設為 **Private**，並遵循公司資料權限規範。
