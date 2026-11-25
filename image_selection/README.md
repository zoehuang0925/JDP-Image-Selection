# JDP 正常影像選樣工具（SubspaceAD / AnomalyVFM）

這是**獨立存放在你自己的 `JDP-Image-Selection` Repository 中的 Python 套件**，不修改或重跑朋友的 SubspaceAD「三篩」`main.py`。公司與本地實測須透過 `feature_extractor.company.subspacead_repo_path` 指向**本機可讀取的**原 SubspaceAD 原始碼（唯讀使用）。第一版依 Stage 1／Stage 2 技術交接文件實作，並依後續討論採用 **方案 A（Stage 1 Shortlist → Stage 2 Batch Ranking）**。**不包含已擱置的 PCA Improvement Gate。**

> **重要：選樣不是自動驗證標籤。** 公司二篩／三篩判正常的圖片，可能仍含真實異常。此工具預設候選圖片已經過適當確認；若候選標籤未知，請先人工確認，再將選出的圖片用於正式訓練。

## 1. 兩個獨立 GitHub Repo 如何安裝與使用？

**你自己的 Repository：`zoehuang0925/JDP-Image-Selection`。朋友的 `yuting8902/SubspaceAD` 保持唯讀，不對朋友 Repo 推送、修改或建立 PR。**

在本地或公司機器上，確保兩份 Repo 的程式碼皆能讀取，例如：

```text
workspace/
├── JDP-Image-Selection/          # 你自己的 Repo（本工具完整原始碼）
│   ├── README.md
│   ├── requirements.txt
│   ├── image_selection/
│   │   ├── config.yaml
│   │   ├── __main__.py
│   │   ├── extractors.py
│   │   ├── stage2.py
│   │   └── ...
│   └── TEST_REPORT.md
└── SubspaceAD/                   # 朋友的 Repo，唯讀依賴（不改）
    ├── main.py
    └── src/subspacead/core/
        ├── anomalyvfm_extractor.py
        └── pca.py
```

若兩份 Repo 如上並列，`feature_extractor.company.subspacead_repo_path: ../SubspaceAD./SubspaceAD` 可直接使用；公司位置不同時，**必須改成公司實際 SubspaceAD Repo 的絕對路徑**。本地公開 DINOv2 也會使用該 Repo 的 `pca.py`（除了純 Random 模式）；公開模型的權重則可以自動下載，不需要 AnomalyVFM 權重。

安裝依賴請在**JDP-Image-Selection 根目錄**執行：

```bash
python -m pip install -r requirements.txt
# PyTorch 與 CUDA 依個人/公司環境安裝；公司優先沿用既有三篩環境。
# 選用本地公開 DINOv2：
python -m pip install -r requirements-public.txt
# 選用 UMAP：
python -m pip install umap-learn
python -m image_selection --config image_selection/config.yaml
```

若未下載朋友 Repo 的原始碼，但公司已有現成 SubspaceAD 資料夾，只要 Config 指向該位置即可，不需要再 Clone 一份。若本地端沒有原 Repo，可僅**唯讀 Clone**：`git clone https://github.com/yuting8902/SubspaceAD.git ../SubspaceAD`。這是取得公開原始碼，不會把任何東西寫回朋友 GitHub。

**公司環境部署前檢查**：Torch/CUDA 是否能沿用現有三篩環境；原 `pca.py`、`anomalyvfm_extractor.py` 是否存在；`AnomalyVFM_` 是否有 `models/dinov2_offline.py`、`peft_local/`；DINOv2 官方庫有 `hubconf.py`；權重路徑可讀。已實作模擬測試，不代表公司 GPU/權重實測通過。

## 2. 公司 / 本地 DINO 如何切換（只修改 Config）

### A. 公司正式使用（優先）

在你自己 Repo 的 `image_selection/config.yaml`：

```yaml
feature_extractor:
  backend: company_anomalyvfm
  image_res: 672
  use_clahe: false
  company:
    variant: a1_adapted_final
    subspacead_repo_path: ../SubspaceAD
    anomalyvfm_root: /workspace/Venessa/AnomalyVFM_
    anomalyvfm_ckpt: /workspace/Venessa/AnomalyVFM_/pretrained_models/anomalyvfm_dinov2.pkl
    dino_repo_path: /workspace/Venessa/dinov2
    dino_weight_path: /workspace/model_weight/dinov2_vitl14_reg4_pretrain.pth
```

**進公司只需改對路徑嗎？** 路徑是重要前提，但還須確認環境依賴、GPU 記憶體、三篩實際使用的是 A1/H1、CLAHE／augmentation／mask 等前處理。`company` 參數直接轉交給原 `AnomalyVFMFeatureExtractor`，不會重新實作 DoRA。模型需已經在本機／掛載檔案系統中；這不是遠端 API 呼叫。

### B. 本地公開 DINOv2（第一次自動下載）

只需先在 `config.yaml` 修改：

```yaml
feature_extractor:
  backend: public_dinov2       # 改這行
  image_res: 672
  public:
    model_id: facebook/dinov2-with-registers-large
    auto_download: true        # 第一次從 Hugging Face 下載並快取
    local_model_dir: null      # null 代表根據 model_id 下載 / 快取
    hf_cache_dir: null         # null 代表 Hugging Face 預設快取
```

保持其他區塊（`selection`、`dataset`、`stage2`、`pca`）不變。第一次需要**可連線到 Hugging Face**，並先安裝 `image_selection/requirements-public.txt`；之後在已有快取下可使用既有下載權重。公司版不必額外安裝公開版依賴，避免干擾已驗證的三篩環境。模型 `facebook/dinov2-with-registers-large` 可能需要不少 RAM／VRAM；如需比較小模型可換成公開小型 DINO，但**Feature Dimension 會變**，而且不能與公司 Features 直接比較。

公開版拿最後一層 Patch Tokens，移除 CLS 與 Register Tokens；公司 A1 使用 AnomalyVFM DoRA-adapted `x_norm_patchtokens`。**兩種模型數值不相同，實驗排名不能直接當成等價結果。** 朋友 Repo 的一般 `FeatureExtractor` 強制離線，本工具的公開 Adapter 則另外支援首次網路下載。

## 3. 資料夾如何準備？

公司與本地共用相同結構，每個 Part 有各自的 PCA：

```text
datasets/mvtec-ad/
├── bottle/
│   ├── train/good/                # 既有可信正常訓練圖片，不會修改
│   │   ├── 001.png
│   │   └── 002.png
│   ├── candidates/good/           # 另一個資料夾，預計從這裡選新圖
│   │   ├── C001.png
│   │   └── C002.png
│   ├── val/good/                  # 可選：評估 threshold 校準（不參與選樣）
│   └── test/good/, test/defect/   # 可選：最終測試（不參與選樣）
└── cable/
    ├── train/good/
    └── candidates/good/
```

接受 `.png`、`.jpg`、`.jpeg`、`.bmp`（副檔名不區分大小寫）；壞圖跳過並記錄。**原始 Training Set 必須至少有一張有效正常圖片**，不實作 Bootstrap；任何模式均不修改既有圖片。

如果已經下載 MVTec-AD，可以用下列指令**建立獨立的新測試資料根目錄**，不修改原始 MVTec 檔案：

```bash
python image_selection/tools/prepare_mvtec_demo.py \
  --source ./datasets/mvtec-ad \
  --destination ./datasets/mvtec-selection-demo \
  --part bottle --initial 3 --candidates 10 --seed 42
```

然後修改 `dataset.root: ./datasets/mvtec-selection-demo`。上述只是本地 Smoke Test 的可信正常樣本分割，不會使用 MVTec Test Set 來做選樣。

## 4. 三種模式與四種 Stage 2 策略

只需要修改 Config：

| 想做什麼 | `selection.mode` | `selection.k` | `stage2.strategy` | 運作方式 |
|---|---|---|---|---|
| Stage 1 Only | `stage1` | 5 | 任意（不使用） | 全候選 Greedy 選最多 5 張 |
| Stage 2 Only：Random | `stage2` | 5 | `random` | 固定 seed 隨機選最多 5 張 |
| Stage 2 Only：Mean | `stage2` | 5 | `mean` | 初始 PCA → 平均 Patch Error 排名 → Top 5 |
| Stage 2 Only：P90 | `stage2` | 5 | `p90` | 初始 PCA → P90 Error 排名 → Top 5 |
| Stage 2 Only：Ratio | `stage2` | 5 | `high_error_ratio` | 初始 PCA → 高誤差 Patch 比例排名 → Top 5 |
| Stage 1 + 2 | `stage1_stage2` | 5 | 以上四種皆可 | Stage 1 先選 **2K = 10** 張，Stage 2 從 10 張中選最多 5 張 |

**K 是新增上限而非原始 Training Set 總數；允許少於 K，但不會超出 K。** Stage 1 如果 Coverage Gain `<= epsilon`、沒有可用候選、或達到預選預算，就停止。整合模式 Stage 1 的 2K 只是 **暫時 shortlist**，不會把未被 Stage 2 選中的圖片加入正式 Training Set。

**Stage 1**：保留 2,304×1,024 Patch Set，逐 Patch L2 normalize，核 `(1+cos)/2`，在整個候選池 U 上計算兩層平均的 Coverage Gain，依最高增益逐張 Greedy。為控制精確比較所需的記憶體，使用 `query_chunk` 與 `reference_chunk` 分塊矩陣乘法，並以磁碟 Memory Map 保存完整候選 Coverage Cache。未在選樣前做 Feature PCA／random projection／patch sampling。因為運算量很大，大型候選池**可能需要很長時間與大量快取磁碟**；`max_coverage_cache_gb` 防止估計快取超出預算。

**Stage 2**：公司/本地都使用原三篩 `src.subspacead.core.pca.PCAModel` 建立 PCA（僅換 DINO Features 來源），Patch Error 採 `sum((x - recon)^2)`。`random` 不需要計算 PCA；其他三種是 Batch 排名、**選樣途中不反覆 Fit PCA**。`high_error_ratio` 的預設 τ 是原始 Train Patch Error Q95，此值是 **in-sample 的探索性備案**，不是獨立校準得到的客觀標準。

### 常改參數對照

| 需求 | 要改的 Config |
|---|---|
| 改成公司正式 DINO | `feature_extractor.backend: company_anomalyvfm`、`feature_extractor.company` 下 5 個路徑／版本 |
| 改成本地自動下載 DINO | `feature_extractor.backend: public_dinov2`、`feature_extractor.public.auto_download: true` |
| 只測 bottle | `dataset.parts: ['bottle']` |
| 全產品依序執行 | `dataset.parts: all` |
| 候選資料夾不同名稱 | `dataset.candidate_rel`，例如 `candidate_images` |
| 修改最多新增張數 | `selection.k` |
| 模式切換 | `selection.mode` |
| Stage2 策略切換 | `stage2.strategy` |
| Stage1 不再有改善就停止 | `stage1.epsilon`，預設 0.0 |
| Stage2 Ratio 高誤差門檻 | `stage2.high_error_threshold_mode`、`high_error_quantile` 或 `high_error_threshold` |
| SubspaceAD 的 PCA 99% EV | `pca.pca_ev: 0.99`、`pca.pca_dim: null` |
| 固定 PCA 維度 | `pca.pca_ev: null`、`pca.pca_dim: 512`（依資料有效秩調整） |
| 不畫 2D 圖 | `visualization.feature_plot: none` |
| 只畫 UMAP | `visualization.feature_plot: umap`，另裝 `umap-learn` |
| 兩種 2D 都要 | `visualization.feature_plot: both` |
| 不畫 Error Heatmap | `visualization.save_heatmap: false` |
| 不產生圖片副本 | `output.copy_selected: false`、`output.copy_merged: false` |
| 只做選樣、不跑測試 | `evaluation.enabled: false`（預設） |

## 5. Feature Cache 與計算成本

- 原始 Patch Features 儲存於 `feature_extractor.cache_dir/<model_signature>/<image_hash>.npy`；Stage 1 再衍生 L2 normalized view、Stage 2 保留 Raw View。
- 模型來源、checkpoint 路徑/檔案時間與大小、影像內容及前處理會納入 Cache Key。不同 DINO 不會誤用相同 Feature Cache。
- Stage 1 精確 Greedy 一開始需要計算每張候選圖對全部候選 Patch 的最佳相似度，這步**可能很慢，尤其資料池包含數百或數千張時**。建議先用少量圖片 Smoke Test，並預留磁碟空間；若超過 `max_coverage_cache_gb`，程式會直接顯示錯誤，不會默默採用 Patch Sampling。
- 只關閉 PCA2D/UMAP 畫圖不會關閉選樣時必要的 DINO 與 Patch 運算。

## 6. 輸出有哪些？

每次執行建立不同時間戳 Run 目錄：

```text
image_selection/.results/20261126_010203_abcdef/
├── run_config.json                 # 本次實際 YAML 設定
├── run_summary.json                # 每產品數量、狀態、停止原因、成功/失敗圖片數
├── feature_manifest.csv            # 原圖/候選圖對應 Feature Cache 與模型 signature
├── run.log                         # 執行過程
└── bottle/
    ├── candidate_scores.csv        # 所有候選圖片與 Stage1/Stage2 分數、選中狀態
    ├── stage1_ranking.csv          # Stage1 所有輪次（同輪全部候選）排名
    ├── stage2_ranking.csv          # Stage2 該次 Batch Ranking
    ├── selection_decisions.csv     # 最終選中圖片的順序/路徑
    ├── invalid_images.csv          # 壞圖與特徵萃取失敗者
    ├── copied_files.csv            # 來源圖片與複製後路徑的對應
    ├── metrics.json                # 該 Part 的執行統計與選用評估
    ├── Selected_Only/              # 只有最後正式選中的 K 張（或少於 K）
    ├── Merged_Training_Set/        # 原始 Training Set + 正式選中者
    └── plots/
        ├── feature_distribution_pca.png
        ├── feature_distribution_umap.png
        ├── stage1_first_round_ranking.png
        ├── stage2_score_ranking.png
        ├── selection_iteration.png
        └── *_pca_error.png
```

依照 Config、模式與是否有選樣結果，並非每個檔案都一定產生。`Selected_Only`／`Merged_Training_Set` 副本不修改原始檔案。產品間結果隔離。若複製的原始檔案檔名衝突，複製副本將自動加後綴，並在 `copied_files.csv` 追蹤。

`candidate_scores.csv` 會記錄 Stage1 是否進 Shortlist、輪次、Coverage Gain、Self/Other Gain、Stage2 Score/Rank、最後是否 Selected。請注意 Stage1 `stage1_ranking.csv` 是**每一輪重新計算的排名**，Stage2 `stage2_ranking.csv` 是**單次 Batch 排名**。

## 7. 評估（非選樣必需）

預設 `evaluation.enabled: false`：公司只提供正常候選時也能正常選樣、輸出圖片與 CSV。

若設定 `evaluation.enabled: true`，在每個 Part 的 `test/good`、`test/<defect>` 上分別建立 Initial/Final PCA 的異常分數，能在兩類標註齊全時算 Image AUROC。FPR/Recall **僅在有固定 `evaluation.threshold` 或獨立 `val/good` 估計門檻時**才計算；絕不從 Test Set 推估分類門檻。以設定 `top_patch_fraction` 的高誤差 Patch 均值作為影像分數。這是基礎測試介面，**不等於三篩整體決策流程的 FPR／Recall**，因為沒有整合三篩額外的去噪、門檻與先前篩選步驟。

## 8. 常見錯誤／注意事項

1. **`AnomalyVFM_ root not found`**：先確認 `feature_extractor.company.anomalyvfm_root`；還需要 `models/dinov2_offline.py` 與 `peft_local/`。
2. **權重載入失敗**：確認 `dino_weight_path` `.pth` 與 `anomalyvfm_ckpt` `.pkl`；使用相容的公司三篩 Python／PyTorch 版本。
3. **公開 DINO 網路下載失敗**：檢查本地端網路、模型 ID、Hugging Face 連線、快取空間。公開版第一次會需要下載權重；之後可使用快取。
4. **Stage2 `pca_dim` 錯誤**：`pca_dim`／`pca_ev` 必須擇一。單張原始正常影像仍有多個 Patch Features，可建立 PCA，但實際維度受有效秩限制。
5. **找不到 Candidate**：建立 `<dataset.root>/<part>/<dataset.candidate_rel>`，不要直接把原始訓練圖和候選圖混在 `train/good`。
6. **Stage1 很慢／Coverage 快取很大**：這是完整 Patch Exact Coverage 的計算成本。先縮小候選池或選更強的 GPU；不要在第一版偷偷改成 Patch Sampling／PCA 降維。
7. **選樣不足 K**：請先查看 `run_summary.json` 的 `stop_reason`（`gain_threshold`、`no_candidates` 或 `max_budget`）。
8. **分布圖 PCA2D 跟選樣 PCA 不一樣**：分布圖把一張圖片的 Patch Features 取平均作可視化，選樣 Stage1 不會做這個平均；PCA2D 也不是 Stage2 的正常模型。

## 9. 不帶權重的單元測試

```bash
pip install pytest
python -m pytest image_selection/tests -q
```

測試透過合成 Patch Features 檢查 Gain、Stage1 逐次更新、四種 Stage2 分數、Config 與完整輸出流程；這不等於已在公司 GPU 成功載入 AnomalyVFM。

### 小技巧：Random Baseline 不需要模型

若只想驗證資料夾與複製輸出，可設定 `selection.mode: stage2`、`stage2.strategy: random`、`visualization.feature_plot: none`、`evaluation.enabled: false`，程式將跳過不必要的 DINO 萃取；使用 Mean／P90／Ratio、Stage1，或開啟特徵分布圖時，仍需要模型 Features。
