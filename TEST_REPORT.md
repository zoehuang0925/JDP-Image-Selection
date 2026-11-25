# JDP Image Selection v1 — Test Report

## 已執行的本地測試

- `python -m compileall image_selection`：通過（語法檢查）。
- `python -m pytest image_selection/tests -q`：**12 passed**。
- 使用 Mock DINO 和 Mock PCA 執行 Stage1 Only、Stage2 Random/Mean/P90/Ratio、Stage1+2：**6 種情境皆通過**；生成圖片副本、CSV、PCA2D、Patch Heatmap 與 Terminal Summary。
- Stage1 數值測試：Cosine Mapping、Global Coverage Gain、Self/Other Gain、逐輪 Greedy 更新與磁碟 Coverage Cache 命中。
- Stage2 數值測試：Mean、P90、High-error Ratio，以及 Random 的 Seed 重現性。
- 公開 DINO Adapter Mock 測試：首次自動下載旗標 `local_files_only=False`，以及 CLS / Register Token 移除後 Patch Shape 正確。
- 真正 CLI（非 Mock DINO）在 `stage2 + random + feature_plot:none` 模式下使用 2 張 Train + 4 張 Candidate 的合成 PNG 驗證完整輸出，**Exit Code 0**（此情境跳過不需要的 DINO）。

## 尚未能在此環境驗證的項目

- 公司 AnomalyVFM / DoRA `.pkl` 與 DINO `.pth` 權重載入（公司檔案及 GPU 環境未提供）。
- Hugging Face 公開 DINOv2 首次真實下載與完整權重推論（本工具執行環境無網路）；Adapter 程式介面有做 Mock Test。
- 多百／千張候選影像的完整 Patch GPU Coverage 效能（需要公司 GPU 實測）。
- 公司實際三篩的 FPR/Recall 表現（需其實際評估資料與門檻設定）。

測試目標為驗證**第一版程式流程與資料輸出完整性**；不代表已在公司成功部署。
