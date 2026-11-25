# 建立自己的 `zoehuang0925/JDP-Image-Selection` Repository

這是**全新獨立 Repo**；**不要**將程式碼推送到 `yuting8902/SubspaceAD`。

## 1. 建立 GitHub Repository

前往 https://github.com/new ，Repository name 輸入 `JDP-Image-Selection`，Owner 選擇 `zoehuang0925`，建議因產學計畫內容選擇 **Private**。建立後，不要預先建立 README、.gitignore 或 LICENSE（本 ZIP 已包含 README、.gitignore）。

## 2. 上傳本專案

將 ZIP 內容解壓到 `JDP-Image-Selection/`，在該資料夾執行：

```bash
git init
git branch -M main
git add README.md .gitignore GITHUB_SETUP.md TEST_REPORT.md requirements.txt requirements-public.txt image_selection/
git commit -m "feat: initial JDP image selection v1"
git remote add origin https://github.com/zoehuang0925/JDP-Image-Selection.git
git push -u origin main
```

已有建立/Clone 的空 Repo 時，請勿重複 `git init` / `git remote add origin`；在 Repo 根目錄直接複製專案檔，`git add`、`git commit`、`git push` 即可。

若你的 GitHub Connector 具備新 Repo `Contents: write` 權限，也可以在 Repo **建立完成後**交由 ChatGPT 將檔案提交上去。目前 Connector 無法建立新的 Repository，必須先透過 GitHub UI 建立。

## 3. 朋友 Repo 保持唯讀依賴

在執行此選樣系統的電腦上，設定 `feature_extractor.company.subspacead_repo_path` 指向**本地檔案系統中**的朋友 SubspaceAD 原始碼目錄；例如兩個資料夾並列時為 `../SubspaceAD`。若公司已安裝該 Repo，改成公司的實際絕對路徑即可。

本地端可以自行 Clone 公開原始碼作為 PCA 依賴，但這只是下載，**不是將檔案提交到朋友 Repo**。
