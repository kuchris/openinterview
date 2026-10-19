# OpenInterview

<p align="center"><a href="README.md">English</a> · <b>繁體中文</b> · <a href="README.ja.md">日本語</a></p>

![OpenInterview：App 正在回答面試問題](docs/images/hero.png)

macOS（Apple Silicon）上的本地、低延遲語音辨識，加上 AI 面試練習教練。它在本機轉錄通話雙方的
聲音，找出面試官提出的問題，並根據你的履歷和應徵的職位，即時顯示一段可以直接說出口的簡短答案。

專為**面試練習、模擬面試，以及允許錄音和使用 AI 輔助的會議**而設計。它是一個普通、可見的視窗，
附有錄音指示燈：沒有隱藏的浮動視窗、沒有暗中錄音，也不會規避螢幕分享。

## 功能

- **本機語音辨識：** MLX Whisper、NVIDIA Parakeet 或 Apple SpeechAnalyzer，提供即時
  部分結果，最終結果延遲約 1–2 秒。音訊不會離開你的 Mac。
- **通話雙方：** 你的麥克風和系統音訊（Core Audio process tap，或 BlackHole），並抑制
  喇叭回音。
- **問題偵測：** 自動找出技術和行為面試問題，忽略寒暄。
- **AI 答案：** 透過 [OpenCode](https://opencode.ai)（例如 DeepSeek）或任何相容 OpenAI
  的伺服器。每個答案包括一段可以說出口的簡短回答、重點，以及有需要時的程式碼範例。
- **個人檔案：** 匯入履歷後，由 AI 整理。行為問題的答案只使用真實事實，絕不出現需要填空
  的佔位文字。缺少資料時會顯示 💡 提示，並可按 **Add to profile** 補充。
- **每次面試的目標：** 公司、職位和職位描述會影響答案。面試前檢查會列出你的個人檔案
  暫時答不好的可能問題。
- **多種語言：** 英文、日文、普通話（簡體／繁體）和廣東話。
- **面試記錄：** 每次面試可以儲存在本機（逐字稿、問答、錄音），之後重播或刪除。也可以
  關閉儲存。
- **模擬面試測試：** 由腳本控制的面試官語音，端對端測試整個流程。

## 快速開始

需求：macOS 14.2 或以上的 Apple Silicon Mac，以及 Xcode 或 Command Line Tools。
Apple 語音辨識需要 macOS 26。

```bash
git clone https://github.com/kuchris/openinterview.git
cd openinterview
scripts/setup.sh            # 安裝相依套件、Swift 輔助程式、模型，並檢查環境
uv run openinterview app    # 開啟 App
```

選用：如要 AI 答案，在一個空資料夾執行 OpenCode，然後在 App 中開啟 **Answers**（✦）。

```bash
opencode auth login                         # 只需一次
mkdir -p ~/opencode-scratch && cd ~/opencode-scratch && opencode serve --port 4096
```

第一次擷取系統音訊時，macOS 會要求「系統音訊錄製」權限。詳見
[安裝說明](docs/installation.md)（英文）。

## 常用指令

```bash
uv run openinterview app                       # App 視窗
uv run openinterview run --source system       # 在終端機顯示即時逐字稿
uv run openinterview -l yue run                # 廣東話（也支援 ja、zh、zh-TW）
uv run openinterview profile import ~/cv.pdf   # 從履歷建立個人檔案
uv run openinterview mock --app                # 模擬面試測試
uv run openinterview check --probe             # 環境和權限報告
```

## 文件（英文）

| | |
|---|---|
| [Installation](docs/installation.md) | 系統需求、安裝、macOS 權限 |
| [Usage](docs/usage.md) | 指令、App 視窗、面試記錄、網頁介面 |
| [Practice assistant](docs/assistant.md) | AI 答案、個人檔案和履歷、OpenCode 設定 |
| [Languages](docs/languages.md) | 日文、普通話、廣東話：模型和問題偵測 |
| [Architecture](docs/architecture.md) | 處理流程、系統音訊後端、回音處理 |
| [Benchmarks](docs/benchmarks.md) | 各語音辨識後端的延遲、準確度和記憶體 |
| [Testing](docs/testing.md) | 模擬面試測試和單元測試 |
| [PLAN.md](PLAN.md) | 原始設計和分階段計劃 |

## 私隱

- **音訊**在 Mac 上轉錄，絕不上載。
- **已儲存的面試**存放在 `~/.openinterview/sessions/`。錄音是獨立的開關，兩者都可以關閉。
- **個人檔案**（`~/.openinterview/profile.md`）存放在 repository 以外。
- **會傳送給 AI 供應商的內容：** 偵測到的問題、最近的逐字稿、你的個人檔案和面試目標。
  AI 助手關閉時不會傳送任何內容。
- **同意：** 錄下通話另一方的聲音可能需要對方同意。請只在允許的情況下使用。
