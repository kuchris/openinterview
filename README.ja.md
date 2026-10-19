# OpenInterview

<p align="center"><a href="README.md">English</a> · <a href="README.zh-Hant.md">繁體中文</a> · <a href="README.zh-Hans.md">简体中文</a> · <b>日本語</b></p>

![OpenInterview：面接の質問に回答するアプリ](docs/images/hero.png)

macOS（Apple Silicon）向けの、ローカルで低遅延な音声認識と AI 面接練習コーチです。通話の
双方の音声をデバイス上で文字起こしし、面接官の質問を見つけて、あなたの履歴書と応募先の
ポジションに合わせた、そのまま話せる短い回答を表示します。

**面接練習、模擬面接、そして録音と AI の利用が許可されている会議**のためのツールです。
録音インジケーター付きの通常の見えるウィンドウで、隠しオーバーレイ、隠し録音、
画面共有の回避は一切ありません。

## 機能

- **デバイス上の音声認識：** MLX Whisper、NVIDIA Parakeet、Apple SpeechAnalyzer。
  途中結果をリアルタイムに表示し、確定までの遅延は約 1〜2 秒。音声は Mac の外に出ません。
- **通話の双方：** マイクとシステム音声（Core Audio プロセスタップ、または BlackHole）。
  スピーカーのエコーを抑制します。
- **質問検出：** 技術的な質問と行動面接の質問を拾い、雑談は無視します。
- **AI 回答：** [OpenCode](https://opencode.ai)（DeepSeek など）または OpenAI 互換サーバー
  経由。回答は、そのまま話せる短い答え、要点、必要に応じたコード例です。
- **プロフィール：** 履歴書をインポートすると AI が整理します。行動面接の回答は事実だけを
  使い、穴埋めのプレースホルダーは出しません。情報が足りない時は 💡 ヒントが表示され、
  **Add to profile** で追加できます。
- **面接ごとのターゲット：** 会社、職種、求人内容が回答に反映されます。面接前チェックで、
  今のプロフィールではうまく答えられない想定質問を一覧にします。
- **多言語：** 英語、日本語、中国語（普通話：簡体字／繁体字）、広東語。
- **セッション履歴：** 各セッション（文字起こし、Q&A、音声）をローカルに保存し、再生や
  削除ができます。保存はオフにもできます。
- **模擬面接テスト：** スクリプト化された面接官の音声で、全体をエンドツーエンドで確認します。

## クイックスタート

必要なもの：macOS 14.2 以降の Apple Silicon Mac と、Xcode または Command Line Tools。
Apple の音声認識には macOS 26 が必要です。

```bash
git clone https://github.com/kuchris/openinterview.git
cd openinterview
scripts/setup.sh            # 依存関係、Swift ヘルパー、モデル、環境チェック
uv run openinterview app    # アプリを開く
```

任意：AI 回答を使うには、空のフォルダで OpenCode を起動し、アプリで **Answers**（✦）を
オンにします。

```bash
opencode auth login                         # 初回のみ
mkdir -p ~/opencode-scratch && cd ~/opencode-scratch && opencode serve --port 4096
```

初めてシステム音声を取り込む時、macOS が「システムオーディオ録音」の許可を求めます。
詳しくは[インストール](docs/installation.md)（英語）を参照してください。

## よく使うコマンド

```bash
uv run openinterview app                       # アプリのウィンドウ
uv run openinterview run --source system       # ターミナルにリアルタイム文字起こし
uv run openinterview -l ja run                 # 日本語（yue、zh、zh-TW も対応）
uv run openinterview profile import ~/cv.pdf   # 履歴書からプロフィールを作成
uv run openinterview mock --app                # 模擬面接テスト
uv run openinterview check --probe             # 環境と権限のレポート
```

## ドキュメント（英語）

| | |
|---|---|
| [Installation](docs/installation.md) | 必要環境、セットアップ、macOS の権限 |
| [Usage](docs/usage.md) | コマンド、アプリのウィンドウ、セッション履歴、Web ダッシュボード |
| [Practice assistant](docs/assistant.md) | AI 回答、プロフィールと履歴書、OpenCode の設定 |
| [Languages](docs/languages.md) | 日本語、中国語、広東語：モデルと質問検出 |
| [Architecture](docs/architecture.md) | 処理の流れ、システム音声のバックエンド、エコー対策 |
| [Benchmarks](docs/benchmarks.md) | 音声認識バックエンドごとの遅延、精度、メモリ |
| [Testing](docs/testing.md) | 模擬面接テストとユニットテスト |
| [PLAN.md](PLAN.md) | 元の設計とフェーズ計画 |

## プライバシー

- **音声**は Mac 上で文字起こしされ、アップロードされることはありません。
- **保存したセッション**は `~/.openinterview/sessions/` にあります。音声の録音は別の
  スイッチで、どちらもオフにできます。
- **プロフィール**（`~/.openinterview/profile.md`）はリポジトリの外に保存されます。
- **AI プロバイダーに送られるもの：** 検出された質問、直近の文字起こし、プロフィール、
  面接のターゲット。アシスタントがオフの間は何も送られません。
- **同意：** 通話相手の音声を録音するには、相手の同意が必要な場合があります。許可されて
  いる場合にのみ使用してください。
