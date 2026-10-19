# OpenInterview

<p align="center"><a href="README.md">English</a> · <a href="README.zh-Hant.md">繁體中文</a> · <b>简体中文</b> · <a href="README.ja.md">日本語</a></p>

![OpenInterview：应用正在回答面试问题](docs/images/hero.png)

macOS（Apple Silicon）上的本地、低延迟语音识别，加上 AI 面试练习教练。它在本机转录通话双方的
声音，找出面试官提出的问题，并根据你的简历和应聘的职位，实时显示一段可以直接说出口的简短答案。

专为**面试练习、模拟面试，以及允许录音和使用 AI 辅助的会议**而设计。它是一个普通、可见的窗口，
带有录音指示灯：没有隐藏的悬浮窗口、没有偷偷录音，也不会规避屏幕共享。

## 功能

- **本地语音识别：** MLX Whisper、NVIDIA Parakeet 或 Apple SpeechAnalyzer，提供实时
  中间结果，最终结果延迟约 1–2 秒。音频不会离开你的 Mac。
- **通话双方：** 你的麦克风和系统音频（Core Audio process tap，或 BlackHole），并抑制
  扬声器回声。
- **问题检测：** 自动找出技术和行为面试问题，忽略寒暄。
- **AI 答案：** 通过 [OpenCode](https://opencode.ai)（例如 DeepSeek）或任何兼容 OpenAI
  的服务器。每个答案包括一段可以说出口的简短回答、要点，以及需要时的代码示例。
- **个人资料：** 导入简历后，由 AI 整理。行为问题的答案只使用真实事实，绝不出现需要填空
  的占位文字。缺少信息时会显示 💡 提示，并可点击 **Add to profile** 补充。
- **每场面试的目标：** 公司、职位和职位描述会影响答案。面试前检查会列出你的个人资料
  暂时答不好的可能问题。
- **多种语言：** 英语、日语、普通话（简体／繁体）和粤语。
- **面试记录：** 每场面试可以保存在本地（文字记录、问答、录音），之后回放或删除。也可以
  关闭保存。
- **模拟面试测试：** 由脚本控制的面试官语音，端到端测试整个流程。

## 快速开始

要求：macOS 14.2 或以上的 Apple Silicon Mac，以及 Xcode 或 Command Line Tools。
Apple 语音识别需要 macOS 26。

```bash
git clone https://github.com/kuchris/openinterview.git
cd openinterview
scripts/setup.sh            # 安装依赖、Swift 辅助程序、模型，并检查环境
uv run openinterview app    # 打开应用
```

可选：如需 AI 答案，在一个空文件夹中运行 OpenCode，然后在应用中打开 **Answers**（✦）。

```bash
opencode auth login                         # 只需一次
mkdir -p ~/opencode-scratch && cd ~/opencode-scratch && opencode serve --port 4096
```

第一次捕获系统音频时，macOS 会请求“系统音频录制”权限。详见
[安装说明](docs/installation.md)（英文）。

## 常用命令

```bash
uv run openinterview app                       # 应用窗口
uv run openinterview run --source system       # 在终端显示实时文字记录
uv run openinterview -l zh run                 # 普通话（也支持 zh-TW、yue、ja）
uv run openinterview profile import ~/cv.pdf   # 从简历建立个人资料
uv run openinterview mock --app                # 模拟面试测试
uv run openinterview check --probe             # 环境和权限报告
```

## 文档（英文）

| | |
|---|---|
| [Installation](docs/installation.md) | 系统要求、安装、macOS 权限 |
| [Usage](docs/usage.md) | 命令、应用窗口、面试记录、网页界面 |
| [Practice assistant](docs/assistant.md) | AI 答案、个人资料和简历、OpenCode 配置 |
| [Languages](docs/languages.md) | 日语、普通话、粤语：模型和问题检测 |
| [Architecture](docs/architecture.md) | 处理流程、系统音频后端、回声处理 |
| [Benchmarks](docs/benchmarks.md) | 各语音识别后端的延迟、准确率和内存 |
| [Testing](docs/testing.md) | 模拟面试测试和单元测试 |
| [PLAN.md](PLAN.md) | 原始设计和分阶段计划 |

## 隐私

- **音频**在 Mac 上转录，绝不上传。
- **已保存的面试**存放在 `~/.openinterview/sessions/`。录音是单独的开关，两者都可以关闭。
- **个人资料**（`~/.openinterview/profile.md`）存放在代码仓库以外。
- **会发送给 AI 服务商的内容：** 检测到的问题、最近的文字记录、你的个人资料和面试目标。
  AI 助手关闭时不会发送任何内容。
- **同意：** 录下通话另一方的声音可能需要对方同意。请只在允许的情况下使用。
