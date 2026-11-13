# TTS Studio

一个在自己电脑上跑的中文配音工作台：把整篇口播稿拆成句子，一句一句生成、重做、换音色，最后合成一条完整音频。

多个配音模型共用一套界面，本地模型和在线 API 可以在同一篇稿子里混着用。

- **本地模型**：Qwen3-TTS（克隆 / 预设音色 / 文字描述造音色）、Breeze TTS 2
- **配音 API**：MiniMax 语音、豆包 / 火山引擎、任意 OpenAI 兼容的 `/audio/speech` 接口

仓库里**不带任何模型**。装好后打开网页，在「设置」里点一下想用的模型，会从[魔搭](https://modelscope.cn)下载，国内直连，断了可以接着下。

## 功能

- **文本朗读**：粘贴整篇稿子 → 自动拆句 → 一键生成全部。每句可以单独改字、重新生成（不改字就换一种读法）、插句、删除。
- **逐句换音色**：整篇选一个音色，某一句想换别的声音，就在那一句后面单独选，可以是另一个模型的音色。
- **语气**：平稳 / 激昂 / 亲切聊天…… 可以自己写提示词并存成语气。支持语气的模型：Breeze、Qwen3 预设音色和描述造音色、OpenAI 兼容接口（gpt-4o-mini-tts 等）。
- **合并导出**：调整语速和句间停顿，一键合成完整 wav。不同模型的采样率会自动统一。
- **音色库**：克隆音色（上传 5～15 秒参考音频）、预设音色、文字描述造音色，都存成带名字的音色，随时选。
- **音色设计**：先用一句话把声音调满意，再存进音色库。
- **开放接口**：OpenAI 兼容的 `/v1/audio/speech`，别的程序、脚本、Agent 可以直接调用这里的模型和音色库。

## 需要什么

| 用法 | 要求 |
|---|---|
| 只用 API | 任意电脑，不需要显卡 |
| Qwen3-TTS 0.6B | NVIDIA 显卡，约 4 GB 显存 |
| Qwen3-TTS 1.7B | NVIDIA 显卡，约 6 GB 显存 |
| Breeze TTS 2 | NVIDIA 显卡，约 8 GB 显存（8 GB 的卡要关掉其他占显卡的软件） |

本地模型同一时间只放一个在显卡上，用到另一个时会自动切换（第一次切换要加载几十秒）。

## 安装

**Windows 安装包（推荐，不用自己装 Python）**：下载 `TTS-Studio-Setup-x.y.z.exe`（约 35 MB），双击安装，不需要管理员权限，默认装在 `%LOCALAPPDATA%\Programs\TTS Studio`。第一次打开会弹一个进度窗口，自动从国内镜像联网安装 PyTorch 等运行环境（有 N 卡装显卡版约 3 GB，没有装 CPU 版），十几分钟，只需要一次。卸载：「设置 → 应用」里找 TTS Studio，可以选保留音色和模型。

自己打安装包：装好 uv 和 `uv python install 3.11`，然后运行 `python packaging/build.py`，输出在 `dist/`。静默安装：`TTS-Studio-Setup-x.y.z.exe /S [/D=目录] [/NOSHORTCUTS] [/NOREG]`。

**从源码运行**：需要 Python 3.10～3.12（推荐 3.11）。

**Windows**：双击 `install.bat`（国内网络可以在命令行运行 `install.bat cn` 用清华镜像），装完双击 `launcher\start-hidden.vbs`，会以独立软件窗口打开（不走浏览器，关掉窗口服务也一起停）。可以右键它 →「发送到 → 桌面快捷方式」，图标用 `launcher\tts-studio.ico`。想用浏览器打开的话，双击 `start.bat`。

**Linux / Mac**：

```bash
./install.sh        # 国内网络：./install.sh cn
./start.sh
```

启动后浏览器会自动打开 http://127.0.0.1:7861 。想换端口就设置环境变量 `TTS_STUDIO_PORT`。

## 第一次使用

1. 打开「设置」：
   - 用本地模型：在「本地模型」里点「下载」。不知道选哪个的话，先下 **Qwen3-TTS 0.6B 预设音色**（2.5 GB，自带 9 个音色）。
   - 用 API：在「接入第三方配音 API」里填 Key，点「保存并试听」确认能通。
2. 打开「音色库 → 新建音色」：选模型、选音色类型、起个名字，存进音色库。
3. 打开「文本朗读」：粘贴稿子 → 拆句 → 右边选音色 → 生成全部 → 合并。

## 模型一览

| 模型 | 能做什么 | 大小 | 许可证 |
|---|---|---|---|
| Qwen3-TTS 1.7B / 0.6B 克隆版 | 上传参考音频克隆声音 | 4.5 / 2.5 GB | Apache 2.0，可商用 |
| Qwen3-TTS 1.7B / 0.6B 预设音色 | 9 个自带音色（含北京话、四川话），可加语气 | 4.5 / 2.5 GB | Apache 2.0，可商用 |
| Qwen3-TTS 1.7B 描述造音色 | 一句话描述想要的声音 | 4.5 GB | Apache 2.0，可商用 |
| Breeze TTS 2 | 克隆 + 语气提示词 | 7.2 GB | **仅限研究和非商业用途**（生成的音频也算） |

> ⚠️ Breeze TTS 2 的权重和用它生成的音频受 [BreezeBlue 非商业许可](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE) 约束。要商用（包括带广告、带货的视频）请用 Qwen3-TTS 或配音 API。

已经下载过模型的话，不用重下：把模型文件夹放到（或链接到）`data/models/` 下，文件夹名和魔搭上一样（比如 `data/models/Breeze-TTS-2`）就行。

## 接入第三方配音 API

| 服务 | 需要填 | 音色怎么填 |
|---|---|---|
| MiniMax | API Key（国内地址 `https://api.minimaxi.com`） | 下拉选自带音色，或粘贴后台克隆好的 `voice_id` |
| 豆包 / 火山引擎 | AppID、Access Token、Cluster | 控制台里的 `voice_type`；声音复刻的音色，Cluster 填 `volcano_icl` |
| OpenAI 兼容 | 接口地址（到 `/v1`）、Key、模型名 | 接口支持的 `voice` 名字 |

Key 只保存在本机的 `data/config/settings.json` 里，不会上传到任何地方。用 API 产生的费用由你自己的账号承担。

## 给其他程序 / Agent 调用

软件开着的时候，同一个端口上还有一套和 OpenAI 语音接口格式一样的 HTTP 接口。地址和 Key 在「开放接口」页里，可以一键复制。

| 接口 | 作用 |
|---|---|
| `GET /v1/voices` | 音色库里的音色，以及各模型自带的音色 ID |
| `GET /v1/models` | 各模型能不能用（没下载、没填 Key 会写原因） |
| `POST /v1/audio/speech` | 读一段文字，返回音频 |
| `/v1/docs` | 在浏览器里看的接口文档 |

`voice` 填音色库里的名字，也可以直接写「模型:音色ID」，不用先建音色，比如 `qwen3:vivian`、`minimax:female-shaonv`。

```bash
curl http://127.0.0.1:7861/v1/audio/speech \
  -H "Authorization: Bearer 你的接口Key" -H "Content-Type: application/json" \
  -d '{"input": "你好，这是一段测试配音。", "voice": "qwen3:vivian", "response_format": "wav"}' -o out.wav
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:7861/v1", api_key="你的接口Key")
client.audio.speech.create(model="tts-studio", voice="qwen3:vivian", input="你好",
                           response_format="wav").write_to_file("out.wav")
```

可选参数：`speed`（0.5～2）、`instructions`（语气提示词）、`seed`、`response_format`（wav / mp3 / flac / aac / opus / pcm）。

默认只有本机能访问。在「开放接口」页勾选「允许局域网其他电脑访问」并重启后，同一个网络里的电脑也能调用；这时打开网页也要登录（用户名随便填，密码是接口 Key）。

## 接入 Agent（MCP）

Claude Code、Cursor、Trae、Codex、豆包 / 扣子、Cherry Studio 等 Agent 接上后，可以直接说“用某某音色把这段稿子配出来”。

软件左边「MCP 连接」页会显示现在有哪些 Agent 连着（已连接 / 已断开），并给出各家软件的接入配置，点一下就能复制。两种接法：

- **本机启动（stdio）**：Agent 自己启动 `mcp_server.py`，适合 Claude Code、Claude Desktop、Codex。
- **填网址（Streamable HTTP）**：`http://127.0.0.1:端口/mcp`，加请求头 `Authorization: Bearer <接口Key>`；只能填网址的软件，就用 `.../mcp?key=<接口Key>`。适合 Cursor、Trae、Cherry Studio，以及豆包 / 扣子里能加自定义 MCP 的地方。云端的智能体连不到本机，需要先开局域网访问，再做内网穿透。

| 工具 | 作用 |
|---|---|
| `list_voices` | 音色库里的音色 + 各模型自带的音色 ID |
| `list_models` | 各模型能不能用、不能用的原因 |
| `speak` | 一段文字 → 音频文件，返回路径和时长 |
| `dub_script` | 整篇稿子 → 自动拆句、逐句生成、按停顿合并成一条 wav，返回每句的起止秒数（可直接对字幕/动画） |

- 软件没开也没关系：第一次调用时会在后台自动启动（不弹窗口），之后一直开着，下次不用重新加载模型。
- 接口 Key 自动从本机 `data/config/settings.json` 读，不用手填。
- 音频默认存到 `data/outputs/mcp/`，也可以让 Agent 指定 `output_path`。

**Claude Code**（命令行）：

```bash
claude mcp add tts-studio --scope user -- "<安装目录>\.venv\Scripts\python.exe" "<安装目录>\mcp_server.py"
```

**Cursor / Claude Desktop / 其他支持 MCP 的工具**（配置文件里的 `mcpServers`）：

```json
{
  "mcpServers": {
    "tts-studio": {
      "command": "<安装目录>\\.venv\\Scripts\\python.exe",
      "args": ["<安装目录>\\mcp_server.py"]
    }
  }
}
```

连另一台电脑上的 TTS Studio（那边要开局域网访问）：加环境变量 `TTS_STUDIO_URL=http://那台电脑IP:7872` 和 `TTS_STUDIO_KEY=接口Key`。

## 数据放在哪

你自己的东西都在 `data/` 下，这个文件夹不会进 git：

```
data/
  config/settings.json   设置和 API Key
  models/                下载的模型
  voices/库/             音色库（每个音色一个文件夹）
  voices/语气.json        语气库
  projects/              每篇稿子的句子和音频
  outputs/               音色设计、API 试听的音频
```

## 负责任地使用

克隆声音之前，请先取得声音本人的同意。不要用这个工具冒充他人、做诈骗或发布误导性内容。

## 许可证

本项目代码：Apache License 2.0。`third_party/breeze/` 是 [breeze-tts](https://github.com/breezeblue-ai/breeze-tts) 的推理代码（Apache 2.0）。各模型和 API 的许可见上表和 [NOTICE](NOTICE)。
