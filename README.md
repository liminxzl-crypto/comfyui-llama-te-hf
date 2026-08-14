# ComfyUI Llama TE HF

`comfyUI-llama-TE` 的无 `llama.cpp` / 无 `llama-cpp-python` 版本。

本仓库保留原节点的工作流习惯和中文界面，但推理后端改为 Hugging Face Transformers。适合不想安装 GGUF、`llama-cpp-python`、`mmproj`，而想直接使用 Qwen2-VL / Qwen2.5-VL 等 HF 视觉语言模型的用户。

## 功能

- Qwen HF TE 模型加载器
- Qwen HF TE 图像推理
- Qwen HF TE 多轮对话聊天
- Qwen HF TE 对话增强设置
- Qwen HF TE Skill 加载器
- Qwen HF TE 卸载模型
- Gemma4 HF TE 模型加载器
- Gemma4 HF TE 图片推理
- Gemma4 HF TE 卸载模型

图像推理支持：

- 图片：只读取第一张图
- 逐帧：逐张图片分别推理
- 视频：从图片序列均匀抽帧后一次性推理
- 文本：纯文本推理

多轮聊天支持：

- ComfyUI 内嵌聊天界面
- 图片上传并保留到对话历史
- Skill 自动选择或固定选择
- Skill reference 按需加载
- 选项按钮、复制消息、复制代码、重新生成最后一条回复

## 模型选择与下载

这个后端使用 Hugging Face Transformers，不是 llama.cpp/GGUF，所以请选择官方或社区提供的 HF 格式模型目录。不要选择 `.gguf`，也不需要 `mmproj`。

### 最推荐

综合稳定性、速度、中文图片理解、多帧输入和多轮聊天，当前最推荐：

1. `Qwen/Qwen2.5-VL-3B-Instruct`：首选，适合大多数 6GB 到 10GB 显存用户，速度快，兼容性最好。
2. `Qwen/Qwen2.5-VL-7B-Instruct`：图片描述和推理质量更好，建议 10GB 到 16GB 以上显存。
3. `Qwen/Qwen3-VL-4B-Instruct`：新版官方 Qwen3-VL，适合想尝试新模型的用户；如果遇到 chat template 或 processor 兼容问题，优先换回 Qwen2.5-VL。

为什么这里默认推荐 Qwen2.5-VL：它在 HF Transformers 里的 `AutoProcessor`、chat template、图片/多图输入支持更稳定。Qwen3.5-VL / Qwen3.6-VL 目前没有确认稳定的官方 HF Transformers 仓库；搜索到的多为第三方、GGUF、MLX 或量化版本，不建议作为这个节点的默认模型。

### 按用途选择

| 用途 | 推荐模型 | 说明 |
| --- | --- | --- |
| 低显存、快速反推、首次测试 | `Qwen/Qwen2.5-VL-3B-Instruct` | 最稳的起点 |
| 更高质量图片描述 | `Qwen/Qwen2.5-VL-7B-Instruct` | 质量和速度比较均衡 |
| 视频抽帧/多图分析 | `Qwen/Qwen2.5-VL-3B-Instruct`、`Qwen/Qwen2.5-VL-7B-Instruct` | 多帧会增加显存，建议降低“最多帧数”和“最大边长” |
| 新版 Qwen 视觉模型 | `Qwen/Qwen3-VL-4B-Instruct`、`Qwen/Qwen3-VL-8B-Instruct` | 可用，但属于 HF 后端的次优先选项 |
| 极小显存/CPU 尝试 | `OpenGVLab/InternVL2_5-1B`、`OpenGVLab/InternVL2_5-2B`、`llava-hf/llava-onevision-qwen2-0.5b-ov-hf` | 质量弱一些，但更容易跑起来 |
| 高质量图片理解 | `OpenGVLab/InternVL2_5-4B`、`OpenGVLab/InternVL2_5-8B` | 可作为 Qwen2.5-VL 之外的备选 |
| 高显存/强推理 | `Qwen/Qwen2.5-VL-32B-Instruct`、`Qwen/Qwen3-VL-32B-Instruct`、`OpenGVLab/InternVL2_5-26B` | 建议 24GB 以上显存 |
| 实验性兼容模型 | `microsoft/Phi-3.5-vision-instruct`、`THUDM/glm-4v-9b` | 能用但不同版本的模板兼容性可能有差异 |

显存只是粗略参考，实际占用还受图片分辨率、帧数、上下文长度、量化方式和 `device_map` 影响。

### 可用模型和下载地址

| 模型 ID | 适合功能 | 建议 | 下载 |
| --- | --- | --- | --- |
| `Qwen/Qwen2.5-VL-3B-Instruct` | 图片、多帧、文本、多轮聊天 | 最推荐 | [HF](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) / [ModelScope](https://modelscope.cn/models/Qwen/Qwen2.5-VL-3B-Instruct) |
| `Qwen/Qwen2.5-VL-7B-Instruct` | 图片理解、视频帧分析、长回复 | 质量优先 | [HF](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) / [ModelScope](https://modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct) |
| `Qwen/Qwen2.5-VL-32B-Instruct` | 强推理、复杂图片任务 | 24GB+ 显存 | [HF](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct) / [ModelScope](https://modelscope.cn/models/Qwen/Qwen2.5-VL-32B-Instruct) |
| `Qwen/Qwen2.5-VL-72B-Instruct` | 最高质量本地推理 | 多卡/大显存 | [HF](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct) / [ModelScope](https://modelscope.cn/models/Qwen/Qwen2.5-VL-72B-Instruct) |
| `Qwen/Qwen2-VL-2B-Instruct` | 低显存、快速测试 | 旧版但轻量 | [HF](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) / [ModelScope](https://modelscope.cn/models/Qwen/Qwen2-VL-2B-Instruct) |
| `Qwen/Qwen2-VL-7B-Instruct` | 图片理解、文本推理 | 旧版备选 | [HF](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) / [ModelScope](https://modelscope.cn/models/Qwen/Qwen2-VL-7B-Instruct) |
| `Qwen/Qwen3-VL-4B-Instruct` | 新版视觉理解 | 推荐尝试 | [HF](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) |
| `Qwen/Qwen3-VL-8B-Instruct` | 新版视觉理解 | 12GB+ 显存更稳 | [HF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |
| `Qwen/Qwen3-VL-32B-Instruct` | 新版强推理 | 24GB+ 显存 | [HF](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) |
| `OpenGVLab/InternVL2_5-1B` | 低显存图片理解 | 轻量备选 | [HF](https://huggingface.co/OpenGVLab/InternVL2_5-1B) |
| `OpenGVLab/InternVL2_5-2B` | 低显存图片理解 | 轻量备选 | [HF](https://huggingface.co/OpenGVLab/InternVL2_5-2B) |
| `OpenGVLab/InternVL2_5-4B` | 图片描述、OCR、视觉问答 | 质量较好 | [HF](https://huggingface.co/OpenGVLab/InternVL2_5-4B) |
| `OpenGVLab/InternVL2_5-8B` | 图片描述、复杂视觉任务 | 12GB+ 显存 | [HF](https://huggingface.co/OpenGVLab/InternVL2_5-8B) |
| `OpenGVLab/InternVL2_5-26B` | 高质量视觉推理 | 24GB+ 显存 | [HF](https://huggingface.co/OpenGVLab/InternVL2_5-26B) |
| `OpenGVLab/InternVL2_5-38B` | 高质量视觉推理 | 大显存/多卡 | [HF](https://huggingface.co/OpenGVLab/InternVL2_5-38B) |
| `microsoft/Phi-3.5-vision-instruct` | 图片理解、轻量多语 | 实验性兼容 | [HF](https://huggingface.co/microsoft/Phi-3.5-vision-instruct) |
| `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` | 极低显存图片理解 | 实验性兼容 | [HF](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf) |
| `THUDM/glm-4v-9b` | 中文视觉理解 | 实验性兼容 | [HF](https://huggingface.co/THUDM/glm-4v-9b) |

不建议在这个 HF 后端中使用：

- `.gguf` 文件：这是 llama.cpp 格式，本节点不会加载。
- 需要 `mmproj` 的模型目录：本节点不需要也不会使用 llama.cpp 的视觉投影文件。
- 未确认支持 Transformers 的第三方 Qwen3.5/Qwen3.6 量化仓库：可能是 GGUF、MLX 或专用推理格式，不能保证 `AutoProcessor` / `AutoModelForVision2Seq` 能加载。

## 快速安装

推荐直接用 `git clone` 安装，后续更新最方便。

### 1. 进入 ComfyUI 自定义节点目录

如果你已经在 ComfyUI 根目录：

```bash
cd custom_nodes
```

如果还没进入 ComfyUI 根目录，请先进入你的 ComfyUI 目录，例如：

```bash
cd /你的/ComfyUI/custom_nodes
```

Windows 示例：

```bat
cd /d D:\ComfyUI\custom_nodes
```

### 2. 克隆仓库

```bash
git clone https://github.com/liminxzl-crypto/comfyui-llama-te-hf.git
cd comfyui-llama-te-hf
```

克隆完成后，插件目录应该是：

```text
ComfyUI/custom_nodes/comfyui-llama-te-hf/
```

### 3. 安装依赖

普通 Python / venv 环境：

```bash
pip install -r requirements.txt
pip install -U transformers accelerate
```

如果使用 ComfyUI 整合包，请使用整合包里的 Python：

```bat
..\..\python_embeded\python.exe -m pip install -r requirements.txt
..\..\python_embeded\python.exe -m pip install -U transformers accelerate
```

macOS / Linux 如果默认 `python3` 指向 ComfyUI 所在环境，也可以用：

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -U transformers accelerate
```

### 4. 重启 ComfyUI

完全重启 ComfyUI 后，在节点菜单里搜索：

```text
Qwen HF TE
```

### 手动安装

如果你不想使用 git：

1. 打开 <https://github.com/liminxzl-crypto/comfyui-llama-te-hf>
2. 点击 `Code -> Download ZIP`
3. 解压到 `ComfyUI/custom_nodes/comfyui-llama-te-hf/`
4. 按上面的“安装依赖”步骤执行

注意：手动下载 ZIP 后，目录不要多套一层，例如不要变成：

```text
ComfyUI/custom_nodes/comfyui-llama-te-hf-main/comfyui-llama-te-hf/
```

## 更新方法

如果你是用 `git clone` 安装的：

```bash
cd ComfyUI/custom_nodes/comfyui-llama-te-hf
git pull
pip install -r requirements.txt
pip install -U transformers accelerate
```

整合包示例：

```bat
cd /d D:\ComfyUI\custom_nodes\comfyui-llama-te-hf
git pull
..\..\python_embeded\python.exe -m pip install -r requirements.txt
..\..\python_embeded\python.exe -m pip install -U transformers accelerate
```

如果你是手动下载 ZIP 安装的，重新下载 ZIP 并覆盖旧目录即可。

## 模型放置

有两种方式。

模型系列建议和“主模型”保持一致：

- 使用推荐默认模型时，模型系列选择 `Qwen2.5-VL`
- 使用 Qwen2-VL 模型时，模型系列选择 `Qwen2-VL`
- 如果使用 Qwen3.x 的 HF 格式模型，再选择对应的 `Qwen3-VL`、`Qwen3.5-VL` 或 `Qwen3.6-VL`

当前 HF 后端最稳定、最推荐的是 `Qwen2.5-VL`。模型系列参数主要用于界面标识和兼容旧工作流，真正决定模型的是“主模型”里填写的模型 ID 或本地路径。

方式一：直接在节点里填写 Hugging Face 模型 ID：

```text
Qwen/Qwen2.5-VL-3B-Instruct
```

首次运行会自动下载到 Hugging Face 缓存。

方式二：手动下载整个模型目录，放到：

```text
ComfyUI/models/LLM/Qwen2.5-VL-3B-Instruct/
```

然后在节点“主模型”中选择或填写：

```text
Qwen2.5-VL-3B-Instruct
```

注意：本版本不支持 `.gguf` 文件，也不需要 `mmproj`。

## 基本工作流

图片反推：

1. 添加 `Qwen HF TE 模型加载器（无 llama）`
2. “主模型”选择 `Qwen/Qwen2.5-VL-3B-Instruct`
3. 添加 `Qwen HF TE 图像推理`
4. 把 `qwen模型` 输出连到图像推理节点
5. 接入 `IMAGE`
6. “输入模式”选择 `图片`
7. “提示词”填写你想要的任务，例如：

```text
请详细描述这张图片，适合作为视频生成提示词。
```

视频抽帧分析：

1. 把视频帧作为 `IMAGE` batch 输入
2. “输入模式”选择 `视频`
3. 设置“最多帧数”
4. 模型会把多张图一起作为序列分析

多轮聊天：

1. 添加 `Qwen HF TE 多轮对话聊天`
2. 连接 `Qwen HF TE 模型加载器`
3. 可选：连接 `Qwen HF TE Skill加载器`
4. 可选：连接 `Qwen HF TE 对话增强设置`
5. 在节点内输入消息或上传图片

## 参数说明

这些参数从原节点保留，HF 后端会忽略：

- `视觉投影mmproj`
- `上下文长度`
- `GPU层数`
- `KV缓存K类型`
- `KV缓存V类型`
- `MoE专家上CPU`
- `前N层专家上CPU`

这些参数只是为了兼容旧工作流。HF 后端会自动选择设备：

- CUDA：优先使用，自动选择 bf16 或 fp16
- Apple Silicon MPS：使用 MPS + fp16
- CPU：使用 fp32

## Skill

Skill 文件位于：

```text
skills/
```

这部分沿用原节点的 Skill 协议。模型会根据 `SKILL.md` 分阶段提问、按需请求 reference，并在最终结果前输出状态标记。聊天界面会把状态标记解析为阶段和选项按钮。

如果你只是做图片反推或普通聊天，不需要连接 Skill 加载器。

## 限制

- 不支持 llama.cpp GGUF 模型
- 不支持 llama.cpp `mmproj`
- Gemma4 音频推理暂未实现，节点会明确报错
- 多轮聊天界面没有改动原前端结构，仍使用 `QwenTE_MultiTurnChat` 类名
- 某些 HF 视觉模型的 chat template 差异较大；推荐优先使用 Qwen2.5-VL

## 常见问题

### 提示找不到 transformers

在 ComfyUI 使用的 Python 环境里安装：

```bash
pip install transformers accelerate pillow safetensors
```

### 提示不支持 GGUF

这是预期行为。本仓库刻意不使用 llama.cpp，请下载 Hugging Face 格式模型目录。

### 图片对话报错

确认你加载的是视觉语言模型，不是纯文本模型。推荐使用 `Qwen/Qwen2.5-VL-3B-Instruct`。

### 显存不够

先尝试：

- 使用 3B 模型
- 降低“最大边长”到 512 或 768
- 降低“最大生成token”
- 逐帧模式改成单图模式

### MPS 很慢或报错

部分模型在 MPS 上支持不完整。可以在启动 ComfyUI 前临时设置：

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

如果仍有问题，建议使用 CUDA 或改用较小模型。

## 致谢

本项目基于 `tl2012tl/comfyUI-llama-TE` 的节点结构、中文参数和 Skill/聊天界面改造，将推理后端替换为 Hugging Face Transformers。
