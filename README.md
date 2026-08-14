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

## 推荐模型

首选 Qwen2.5-VL：

- `Qwen/Qwen2.5-VL-3B-Instruct`
- `Qwen/Qwen2.5-VL-7B-Instruct`

也可以尝试：

- `Qwen/Qwen2-VL-2B-Instruct`
- `Qwen/Qwen2-VL-7B-Instruct`
- `OpenGVLab/InternVL2_5-2B`
- `OpenGVLab/InternVL2_5-4B`
- `OpenGVLab/InternVL2_5-8B`
- `zai-org/GLM-4.1V-Thinking-9B`
- `microsoft/Phi-3.5-vision-instruct`
- `llava-hf/llava-onevision-qwen2-0.5b-ov-hf`

模型下载：

- Hugging Face: <https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct>
- ModelScope: <https://modelscope.cn/models/Qwen/Qwen2.5-VL-3B-Instruct>

如果显存不大，先从 3B 开始。跑通后再换 7B 或更大的模型。

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
