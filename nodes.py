# -*- coding: utf-8 -*-
import gc
import inspect
import io
import base64
import os
import random
import re
from dataclasses import dataclass
from functools import wraps

import numpy as np
from PIL import Image

import folder_paths
import comfy.model_management as mm


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


any_type = AnyType("*")

默认图片提示词 = ""
默认图片系统提示词 = "描述这张图,300字左右."
默认文本系统提示词 = "描述这张图,300字左右."
默认KV缓存类型 = "默认(F16)"
Q8_0缓存类型 = "q8_0"
KV缓存类型选项 = [默认KV缓存类型, Q8_0缓存类型]
HF后端无mmproj = "无（HF后端不需要）"

常用HF模型 = [
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2-VL-2B-Instruct",
    "Qwen/Qwen2-VL-7B-Instruct",
    "OpenGVLab/InternVL2_5-2B",
    "OpenGVLab/InternVL2_5-4B",
    "OpenGVLab/InternVL2_5-8B",
    "THUDM/glm-4v-9b",
    "zai-org/GLM-4.1V-Thinking-9B",
    "microsoft/Phi-3.5-vision-instruct",
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
]


def _确保_llm目录已注册():
    folder_name = "LLM"
    llm_dir = os.path.join(folder_paths.models_dir, folder_name)
    supported_exts = set(getattr(folder_paths, "supported_pt_extensions", set()))
    llm_exts = supported_exts | {".gguf", ".safetensors", ".bin", ".json"}
    try:
        if folder_name not in folder_paths.folder_names_and_paths:
            folder_paths.folder_names_and_paths[folder_name] = ([llm_dir], llm_exts)
            return
        paths, exts = folder_paths.folder_names_and_paths[folder_name]
        if llm_dir not in paths:
            paths.append(llm_dir)
        if isinstance(exts, set):
            exts.update(llm_exts)
        else:
            folder_paths.folder_names_and_paths[folder_name] = (paths, set(exts) | llm_exts)
    except Exception:
        return


def _列出llm文件():
    _确保_llm目录已注册()
    try:
        return folder_paths.get_filename_list("LLM")
    except Exception:
        return []


def _列出hf模型候选():
    _确保_llm目录已注册()
    llm_dir = os.path.join(folder_paths.models_dir, "LLM")
    candidates = list(常用HF模型)
    try:
        for name in sorted(os.listdir(llm_dir)):
            path = os.path.join(llm_dir, name)
            if os.path.isdir(path) or os.path.isfile(os.path.join(path, "config.json")):
                if name not in candidates:
                    candidates.append(name)
    except Exception:
        pass
    for filename in _列出llm文件():
        if filename.lower().endswith(".gguf") or "mmproj" in filename.lower():
            continue
        if filename not in candidates:
            candidates.append(filename)
    return candidates


def _列出mmproj候选():
    files = [f for f in _列出llm文件() if "mmproj" in f.lower()]
    return [HF后端无mmproj] + files


def _缩放图片到最大边(pil, 最大边长):
    if 最大边长 <= 0:
        return pil
    w, h = pil.size
    long_edge = max(w, h)
    if long_edge <= 最大边长:
        return pil
    scale = 最大边长 / float(long_edge)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return pil.resize((new_w, new_h), resample=Image.BICUBIC)


def _图片张量转PIL(image_tensor, index, 最大边长):
    if image_tensor is None:
        raise ValueError("图片输入为空。")
    if index < 0 or index >= int(image_tensor.shape[0]):
        raise IndexError(f"图片索引越界：{index}")
    img = image_tensor[index].cpu().numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img).convert("RGB")
    return _缩放图片到最大边(pil, int(最大边长))


def _本地图片文件转data_uri(image_path, 最大边长):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到图片文件：{image_path}")

    with Image.open(image_path) as pil:
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        pil = _缩放图片到最大边(pil, int(最大边长))
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=90)
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{image_b64}"


def _清洗think块文本(text):
    if not isinstance(text, str) or not text:
        return "" if text is None else str(text)
    cleaned = text
    cleaned = re.sub(r"<think\b[^>]*>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if re.search(r"</think>", cleaned, flags=re.IGNORECASE):
        cleaned = re.sub(r"^.*?</think>\s*", "", cleaned, count=1, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.replace("<think>", "").replace("</think>", "")
    return cleaned.strip()


def _规范化随机种子(seed_value):
    try:
        seed_value = int(seed_value)
    except Exception:
        return None
    if seed_value < 0:
        return None
    return seed_value


def _设置随机种子(seed_value):
    seed = _规范化随机种子(seed_value)
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _解析模型路径(模型名称):
    raw = str(模型名称 or "").strip()
    if not raw:
        raise ValueError("请填写模型名称、models/LLM 相对路径或本地绝对路径。")
    if raw.lower().endswith(".gguf"):
        raise ValueError("HF/Transformers 后端不支持 GGUF 文件。请使用 Hugging Face 格式的 Qwen2-VL/Qwen2.5-VL 等模型。")
    if os.path.exists(raw):
        return raw
    llm_dir = os.path.join(folder_paths.models_dir, "LLM")
    candidate = os.path.join(llm_dir, raw)
    if os.path.exists(candidate):
        return candidate
    if "/" in raw and os.sep not in raw:
        return raw
    raise FileNotFoundError(
        f"找不到模型：{raw}。可放到 ComfyUI/models/LLM/ 后填写相对路径，或直接填写 Hugging Face 模型 ID。"
    )


def _选择设备和精度():
    import torch

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return "cuda", dtype, "auto"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps", torch.float16, "direct"
    return "cpu", torch.float32, "direct"


def _加载_transformers_model(model_path):
    from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    except TypeError:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    _device, dtype, placement = _选择设备和精度()
    base_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True, "torch_dtype": dtype}
    if placement == "auto":
        base_kwargs["device_map"] = "auto"

    errors = []
    for model_cls in (AutoModelForVision2Seq, AutoModelForCausalLM):
        try:
            try:
                model = model_cls.from_pretrained(
                    model_path,
                    attn_implementation="sdpa",
                    **base_kwargs,
                )
            except (TypeError, ValueError) as exc:
                if "attn_implementation" not in str(exc).lower():
                    raise
                model = model_cls.from_pretrained(model_path, **base_kwargs)
            if placement == "direct":
                model = model.to(_device)
            model.eval()
            return processor, model, _device
        except Exception as exc:
            errors.append(f"{model_cls.__name__}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Transformers 模型加载失败：\n" + "\n".join(errors))


def _模型设备(model):
    import torch

    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict) and device_map:
        return torch.device(next(iter(device_map.values())))
    return model.device


def _应用聊天模板(processor, messages, think):
    apply = getattr(processor, "apply_chat_template", None)
    if apply is None or not callable(apply):
        raise RuntimeError("当前 processor 不支持 apply_chat_template，无法使用该节点。")
    try:
        sig = inspect.signature(apply)
    except Exception:
        sig = None

    kwargs = {"conversation": messages, "tokenize": False, "add_generation_prompt": True}
    if sig is not None and "enable_thinking" in sig.parameters:
        kwargs["enable_thinking"] = bool(think)

    try:
        return apply(**kwargs)
    except TypeError as exc:
        if "enable_thinking" in str(exc).lower():
            kwargs.pop("enable_thinking", None)
            return apply(**kwargs)
        if "conversation" in str(exc).lower():
            kwargs.pop("conversation", None)
            return apply(messages, **kwargs)
        raise
    except Exception:
        if messages and messages[0].get("role") == "system":
            merged = str(messages[0].get("content", "")).strip()
            if messages[-1].get("role") == "user":
                merged = merged + "\n\n" + str(messages[-1].get("content", "")).strip()
            kwargs["conversation"] = [{"role": "user", "content": merged}]
            return apply(**kwargs)
        raise


def _构建生成参数(温度, top_p, top_k, 重复惩罚, 最大生成token):
    do_sample = float(温度) > 0.0
    params = {
        "max_new_tokens": max(1, int(最大生成token)),
        "do_sample": do_sample,
        "repetition_penalty": float(重复惩罚),
    }
    if do_sample:
        params["temperature"] = float(温度)
        if 0.0 < float(top_p) < 1.0:
            params["top_p"] = float(top_p)
        if int(top_k) > 0:
            params["top_k"] = int(top_k)
    return params


def _生成文本(model, processor, prompt_text, images, 最大生成token, 温度, top_p, top_k, 重复惩罚):
    import torch

    processor_kwargs = {"text": [prompt_text], "return_tensors": "pt", "padding": True}
    if images:
        processor_kwargs["images"] = images
    inputs = processor(**processor_kwargs)
    device = _模型设备(model)
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}

    gen_kwargs = _构建生成参数(温度, top_p, top_k, 重复惩罚, 最大生成token)
    pad_token_id = getattr(getattr(processor, "tokenizer", None), "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(model.config, "eos_token_id", None)
    if pad_token_id is not None:
        gen_kwargs["pad_token_id"] = pad_token_id

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)
    input_length = int(inputs["input_ids"].shape[1])
    output_ids = generated_ids[:, input_length:]
    return processor.batch_decode(
        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


@dataclass
class _HFModel:
    model: object
    processor: object
    device: str
    settings: dict

    def __post_init__(self):
        # 原多轮聊天/Skill 模块会读取 .llm 和 .chat_handler；这里把 HF 模型包装成兼容对象。
        self.llm = _HFChatCompletionBackend(self)
        self.chat_handler = object()


class _QwenStorage:
    model = None

    @classmethod
    def unload(cls):
        if cls.model is not None:
            try:
                model = cls.model.model
                if hasattr(model, "cpu"):
                    model.cpu()
            except Exception:
                pass
        cls.model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:
            pass
        mm.soft_empty_cache()

    @classmethod
    def load(cls, config):
        if cls.model and cls.model.settings == config:
            return cls.model
        cls.unload()

        model_path = _解析模型路径(config["model"])
        processor, model, device = _加载_transformers_model(model_path)
        cls.model = _HFModel(
            model=model,
            processor=processor,
            device=device,
            settings=dict(config),
        )
        return cls.model


def _重置llm推理状态(llm):
    # llama.cpp 版本会清理 KV cache；Transformers 的 generate 每次按输入重新前向，这里只做 GC 兜底。
    gc.collect()


def _调用chat_completion(llm, *, messages, params):
    if isinstance(llm, _HFChatCompletionBackend):
        return llm.create_chat_completion(messages=messages, **dict(params or {}))
    if hasattr(llm, "create_chat_completion"):
        return llm.create_chat_completion(messages=messages, **dict(params or {}))
    raise RuntimeError("当前模型对象不支持 create_chat_completion。")


class _HFChatCompletionBackend:
    def __init__(self, bundle):
        self.bundle = bundle

    def tokenize(self, text, add_bos=False):
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="ignore")
        tokenizer = getattr(self.bundle.processor, "tokenizer", None)
        if tokenizer is None or not hasattr(tokenizer, "encode"):
            return list(range(max(1, len(text) // 3)))
        return tokenizer.encode(text, add_special_tokens=add_bos)

    def create_chat_completion(self, messages=None, **params):
        _设置随机种子(params.get("seed"))
        model = self.bundle.model
        processor = self.bundle.processor
        think = bool(self.bundle.settings.get("think", False))
        prompt, images = self._build_prompt(messages or [], processor, think)

        generated = _生成文本(
            model,
            processor,
            prompt,
            images,
            params.get("max_tokens", params.get("max_new_tokens", 1024)),
            params.get("temperature", 0.7),
            params.get("top_p", 0.9),
            params.get("top_k", 20),
            params.get("repeat_penalty", params.get("repetition_penalty", 1.0)),
        )
        return {"choices": [{"message": {"content": generated}}]}

    def _build_prompt(self, messages, processor, think):
        normalized = []
        images = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if isinstance(content, str):
                normalized.append({"role": role, "content": content})
                continue
            if not isinstance(content, list):
                normalized.append({"role": role, "content": str(content)})
                continue

            text_parts = []
            for item in content:
                if not isinstance(item, dict):
                    text_parts.append(str(item))
                    continue
                item_type = item.get("type", "")
                if item_type == "text":
                    text_parts.append(str(item.get("text", "")))
                elif item_type in ("image_url", "image"):
                    image = self._extract_image(item)
                    if image is not None:
                        images.append(image)
                        text_parts.append({"type": "image"})
                elif item_type == "input_audio":
                    raise NotImplementedError("HF 后端暂不支持音频输入。")
            normalized.append({"role": role, "content": text_parts})

        prompt = _应用聊天模板(processor, normalized, think)
        return prompt, images

    def _extract_image(self, item):
        if item.get("type") == "image" and "image" in item:
            return item["image"]

        image_url = item.get("image_url")
        if isinstance(image_url, dict):
            url = str(image_url.get("url", ""))
        else:
            url = str(image_url or "")

        if not url or url.startswith("data:;"):
            return None

        if url.startswith("data:image/"):
            header, _, payload = url.partition(",")
            image_bytes = base64.b64decode(payload)
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if url.startswith(("http://", "https://")):
            import urllib.request

            with urllib.request.urlopen(url, timeout=30) as response:
                return Image.open(io.BytesIO(response.read())).convert("RGB")

        return Image.open(url).convert("RGB")


class _Gemma4HFStorage(_QwenStorage):
    model = None


def _安装全局卸载挂钩():
    try:
        if hasattr(mm, "_qwen_te_hf_unload_hook_installed") and mm._qwen_te_hf_unload_hook_installed:
            return
        original = getattr(mm, "unload_all_models", None)
        if original is None or not callable(original):
            return

        @wraps(original)
        def wrapped_unload_all_models(*args, **kwargs):
            try:
                _QwenStorage.unload()
            except Exception:
                pass
            try:
                _Gemma4HFStorage.unload()
            except Exception:
                pass
            return original(*args, **kwargs)

        mm.unload_all_models = wrapped_unload_all_models
        mm._qwen_te_hf_unload_hook_installed = True
    except Exception:
        return


_安装全局卸载挂钩()


class QwenTE模型加载器:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "模型系列": (["Qwen3-VL", "Qwen3.5-VL", "Qwen3.6-VL"], {"default": "Qwen3.6-VL"}),
                "主模型": (
                    "STRING",
                    {
                        "default": "Qwen/Qwen2.5-VL-3B-Instruct",
                        "multiline": False,
                        "tooltip": (
                            "可直接粘贴 Hugging Face 模型 ID，也可填 ComfyUI/models/LLM "
                            "下的相对路径或本地绝对路径。"
                        ),
                    },
                ),
                "视觉投影mmproj": (
                    _列出mmproj候选(),
                    {
                        "default": HF后端无mmproj,
                        "tooltip": "HF/Transformers 后端不需要 llama.cpp 的 mmproj，保留该参数仅为兼容旧工作流。",
                    },
                ),
                "启用思考": ("BOOLEAN", {"default": False, "tooltip": "若模型模板支持 enable_thinking，会传给模板；否则自动忽略。"}),
                "保留历史think": ("BOOLEAN", {"default": False, "tooltip": "当前 HF 节点未实现多轮历史，该参数保留以兼容旧工作流。"}),
                "上下文长度": ("INT", {"default": 8192, "min": 1024, "max": 327680, "step": 256, "tooltip": "保留以兼容原节点；Transformers 会根据模型配置处理上下文。"}),
                "GPU层数": ("INT", {"default": -1, "min": -1, "max": 9999, "step": 1, "tooltip": "保留以兼容原节点；HF 后端会自动使用 CUDA/MPS/CPU。"}),
                "KV缓存K类型": (KV缓存类型选项, {"default": 默认KV缓存类型, "tooltip": "llama.cpp 参数，HF 后端会忽略。"}),
                "KV缓存V类型": (KV缓存类型选项, {"default": 默认KV缓存类型, "tooltip": "llama.cpp 参数，HF 后端会忽略。"}),
                "MoE专家上CPU": ("BOOLEAN", {"default": False, "tooltip": "llama.cpp 参数，HF 后端会忽略。"}),
                "前N层专家上CPU": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1, "tooltip": "llama.cpp 参数，HF 后端会忽略。"}),
            }
        }

    RETURN_TYPES = ("QWENLLAMA",)
    RETURN_NAMES = ("qwen模型",)
    FUNCTION = "load"
    CATEGORY = "Qwen TE"

    def load(
        self,
        模型系列,
        主模型,
        视觉投影mmproj,
        启用思考,
        保留历史think,
        上下文长度,
        GPU层数,
        KV缓存K类型,
        KV缓存V类型,
        MoE专家上CPU,
        前N层专家上CPU,
    ):
        config = {
            "backend": "transformers",
            "family": 模型系列,
            "model": 主模型,
            "mmproj": 视觉投影mmproj,
            "think": bool(启用思考),
            "preserve_thinking": bool(保留历史think),
            "cpu_moe": bool(MoE专家上CPU),
            "n_cpu_moe": int(前N层专家上CPU),
            "n_ctx": int(上下文长度),
            "n_gpu_layers": int(GPU层数),
            "cache_type_k": KV缓存K类型,
            "cache_type_v": KV缓存V类型,
        }
        return (_QwenStorage.load(config),)


class _HF图像推理基类:
    模型输入名 = "qwen模型"
    模型类型 = "QWENLLAMA"
    CATEGORY = "Qwen TE"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                s.模型输入名: (s.模型类型,),
                "输入模式": (
                    ["图片", "逐帧", "视频", "文本"],
                    {
                        "default": "图片",
                        "tooltip": "图片=只读第1张；逐帧=一张一张推理；视频=抽帧后一次性推理；文本=仅文字输入，无需图片。",
                    },
                ),
                "提示词": ("STRING", {"default": 默认图片提示词, "multiline": True}),
                "系统提示词": ("STRING", {"default": 默认图片系统提示词, "multiline": True}),
                "最多帧数": ("INT", {"default": 24, "min": 2, "max": 1024, "step": 1}),
                "最大边长": ("INT", {"default": 1024, "min": 128, "max": 16384, "step": 64}),
                "最大生成token": ("INT", {"default": 1024, "min": 20, "max": 0xFFFFFFFFFFFFFFFF, "step": 1}),
                "温度": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 200, "step": 1}),
                "重复惩罚": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01}),
                "频率惩罚": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "保留以兼容原节点；Transformers generate 不直接使用该参数。"}),
                "存在惩罚": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "保留以兼容原节点；Transformers generate 不直接使用该参数。"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1, "control_after_generate": True}),
                "输出think块": ("BOOLEAN", {"default": True}),
                "生成后自动卸载模型": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "图片": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("文本",)
    FUNCTION = "run"
    CATEGORY = "Qwen TE"

    def run(
        self,
        **kwargs,
    ):
        return self._run(**kwargs)

    def _run(
        self,
        输入模式,
        提示词,
        系统提示词,
        最多帧数,
        最大边长,
        最大生成token,
        温度,
        top_p,
        top_k,
        重复惩罚,
        频率惩罚,
        存在惩罚,
        seed,
        输出think块,
        生成后自动卸载模型=False,
        图片=None,
        **_ignored,
    ):
        del 频率惩罚, 存在惩罚, _ignored
        model_bundle = self._同步模型(kwargs.get(self.模型输入名))
        processor = model_bundle.processor
        model = model_bundle.model
        think = bool(model_bundle.settings.get("think", False))

        messages = []
        system_text = (系统提示词 or "").strip()
        if 输入模式 == "文本":
            if not system_text or system_text == 默认图片系统提示词:
                system_text = 默认文本系统提示词
        elif 输入模式 == "视频" and system_text:
            system_text = "请将输入的图片序列当做视频而不是静态帧序列, " + system_text
        if system_text:
            messages.append({"role": "system", "content": system_text})

        total_images = int(图片.shape[0]) if 图片 is not None else 0
        if 输入模式 in ("图片", "逐帧", "视频") and total_images == 0:
            raise ValueError("未检测到图片输入。")

        if 输入模式 == "图片":
            frame_indices = [0]
        elif 输入模式 == "逐帧":
            frame_indices = list(range(total_images))
        elif 输入模式 == "视频":
            if total_images == 1:
                frame_indices = [0]
            else:
                count = min(max(int(最多帧数), 2), total_images)
                frame_indices = np.linspace(0, total_images - 1, count, dtype=int).tolist()
        elif 输入模式 == "文本":
            frame_indices = []
        else:
            raise ValueError(f"未知输入模式：{输入模式}")

        prompt_text = (提示词 or "").strip()
        if 输入模式 == "文本" and not prompt_text:
            raise ValueError("文本模式下，提示词不能为空。")

        _设置随机种子(seed)

        if 输入模式 == "文本":
            messages.append({"role": "user", "content": prompt_text})
            prompt = _应用聊天模板(processor, messages, think)
            text = _生成文本(
                model, processor, prompt, None, 最大生成token, 温度, top_p, top_k, 重复惩罚
            )
        elif 输入模式 == "逐帧":
            out_parts = []
            for idx, frame_index in enumerate(frame_indices):
                if mm.processing_interrupted():
                    raise mm.InterruptProcessingException()
                pil_image = _图片张量转PIL(图片, frame_index, int(最大边长))
                user_content = [
                    {"type": "text", "text": prompt_text},
                    {"type": "image"},
                ]
                frame_messages = list(messages) + [{"role": "user", "content": user_content}]
                prompt = _应用聊天模板(processor, frame_messages, think)
                part = _生成文本(
                    model,
                    processor,
                    prompt,
                    [pil_image],
                    最大生成token,
                    温度,
                    top_p,
                    top_k,
                    重复惩罚,
                )
                if len(frame_indices) > 1:
                    out_parts.append(f"====== 第{idx+1}帧 ======\n{part}".strip())
                else:
                    out_parts.append(str(part).strip())
            text = "\n\n".join([part for part in out_parts if part])
        else:
            user_content = [{"type": "text", "text": prompt_text}]
            pil_images = []
            for frame_index in frame_indices:
                if mm.processing_interrupted():
                    raise mm.InterruptProcessingException()
                user_content.append({"type": "image"})
                pil_images.append(_图片张量转PIL(图片, frame_index, int(最大边长)))
            messages.append({"role": "user", "content": user_content})
            prompt = _应用聊天模板(processor, messages, think)
            text = _生成文本(
                model,
                processor,
                prompt,
                pil_images,
                最大生成token,
                温度,
                top_p,
                top_k,
                重复惩罚,
            )

        if not bool(输出think块):
            text = _清洗think块文本(text)

        if mm.processing_interrupted():
            raise mm.InterruptProcessingException()

        result_text = text.lstrip().removeprefix(": ").strip()
        if bool(生成后自动卸载模型):
            self._storage().unload()
        return (result_text,)

    @staticmethod
    def _storage():
        return _QwenStorage

    def _加载配置(self, config):
        return self._storage().load(config)

    def _同步模型(self, qwen模型):
        storage = self._storage()
        if storage.model is None:
            need_reload = True
        elif qwen模型 is not storage.model:
            if hasattr(qwen模型, "settings") and getattr(qwen模型, "settings") == storage.model.settings:
                need_reload = False
                qwen模型 = storage.model
            else:
                need_reload = True
        else:
            need_reload = False

        if need_reload:
            if not hasattr(qwen模型, "settings"):
                raise RuntimeError("输入的模型对象缺少配置信息，无法自动重载。请先运行模型加载器。")
            qwen模型 = self._加载配置(qwen模型.settings)

        if qwen模型 is None or getattr(qwen模型, "model", None) is None:
            raise RuntimeError("模型对象内部实例无效，请重新加载模型。")
        return qwen模型


class QwenTE图像推理(_HF图像推理基类):
    模型输入名 = "qwen模型"
    模型类型 = "QWENLLAMA"
    CATEGORY = "Qwen TE"


class QwenTE卸载模型:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"任意输入": (any_type,)}}

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("任意输出",)
    FUNCTION = "run"
    CATEGORY = "Qwen TE"

    def run(self, 任意输入):
        _QwenStorage.unload()
        return (任意输入,)


class Gemma4TE模型加载器:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "主模型": (
                    "STRING",
                    {
                        "default": "Qwen/Qwen2.5-VL-3B-Instruct",
                        "multiline": False,
                        "tooltip": (
                            "可直接粘贴 Hugging Face 模型 ID，也可填本地路径。"
                            "推荐 Qwen2.5-VL；其他 HF 视觉模型会按通用 chat template 尝试加载。"
                        ),
                    },
                ),
                "视觉投影mmproj": (
                    _列出mmproj候选(),
                    {
                        "default": HF后端无mmproj,
                        "tooltip": "HF/Transformers 后端不需要 llama.cpp 的 mmproj，保留该参数仅为兼容旧工作流。",
                    },
                ),
                "启用思考": ("BOOLEAN", {"default": False, "tooltip": "若模型模板支持 enable_thinking，会传给模板；否则自动忽略。"}),
                "上下文长度": ("INT", {"default": 8192, "min": 1024, "max": 327680, "step": 256, "tooltip": "保留以兼容原节点；Transformers 会根据模型配置处理上下文。"}),
                "GPU层数": ("INT", {"default": -1, "min": -1, "max": 9999, "step": 1, "tooltip": "保留以兼容原节点；HF 后端会自动使用 CUDA/MPS/CPU。"}),
                "KV缓存K类型": (KV缓存类型选项, {"default": 默认KV缓存类型, "tooltip": "llama.cpp 参数，HF 后端会忽略。"}),
                "KV缓存V类型": (KV缓存类型选项, {"default": 默认KV缓存类型, "tooltip": "llama.cpp 参数，HF 后端会忽略。"}),
            }
        }

    RETURN_TYPES = ("GEMMA4LLAMA",)
    RETURN_NAMES = ("gemma4模型",)
    FUNCTION = "load"
    CATEGORY = "Gemma4 TE"

    def load(
        self,
        主模型,
        视觉投影mmproj,
        启用思考,
        上下文长度,
        GPU层数,
        KV缓存K类型,
        KV缓存V类型,
    ):
        config = {
            "backend": "transformers",
            "family": "Gemma4-HF",
            "model": 主模型,
            "mmproj": 视觉投影mmproj,
            "think": bool(启用思考),
            "n_ctx": int(上下文长度),
            "n_gpu_layers": int(GPU层数),
            "cache_type_k": KV缓存K类型,
            "cache_type_v": KV缓存V类型,
        }
        return (_Gemma4HFStorage.load(config),)


class Gemma4TE图像推理(_HF图像推理基类):
    模型输入名 = "gemma4模型"
    模型类型 = "GEMMA4LLAMA"
    CATEGORY = "Gemma4 TE"

    @staticmethod
    def _storage():
        return _Gemma4HFStorage


class Gemma4TE卸载模型:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"任意输入": (any_type,)}}

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("任意输出",)
    FUNCTION = "run"
    CATEGORY = "Gemma4 TE"

    def run(self, 任意输入):
        _Gemma4HFStorage.unload()
        return (任意输入,)


class Gemma4TE音频推理:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "gemma4模型": ("GEMMA4LLAMA",),
                "提示": ("STRING", {"default": "当前 HF 替代节点不支持音频推理。", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("文本",)
    FUNCTION = "run"
    CATEGORY = "Gemma4 TE"

    def run(self, gemma4模型, 提示):
        del gemma4模型, 提示
        raise NotImplementedError(
            "Gemma4 音频推理依赖原 llama.cpp/Gemma4ChatHandler 的多模态音频输入，当前替代版本暂未实现。"
            "当前版本已实现图片、逐帧、视频抽帧和文本推理。"
        )
