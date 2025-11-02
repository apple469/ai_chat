import onnxruntime
import numpy as np
from tokenizers import BertWordPieceTokenizer
import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import json


# --- 配置 ---
# 获取脚本所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# ONNX模型和tokenizer文件所在的目录
ONNX_MODEL_DIR = os.path.join(SCRIPT_DIR, "onnx_models")
ONNX_MODEL_PATH = os.path.join(ONNX_MODEL_DIR, "model.onnx")
TOKENIZER_VOCAB_PATH = os.path.join(ONNX_MODEL_DIR, "vocab.txt")


class ONNXSentenceEncoder:
    """ONNX模型和tokenizer封装类"""
    def __init__(self, model_path, vocab_path, max_seq_len=128):
        """
        初始化ONNX模型和tokenizer
        :param model_path: ONNX模型文件路径
        :param vocab_path: 词汇表文件路径
        :param max_seq_len: 最大序列长度
        """
        self.session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.tokenizer = BertWordPieceTokenizer(
            vocab=vocab_path,
            lowercase=True
        )
        self.max_seq_len = max_seq_len
        self.input_names = [inp.name for inp in self.session.get_inputs()]

    def _tokenize(self, text):
        """对文本进行tokenization"""
        encoded = self.tokenizer.encode(text)
        input_ids = encoded.ids
        attention_mask = encoded.attention_mask
        token_type_ids = encoded.type_ids

        # 裁剪或填充到 max_seq_len
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            attention_mask = attention_mask[:self.max_seq_len]
            token_type_ids = token_type_ids[:self.max_seq_len]
        else:
            padding_len = self.max_seq_len - len(input_ids)
            input_ids = input_ids + [0] * padding_len
            attention_mask = attention_mask + [0] * padding_len
            token_type_ids = token_type_ids + [0] * padding_len

        return np.array([input_ids], dtype=np.int64), \
               np.array([attention_mask], dtype=np.int64), \
               np.array([token_type_ids], dtype=np.int64)

    def encode(self, texts):
        """将文本列表编码为嵌入向量"""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            input_ids, attention_mask, token_type_ids = self._tokenize(text)
            
            # 准备输入字典，确保与ONNX模型输入名称匹配
            inputs = {}
            if "input_ids" in self.input_names:
                inputs["input_ids"] = input_ids
            if "attention_mask" in self.input_names:
                inputs["attention_mask"] = attention_mask
            if "token_type_ids" in self.input_names:
                 inputs["token_type_ids"] = token_type_ids

            outputs = self.session.run(None, inputs)
            
            # Sentence-BERT通常输出last_hidden_state，然后池化
            last_hidden_state = outputs[0]
            
            # 平均池化，并考虑到attention_mask
            input_mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(float)
            sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
            sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
            sentence_embedding = sum_embeddings / sum_mask
            
            # 归一化嵌入向量 (L2范数)
            sentence_embedding = sentence_embedding / np.linalg.norm(sentence_embedding, axis=1, keepdims=True)
            embeddings.append(sentence_embedding[0])
        
        return np.array(embeddings)


# --- 核心优化：全局初始化模型（仅加载一次）---
# 程序启动时就完成模型和Tokenizer的加载，后续分类直接复用
try:
    global_encoder = ONNXSentenceEncoder(ONNX_MODEL_PATH, TOKENIZER_VOCAB_PATH)
    print("分类模型初始化成功")
except Exception as e:
    print(f"模型初始化失败：{str(e)}")
    print("请检查 onnx_models 目录下是否存在 model.onnx 和 vocab.txt 文件")
    exit(1)


def cosine_similarity(vec1, vec2):
    """
    计算两个向量之间的余弦相似度
    :param vec1: 向量1
    :param vec2: 向量2
    :return: 余弦相似度
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# 添加缓存来存储原型句子的嵌入向量
from collections import OrderedDict
# 使用OrderedDict来维护缓存顺序，并限制最大大小为100
prototype_embeddings_cache = OrderedDict()
MAX_CACHE_SIZE = 100

def classify_text(text, prototypes):
    """
    根据原型句子对文本进行分类（复用全局模型实例）
    :param text: 需要分类的文本
    :param prototypes: 原型句子列表，格式为 [{"text": "示例文本", "label": "标签"}, ...]
    :return: (最优标签, 置信度)
    """
    # 直接使用全局初始化好的 encoder，不再重复创建
    encoder = global_encoder
    
    # 编码待分类文本
    text_embedding = encoder.encode(text)[0]
    
    # 按标签分组原型句子
    label_groups = {}
    for prototype in prototypes:
        label = prototype["label"]
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(prototype["text"])
    
    # 计算每个标签组的平均相似度
    label_similarities = {}
    for label, texts in label_groups.items():
        # 使用缓存避免重复计算原型句子的嵌入向量
        cache_key = tuple(texts)  # 使用文本元组作为缓存键
        if cache_key in prototype_embeddings_cache:
            # 当访问已存在的缓存项时，将其移到最后（最近使用）
            embeddings = prototype_embeddings_cache.pop(cache_key)
            prototype_embeddings_cache[cache_key] = embeddings
        else:
            embeddings = encoder.encode(texts)
            # 检查缓存大小，如果超过限制则删除最早添加的条目
            if len(prototype_embeddings_cache) >= MAX_CACHE_SIZE:
                prototype_embeddings_cache.popitem(last=False)
            prototype_embeddings_cache[cache_key] = embeddings  # 缓存结果
            
        similarities = [cosine_similarity(text_embedding, emb) for emb in embeddings]
        label_similarities[label] = np.mean(similarities)
    
    # 返回具有最高平均相似度的标签和置信度
    if not label_similarities:
        return None, 0.0
    
    best_label = max(label_similarities, key=label_similarities.get)
    confidence = label_similarities[best_label]
    
    return best_label, confidence


# 定义请求和响应模型
class ClassifyRequest(BaseModel):
    text: str

class ClassifyResponse(BaseModel):
    label: str
    confidence: float

class AIResponseRequest(BaseModel):
    apiurl : str
    apikey: str
    model: str
    thinking_budget: int
    query: str

class AIResponseResponse(BaseModel):
    think_text: str
    text: str

class API_action(BaseModel):
    provider: str | None = None # 模型提供商，如 "openai"
    apiurl : str | None = None
    apikey: str | None = None
    model_name: str | None = None
    action: str # 只允许传入 "add" 或 "delete" 或 "query"
    name: str | None = None # 模型别名，用于添加和删除

# 创建FastAPI应用实例
app = FastAPI(title="AI Text Classification API", version="1.0.0")

import json
try:
    with open("api_config.json", "r") as f:
        api_config = json.load(f)
except Exception as e:
    print(f"Failed to read api_config.json: {e}")
    api_config = []

# 分类标签定义
requires_reasoning = [
    {"text": "please thinking", "label": "thinking"},
    {"text": "please non-thinking", "label": "non-thinking"},
    {"text": "请思考", "label": "thinking"},
    {"text": "不要思考", "label": "non-thinking"},
    {"text": "这个问题很复杂", "label": "thinking"},
    {"text": "这个问题很简单", "label": "non-thinking"},
    {"text": "深入分析一下", "label": "thinking"},
    {"text": "请直接回答", "label": "non-thinking"},
    {"text": "验证", "label": "thinking"},
    {"text": "事实", "label": "non-thinking"},
    {"text": "你好", "label": "thinking"},
    {"text": "今天天气怎么样", "label": "non-thinking"}
]

@app.post("/admin/config")
async def config_endpoint(request: API_action):
    global api_config
    try:
        if request.action == "add":
            if not request.apiurl or not request.provider or not request.model_name:
                raise HTTPException(status_code=400, detail="apiurl, apikey, and model_name are required for add action")
            else:
                if not any(item["name"] == request.name for item in api_config):
                    # 创建新配置对象，只包含非空的字段
                    new_config = {}
                    if request.provider is not None:
                        new_config["provider"] = request.provider
                    if request.apiurl is not None:
                        new_config["apiurl"] = request.apiurl
                    if request.apikey is not None:
                        new_config["apikey"] = request.apikey
                    if request.model_name is not None:
                        new_config["model_name"] = request.model_name
                    if request.name is not None:
                        new_config["name"] = request.name
                    
                    # 将新配置追加到全局 api_config 列表
                    api_config.append(new_config)
                    with open("api_config.json", "w", encoding="utf-8") as f:
                        json.dump(api_config, f, ensure_ascii=False, indent=2)
                else:
                    raise HTTPException(status_code=400, detail="Duplicate name!")
        elif request.action == "delete":
            if not request.name:
                raise HTTPException(status_code=400, detail="name is required for delete action")
            else:
                if any(item["name"] == request.name for item in api_config):
                    api_config = [item for item in api_config if item["name"] != request.name]
                    with open("api_config.json", "w", encoding="utf-8") as f:
                        json.dump(api_config, f, ensure_ascii=False, indent=2)
                else:
                    raise HTTPException(status_code=400, detail="name not found!")
        elif request.action == "query":
            cleaned_config = [{k: v for k, v in item.items() if k != "apikey"} for item in api_config]
            return cleaned_config
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"error: {str(e)}")

@app.post("/router", response_model=ClassifyResponse)
async def classify_text_endpoint(request: ClassifyRequest):
    """
    对输入文本进行分类
    """
    try:
        label, confidence = classify_text(request.text, requires_reasoning)
        return ClassifyResponse(label=label or "non-thinking", confidence=float(confidence))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"error: {str(e)}")

@app.post("/ai", response_model=AIResponseResponse)
async def ai_response_endpoint(request: AIResponseRequest):
    """
    对输入文本进行AI响应
    """
    try:
        from google import genai
        from google.genai import types
        

        client = genai.Client(
            api_key=request.apikey
        )

        response = client.models.generate_content(
        model=request.model,
        contents=request.query,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=request.thinking_budget,
                include_thoughts=True
            )
        )
        )

        # 初始化思考内容和最终回复内容为空字符串
        think_text = ""
        text = ""
        # 如果响应中包含候选结果，则遍历处理
        if response.candidates:
            for candidate in response.candidates:
                # 安全获取候选结果中的 parts，避免空指针
                parts = candidate.content.parts if candidate.content and candidate.content.parts else []
                for part in parts:
                    # 判断当前 part 是否为思考内容（thought 字段存在且为真）
                    if hasattr(part, "thought") and part.thought:
                        # 如果是思考内容，提取其 text 字段作为思考文本
                        think_text = part.text if hasattr(part, "text") else ""
                    else:
                        # 否则视为最终回复内容，提取其 text 字段
                        text = part.text if hasattr(part, "text") else ""
        # 构造并返回响应对象，确保字段不为 None
        return AIResponseResponse(think_text=think_text or "", text=text or "")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"error: {str(e)}")



# 挂载静态文件服务，将index.html作为首页
# 注意：必须在所有API路由定义之后挂载静态文件服务
app.mount("/", StaticFiles(directory=".", html=True), name="static")

# 如果直接运行此脚本，则启动服务器
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)