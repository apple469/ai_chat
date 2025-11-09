import onnxruntime
import numpy as np
from tokenizers import BertWordPieceTokenizer
import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import json


# --- Configuration ---
# Get the absolute path of the script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Directory for ONNX model and tokenizer files
ONNX_MODEL_DIR = os.path.join(SCRIPT_DIR, "onnx_models")
ONNX_MODEL_PATH = os.path.join(ONNX_MODEL_DIR, "model.onnx")
TOKENIZER_VOCAB_PATH = os.path.join(ONNX_MODEL_DIR, "vocab.txt")


class ONNXSentenceEncoder:
    """ONNX Model and Tokenizer Wrapper Class"""
    def __init__(self, model_path, vocab_path, max_seq_len=128):
        """
        Initializes ONNX model and tokenizer
        :param model_path: Path to the ONNX model file
        :param vocab_path: Path to the vocabulary file
        :param max_seq_len: Maximum sequence length
        """
        self.session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.tokenizer = BertWordPieceTokenizer(
            vocab=vocab_path,
            lowercase=True
        )
        self.max_seq_len = max_seq_len
        self.input_names = [inp.name for inp in self.session.get_inputs()]

    def _tokenize(self, text):
        """Tokenizes the input text"""
        encoded = self.tokenizer.encode(text)
        input_ids = encoded.ids
        attention_mask = encoded.attention_mask
        token_type_ids = encoded.type_ids

        # Truncate or pad to max_seq_len
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
        """Encodes a list of texts into embedding vectors"""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            input_ids, attention_mask, token_type_ids = self._tokenize(text)
            
            # Prepare input dictionary, ensuring it matches ONNX model input names
            inputs = {}
            if "input_ids" in self.input_names:
                inputs["input_ids"] = input_ids
            if "attention_mask" in self.input_names:
                inputs["attention_mask"] = attention_mask
            if "token_type_ids" in self.input_names:
                 inputs["token_type_ids"] = token_type_ids

            outputs = self.session.run(None, inputs)
            
            # Sentence-BERT typically outputs last_hidden_state, then pools
            last_hidden_state = outputs[0]
            
            # Average pooling, considering attention_mask
            input_mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(float)
            sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
            sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
            sentence_embedding = sum_embeddings / sum_mask
            
            # Normalize embedding vectors (L2 norm)
            sentence_embedding = sentence_embedding / np.linalg.norm(sentence_embedding, axis=1, keepdims=True)
            embeddings.append(sentence_embedding[0])
        
        return np.array(embeddings)


# --- Core Optimization: Global model initialization (loaded only once) ---
# Model and Tokenizer are loaded at program startup, subsequent classifications reuse them
try:
    global_encoder = ONNXSentenceEncoder(ONNX_MODEL_PATH, TOKENIZER_VOCAB_PATH)
    print("Classification model initialized successfully")
except Exception as e:
    print(f"Model initialization failed: {str(e)}")
    print("Please check if model.onnx and vocab.txt exist in the onnx_models directory")
    exit(1)


def cosine_similarity(vec1, vec2):
    """
    Calculates the cosine similarity between two vectors
    :param vec1: Vector 1
    :param vec2: Vector 2
    :return: Cosine similarity
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# Add cache to store prototype sentence embeddings
from collections import OrderedDict
# Use OrderedDict to maintain cache order and limit max size to 100
prototype_embeddings_cache = OrderedDict()
MAX_CACHE_SIZE = 100

def classify_text(text, prototypes):
    """
    Classifies text based on prototype sentences (reusing global model instance)
    :param text: Text to be classified
    :param prototypes: List of prototype sentences, format: [{"text": "sample text", "label": "label"}, ...]
    :return: (best label, confidence)
    """
    # Directly use the globally initialized encoder, no need to recreate
    encoder = global_encoder
    
    # Encode text to be classified
    text_embedding = encoder.encode(text)[0]
    
    # Group prototype sentences by label
    label_groups = {}
    for prototype in prototypes:
        label = prototype["label"]
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(prototype["text"])
    
    # Calculate average similarity for each label group
    label_similarities = {}
    for label, texts in label_groups.items():
        # Use cache to avoid recomputing prototype sentence embeddings
        cache_key = tuple(texts)  # Use text tuple as cache key
        if cache_key in prototype_embeddings_cache:
            # When accessing an existing cache item, move it to the end (most recently used)
            embeddings = prototype_embeddings_cache.pop(cache_key)
            prototype_embeddings_cache[cache_key] = embeddings
        else:
            embeddings = encoder.encode(texts)
            # Check cache size, remove oldest entry if limit exceeded
            if len(prototype_embeddings_cache) >= MAX_CACHE_SIZE:
                prototype_embeddings_cache.popitem(last=False)
            prototype_embeddings_cache[cache_key] = embeddings  # Cache result
            
        similarities = [cosine_similarity(text_embedding, emb) for emb in embeddings]
        label_similarities[label] = np.mean(similarities)
    
    # Return the label with the highest average similarity and confidence
    if not label_similarities:
        return None, 0.0
    
    best_label = max(label_similarities, key=label_similarities.get)
    confidence = label_similarities[best_label]
    
    return best_label, confidence


# Define request and response models
class ClassifyRequest(BaseModel):
    text: str

class ClassifyResponse(BaseModel):
    label: str
    confidence: float

class AIResponseRequest(BaseModel):
    name: str
    thinking_budget: int
    query: str
    # ...

class AIResponseResponse(BaseModel):
    think_text: str
    text: str

class API_action(BaseModel):
    provider: str | None = None # Model provider, e.g., "openai"
    apiurl : str | None = None
    apikey: str | None = None
    model_name: str | None = None
    action: str # Only "add", "delete", or "query" allowed
    name: str | None = None # Model alias, used for adding and deleting

class APIConfigResponse(BaseModel):
    success: bool
    message: str
    data: list | dict | None = None

# Create FastAPI application instance
app = FastAPI(title="AI Text Classification API", version="1.0.0")

import json
try:
    with open("api_config.json", "r") as f:
        api_config = json.load(f)
except Exception as e:
    print(f"Failed to read api_config.json: {e}")
    api_config = []
    with open("api_config.json", "w", encoding="utf-8") as f:
        json.dump(api_config, f, ensure_ascii=False, indent=2)

# Classification label definitions
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

@app.post("/admin/config", response_model=APIConfigResponse)
async def config_endpoint(request: API_action):
    global api_config
    try:
        if request.action == "add":
            if not request.apiurl or not request.provider or not request.model_name or not request.name:
                raise HTTPException(status_code=400, detail="apiurl, provider, model_name, and name are required for add action")
            else:
                # Create new configuration object, only including non-empty fields
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
                
                # Append new configuration to the global api_config list
                api_config.append(new_config)
                with open("api_config.json", "w", encoding="utf-8") as f:
                    json.dump(api_config, f, ensure_ascii=False, indent=2)
                return APIConfigResponse(success=True, message="Configuration added successfully")
        elif request.action == "delete":
            if not request.name:
                raise HTTPException(status_code=400, detail="name is required for delete action")
            else:
                if any(item["name"] == request.name for item in api_config):
                    api_config = [item for item in api_config if item["name"] != request.name]
                    with open("api_config.json", "w", encoding="utf-8") as f:
                        json.dump(api_config, f, ensure_ascii=False, indent=2)
                    return APIConfigResponse(success=True, message="Configuration deleted successfully")
                else:
                    raise HTTPException(status_code=400, detail="name not found!")
        elif request.action == "query":
            cleaned_config = [{k: v for k, v in item.items() if k != "apikey"} for item in api_config]
            return APIConfigResponse(success=True, message="Query successful", data=cleaned_config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"error: {str(e)}")

@app.post("/router", response_model=ClassifyResponse)
async def classify_text_endpoint(request: ClassifyRequest):
    """
    Classifies the input text
    """
    try:
        label, confidence = classify_text(request.text, requires_reasoning)
        return ClassifyResponse(label=label or "non-thinking", confidence=float(confidence))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"error: {str(e)}")

@app.post("/ai", response_model=AIResponseResponse)
async def ai_response_endpoint(request: AIResponseRequest):
    """
    Provides AI response for the input text
    """

    # Google Gemini model
    try:
        from google import genai
        from google.genai import types
        
        apikey = next((item["apikey"] for item in api_config if item.get("name") == request.name), None)
        model_name = next((item["model_name"] for item in api_config if item.get("name") == request.name), None)
        if apikey is None:
            raise HTTPException(status_code=404, detail=f"Model configuration named '{request.name}' not found")

        client = genai.Client(
            api_key=apikey
        )

        response = client.models.generate_content(
        model=model_name,
        contents=request.query,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=request.thinking_budget,
                include_thoughts=True
            )
        )
        )

        # Initialize think_text and text as empty strings
        think_text = ""
        text = ""
        # If response contains candidates, iterate and process
        if response.candidates:
            for candidate in response.candidates:
                # Safely get parts from candidate content, avoid null pointer
                parts = candidate.content.parts if candidate.content and candidate.content.parts else []
                for part in parts:
                    # Check if current part is thought content (thought field exists and is true)
                    if hasattr(part, "thought") and part.thought:
                        # If it's thought content, extract its text field as think_text
                        think_text = part.text if hasattr(part, "text") else ""
                    else:
                        # Otherwise, treat as final response content, extract its text field
                        text = part.text if hasattr(part, "text") else ""
        # Construct and return response object, ensuring fields are not None
        return AIResponseResponse(think_text=think_text or "", text=text or "")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"error: {str(e)}")



# Mount static file service, using index.html as the homepage
# Note: Static file service must be mounted after all API route definitions
app.mount("/", StaticFiles(directory=".", html=True), name="static")

# If this script is run directly, start the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)