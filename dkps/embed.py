import os
import asyncio
import numpy as np
from tqdm import trange
from tqdm.asyncio import tqdm
import httpx

try:
    from google import genai
    from google.genai import types
    from google.genai.types import HttpOptions
except:
    print('dkps.embed: unable to load google-genai')

try:
    from jlai.embed.tei import embed_dataset as embed_jlai_tei
except:
    print('dkps.embed: unable to load jlai')

try:
    from openai import OpenAI
except:
    print('dkps.embed: unable to load openai')

try:
    from litellm import embedding as litellm_embedding
except:
    print('dkps.embed: unable to load litellm')

try:
    from huggingface_hub import InferenceClient as HFInferenceClient
except:
    print('dkps.embed: unable to load huggingface_hub')

try:
    from sentence_transformers import SentenceTransformer
except:
    print('dkps.embed: unable to load sentence_transformers')

from .cache import disk_cache

# --
# Provider-specific chunk functions

# Google
@disk_cache(cache_dir='./.cache/embed/google', verbose=False, ignore_fields=['client'])
async def _aembed_google_chunk(chunk_id, client, chunk, model):
    # retry with backoff on 429 RESOURCE_EXHAUSTED (per-minute content quota)
    for attempt in range(10):
        try:
            chunk_response = await client.aio.models.embed_content(
                model    = model,
                contents = chunk,
            )
            return chunk_id, np.array([xx.values for xx in chunk_response.embeddings])
        except Exception as e:
            if '429' in str(e) and attempt < 9:
                await asyncio.sleep(20 + 5 * attempt + (chunk_id % 7))
            else:
                raise

# Jina
class JinaClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("JINA_API_KEY")
    
    async def embed(self, input_data, model):
        if not self.api_key:
            raise Exception("JINA_API_KEY is not set")
        
        headers = {
            "Accept"          : "application/json",
            "Authorization"   : f"Bearer {self.api_key}",
            "Content-Type"    : "application/json",
        }

        async with httpx.AsyncClient(timeout=None, headers=headers) as client:
            res = await client.post("https://api.jina.ai/v1/embeddings", json={
                "model" : model,
                "task"  : "text-matching",
                "input" : input_data,
            })
            res.raise_for_status()
            return res.json()

@disk_cache(cache_dir="./.cache/embed/jina", verbose=False, ignore_fields=['client'])
async def _aembed_jina_chunk(chunk_id, client, chunk, model):
    data = await client.embed(chunk, model)
    return chunk_id, np.array([item['embedding'] for item in data['data']])

# OpenRouter
class OpenRouterClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

    def embed(self, input_data, model):
        if not self.api_key:
            raise Exception("OPENROUTER_API_KEY is not set")

        response = self.client.embeddings.create(
            extra_headers={
                "HTTP-Referer": "https://github.com/dkps",
                "X-Title": "dkps",
            },
            model=model,
            input=input_data,
            encoding_format="float",
        )
        return response

@disk_cache(cache_dir="./.cache/embed/openrouter", verbose=False, ignore_fields=['client'])
async def _aembed_openrouter_chunk(chunk_id, client, chunk, model):
    # OpenRouter uses sync OpenAI client, run in executor
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, lambda: client.embed(chunk, model))
    return chunk_id, np.array([item.embedding for item in response.data])

# LiteLLM
class LiteLLMClient:
    def __init__(self):
        pass

    def embed(self, input_data, model):
        response = litellm_embedding(
            model=model,
            input=input_data,
        )
        return response

@disk_cache(cache_dir="./.cache/embed/litellm", verbose=False, ignore_fields=['client'])
async def _aembed_litellm_chunk(chunk_id, client, chunk, model):
    # LiteLLM is sync, run in executor
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, lambda: client.embed(chunk, model))
    return chunk_id, np.array([item['embedding'] for item in response.data])

# Hugging Face
class HuggingFaceClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("HF_TOKEN")
        self.client = HFInferenceClient(
            provider="hf-inference",
            api_key=self.api_key,
        )

    def embed(self, input_data, model):
        if not self.api_key:
            raise Exception("HF_TOKEN is not set")

        result     = self.client.feature_extraction(input_data[0], model=model)
        embeddings = [np.array(x) for x in result]
        return np.array(embeddings)

@disk_cache(cache_dir="./.cache/embed/huggingface", verbose=False, ignore_fields=['client'])
async def _aembed_huggingface_chunk(chunk_id, client, chunk, model):
    # HuggingFace client is sync, run in executor
    loop = asyncio.get_event_loop()
    embeddings = await loop.run_in_executor(None, lambda: client.embed(chunk, model))
    return chunk_id, np.array(embeddings)

# Sentence Transformers (local)
class SentenceTransformersClient:
    def __init__(self, model=None):
        self.model_name = model
        self._model = None

    def _get_model(self, model):
        # Lazy load model, cache it if same model
        if self._model is None or self.model_name != model:
            self.model_name = model
            self._model = SentenceTransformer(model, trust_remote_code=True)
        return self._model

    def embed(self, input_data, model):
        st_model = self._get_model(model)
        embeddings = st_model.encode(input_data, convert_to_numpy=True)
        return embeddings

@disk_cache(cache_dir="./.cache/embed/sentence_transformers", verbose=False, ignore_fields=['client'])
async def _aembed_sentence_transformers_chunk(chunk_id, client, chunk, model):
    # Sentence Transformers is sync, run in executor
    loop = asyncio.get_event_loop()
    embeddings = await loop.run_in_executor(None, lambda: client.embed(chunk, model))
    return chunk_id, np.array(embeddings)


# Generic API embedding functions
async def _aembed_api(provider, input_strs, chunk_size=50, max_concurrency=5, model=None):
    """Generic async embedding function that handles chunking and concurrency"""
    assert isinstance(input_strs, list), 'input_strs must be a list'
    
    if provider == 'google':
        client = genai.Client(
            api_key=os.getenv('GEMINI_API_KEY'),
            http_options=HttpOptions(
                api_version='v1beta',
                timeout=10 * 1000, # 10 seconds
            ),
        )
        _aembed_chunk = _aembed_google_chunk
        if model is None:
            model = 'gemini-embedding-001'
        
    elif provider == 'jina':
        client = JinaClient()
        _aembed_chunk = _aembed_jina_chunk
        if model is None:
            model = 'jina-embeddings-v3'

    elif provider == 'openrouter':
        client = OpenRouterClient()
        _aembed_chunk = _aembed_openrouter_chunk
        if model is None:
            model = 'sentence-transformers/all-minilm-l6-v2'

    elif provider == 'litellm':
        client = LiteLLMClient()
        _aembed_chunk = _aembed_litellm_chunk
        if model is None:
            model = 'text-embedding-3-small'

    elif provider == 'huggingface':
        client = HuggingFaceClient()
        _aembed_chunk = _aembed_huggingface_chunk
        if model is None:
            model = 'intfloat/multilingual-e5-large'

    elif provider == 'sentence-transformers':
        client = SentenceTransformersClient()
        _aembed_chunk = _aembed_sentence_transformers_chunk
        if model is None:
            model = 'all-MiniLM-L6-v2'

    else:
        raise Exception(f"Unknown provider: {provider}")


    sem = asyncio.Semaphore(max_concurrency)
    async def _fn(chunk_id, chunk):
        async with sem:
            return await _aembed_chunk(chunk_id, client, chunk, model)
    
    chunks = [input_strs[i:i+chunk_size] for i in range(0, len(input_strs), chunk_size)]
    
    tasks = []
    for chunk_id, chunk in enumerate(chunks):
        task = _fn(chunk_id, chunk)
        tasks.append(task)
    
    out = [None] * len(chunks)
    for task in tqdm(asyncio.as_completed(tasks), desc="Embedding chunks", total=len(tasks)):
        chunk_id, embedding = await task
        out[chunk_id] = embedding
    
    return np.concatenate(out)

# jlai
@disk_cache(cache_dir="./.cache/embed/jlai_tei", verbose=False, ignore_fields=['client'])
def _embed_jlai_tei(input_strs, model='nomic-ai/nomic-embed-text-v1', **kwargs):
    assert isinstance(input_strs, list), 'input_strs must be a list'
    
    return embed_jlai_tei(input_strs, model_id=model)


# Synchronous wrapper functions
def embed_api(provider, *args, **kwargs):
    if provider == 'jlai_tei':
        return _embed_jlai_tei(*args, **kwargs)
    else:
        return asyncio.run(_aembed_api(provider, *args, **kwargs))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test embedding APIs")
    parser.add_argument("--provider", type=str, default="openrouter",
                        choices=["google", "jina", "openrouter", "litellm", "huggingface", "sentence-transformers", "jlai_tei"],
                        help="Embedding provider to use")
    parser.add_argument("--model", type=str, default=None,
                        help="Model to use (provider-specific)")
    parser.add_argument("--text", type=str, nargs="+",
                        default=["Hello, world!", "This is a test."],
                        help="Text(s) to embed")
    args = parser.parse_args()

    print(f"Provider: {args.provider}")
    print(f"Model: {args.model or '(default)'}")
    print(f"Input texts: {args.text}")
    print()

    kwargs = {}
    if args.model:
        kwargs["model"] = args.model

    embeddings = embed_api(args.provider, args.text, **kwargs)

    print(f"Output shape: {embeddings.shape}")
    print(f"First embedding (first 10 dims): {embeddings[0][:10]}")

    if len(args.text) > 1:
        # Compute cosine similarity between first two embeddings
        sim = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        print(f"Cosine similarity between first two texts: {sim:.4f}")
