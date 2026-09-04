"""qubric: query-specific rubric representations for agentic traces.

The three-stage pipeline as a reusable API:

  1. write_rubrics(tasks, api_key, model_name)        -> rubric per task
  2. grade_traces(rubrics, traces, api_key, model_name) -> graded trace(s)
  3. embed_graded(graded, api_key, embedding_model_name) -> (n, k*d) array

plus consensus_center() for the per-instance centering used everywhere.

Chat models are addressed through any OpenAI-compatible endpoint (default
OpenRouter). Embedding models: a sentence-transformers name runs locally
(api_key ignored); anything else goes to an OpenAI-compatible /embeddings
endpoint with the given key.
"""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

DEFAULT_SECTIONS = ('understanding', 'localization', 'reproduction',
                    'editing', 'verification', 'final_state')

RUBRIC_PROMPT = """Here is a repository issue that a software-engineering agent will be asked to fix:

{problem}

Write an instance-specific rubric for judging an agent's trace on THIS issue:
for each section below, one sentence saying what specifically to look for in
this instance (name the likely relevant modules/files/tests if inferable).

Sections: {sections}.
Respond with ONLY a JSON object with those keys, each a one-sentence string."""

GRADE_PROMPT = """Below is the (truncated) execution trace of a software-engineering agent, and an
instance-specific rubric. For EACH rubric section, write 30-60 words factually
describing what the agent actually did with respect to that section's
criteria. Describe behavior; do not score it and do not speculate.

Rubric:
{rubric}

Respond with ONLY a JSON object with the keys {keys}, each a string.

=== TRACE ===
{trace}"""

DEFAULT_BASE_URL = 'https://openrouter.ai/api/v1'


def _schema(sections):
    return {'type': 'json_schema',
            'json_schema': {'name': 'sections', 'strict': True, 'schema': {
                'type': 'object',
                'properties': {k: {'type': 'string'} for k in sections},
                'required': list(sections),
                'additionalProperties': False}}}


def _chat(api_key, model_name, content, sections, base_url=DEFAULT_BASE_URL,
          max_tokens=4000, effort='none'):
    reasoning = {'enabled': False} if effort == 'none' else {'effort': effort}
    body = {'model': model_name,
            'messages': [{'role': 'user', 'content': content}],
            'max_tokens': max_tokens,
            'response_format': _schema(sections),
            'reasoning': reasoning,
            'provider': {'sort': 'throughput', 'require_parameters': True}}
    if 'openrouter' not in base_url:
        body.pop('reasoning'); body.pop('provider')
    delay = 2.0
    for _ in range(10):
        try:
            r = requests.post(base_url.rstrip('/') + '/chat/completions', json=body,
                              headers={'Authorization': f'Bearer {api_key}'},
                              timeout=(15, 120))
        except requests.RequestException:
            time.sleep(delay); delay = min(delay * 2, 60); continue
        if r.status_code == 200:
            try:
                j = r.json()
            except ValueError:
                time.sleep(delay); delay = min(delay * 2, 60); continue
            if 'choices' not in j:
                time.sleep(delay); delay = min(delay * 2, 60); continue
            if j['choices'][0].get('finish_reason') == 'length':
                body['max_tokens'] = 8000
                continue
            out = j['choices'][0]['message']['content'] or ''
            try:
                d = json.loads(out)
                if isinstance(d, list) and d:
                    d = d[0]
                if isinstance(d, dict) and any(d.get(k) for k in sections):
                    return d
            except json.JSONDecodeError:
                pass
            # empty/invalid content: escalate once, then retry
            body['max_tokens'] = 8000
            body['reasoning'] = {'effort': 'medium'}
            time.sleep(delay); delay = min(delay * 2, 60); continue
        if r.status_code in (429, 500, 502, 503, 529):
            time.sleep(max(delay, float(r.headers.get('retry-after', 0) or 0)))
            delay = min(delay * 2, 60)
            continue
        raise RuntimeError(f'{r.status_code}: {r.text[:200]}')
    raise RuntimeError('chat retries exhausted')


def _as_dict(x, prefix):
    if isinstance(x, dict):
        return dict(x)
    if isinstance(x, str):
        return {f'{prefix}0': x}
    return {f'{prefix}{i}': t for i, t in enumerate(x)}


def write_rubrics(tasks, api_key, model_name, sections=DEFAULT_SECTIONS,
                  base_url=DEFAULT_BASE_URL, workers=8):
    """tasks: str | list[str] | dict[id -> problem description].
    Returns dict[id -> rubric dict] (str/list inputs get ids 'task0', ...)."""
    tasks = _as_dict(tasks, 'task')
    out = {}
    lock = threading.Lock()

    def one(item):
        tid, problem = item
        rub = _chat(api_key, model_name,
                    RUBRIC_PROMPT.format(problem=problem[:20_000],
                                         sections=', '.join(sections)),
                    sections, base_url, max_tokens=1500)
        with lock:
            out[tid] = rub

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, tasks.items()))
    return out


def grade_traces(rubrics, traces, api_key, model_name,
                 task_ids=None, sections=DEFAULT_SECTIONS,
                 base_url=DEFAULT_BASE_URL, workers=16,
                 max_trace_chars=60_000, on_error='raise'):
    """rubrics: a single rubric dict, or dict[task_id -> rubric].
    traces: str | list[str] | dict[key -> trace text].
    task_ids: when rubrics is per-task, aligns each trace to its task
    (same length/keys as traces; unnecessary for a single rubric).
    Returns graded traces with the same keying as `traces`."""
    single = all(isinstance(v, str) for v in rubrics.values())
    traces = _as_dict(traces, 'trace')
    keys = list(traces)
    if single:
        rub_for = {k: rubrics for k in keys}
    else:
        if task_ids is None:
            raise ValueError('task_ids required when rubrics is per-task')
        tid = (task_ids if isinstance(task_ids, dict)
               else dict(zip(keys, task_ids)))
        rub_for = {k: rubrics[tid[k]] for k in keys}
    out = {}
    lock = threading.Lock()

    def one(k):
        t = traces[k]
        if len(t) > max_trace_chars:
            t = t[:max_trace_chars * 2 // 3] + '\n...[omitted]...\n' \
                + t[-max_trace_chars // 3:]
        rub = '\n'.join(f'- {s}: {rub_for[k].get(s, "")}' for s in sections)
        try:
            g = _chat(api_key, model_name,
                      GRADE_PROMPT.format(rubric=rub, keys=list(sections), trace=t),
                      sections, base_url)
        except RuntimeError:
            if on_error == 'raise':
                raise
            return                      # on_error='skip': caller retries later
        with lock:
            out[k] = g

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, keys))
    return out


def embed_graded(graded, api_key, embedding_model_name,
                 sections=DEFAULT_SECTIONS, base_url='https://api.openai.com/v1',
                 batch=256):
    """graded: one graded trace (section dict) or list/dict of them.
    Returns (n, len(sections)*d) float32 array, row order = input order.
    sentence-transformers names run locally (api_key may be None)."""
    if isinstance(graded, dict) and any(k in graded for k in sections):
        graded = [graded]
    if isinstance(graded, dict):
        graded = list(graded.values())
    texts = [str(g.get(s, '') or ' ') for g in graded for s in sections]

    # HF ids carry a slash (org/name); OpenAI-style API ids do not.
    local = api_key is None or '/' in embedding_model_name
    if local:
        from .embedder import make_sentence_transformer_embed_fn
        E = make_sentence_transformer_embed_fn(model_name=embedding_model_name)(texts)
    else:
        rows = []
        for i in range(0, len(texts), batch):
            r = requests.post(base_url.rstrip('/') + '/embeddings',
                              json={'model': embedding_model_name,
                                    'input': texts[i:i + batch]},
                              headers={'Authorization': f'Bearer {api_key}'},
                              timeout=(15, 120))
            r.raise_for_status()
            rows.extend(d['embedding'] for d in r.json()['data'])
        E = np.asarray(rows)
    E = np.asarray(E, dtype=np.float32).reshape(len(graded), len(sections), -1)
    return E.reshape(len(graded), -1)


def consensus_center(X, groups, n_sections=len(DEFAULT_SECTIONS)):
    """Per-group (typically per-instance) median centering + per-section L2.
    X: (n, k*d) from embed_graded; groups: length-n labels (instance ids)."""
    X = np.asarray(X, dtype=np.float32)
    n = len(X)
    S = X.reshape(n, n_sections, -1)
    out = np.zeros_like(S)
    groups = np.asarray(groups)
    for g in np.unique(groups):
        sel = groups == g
        block = S[sel] - np.median(S[sel], axis=0, keepdims=True)
        norm = np.linalg.norm(block, axis=-1, keepdims=True)
        out[sel] = block / np.maximum(norm, 1e-9)
    return out.reshape(n, -1)
