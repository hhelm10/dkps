"""Build a portable, deduplicated Hugging Face release from verified EG2 caches.

Run from the dkps repository root. No encoding or network access is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[4]
ASSETS = ROOT / "projects/pkps/hf_dataset"
sys.path.insert(0, str(ASSETS))
from load_embeddings import sha256_file, verify

COHORTS = {"eee_69": "eee-extensions", "eee_45": "eee-followups", "style": "eee-style"}
CONFIG_KEYS = ("model", "revision", "dtype", "prompt", "max_length", "dimension", "normalize")


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def portable_metadata(value):
    if isinstance(value, dict):
        return {k: portable_metadata(v) for k, v in value.items()
                if not (k in {"model_path", "source"} and isinstance(v, str) and Path(v).is_absolute())}
    if isinstance(value, list):
        return [portable_metadata(v) for v in value]
    return value


def read_cache(source):
    pool = pd.read_parquet(source / "pool.parquet")
    rows = pd.read_parquet(source / "embeddings/rows.parquet")
    config = json.loads((source / "embeddings/config.json").read_text())
    complete = json.loads((source / "embeddings/complete.json").read_text())
    texts = list(dict.fromkeys(pool["query"].tolist() + pool.response.tolist()))
    fingerprint = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()
    if fingerprint != config["text_sha256"] or complete["text_sha256"] != fingerprint:
        raise ValueError(f"Ordered-text integrity failure: {source.name}")
    arrays = [np.load(source / "embeddings" / f"{start:07d}.npy", allow_pickle=False)
              for start in range(0, config["n_texts"], config["chunk_size"])]
    vectors = np.concatenate(arrays)
    if vectors.shape != (len(texts), 768) or vectors.dtype != np.float32 or not np.isfinite(vectors).all():
        raise ValueError(f"Invalid vectors: {source.name}")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-6, rtol=0):
        raise ValueError(f"Non-unit vectors: {source.name}")
    pd.testing.assert_frame_equal(rows[pool.columns.drop(["query", "response"])],
                                  pool.drop(columns=["query", "response"]), check_like=True)
    for column, index in [("query", "q_index"), ("response", "r_index")]:
        if [texts[i] for i in rows[index]] != pool[column].tolist():
            raise ValueError(f"Incorrect text/index alignment: {source.name}/{index}")
    return pool, rows, config, texts, vectors


def build(out):
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Output must be empty; use a fresh directory for a new release")
    registry, text_list, bank, roles, summaries, inputs = {}, [], [], {}, {}, {}
    canonical = None
    checks = {}
    for cohort, dirname in COHORTS.items():
        source = ROOT / "data" / dirname
        pool, rows, config, texts, vectors = read_cache(source)
        signature = {k: config[k] for k in CONFIG_KEYS}
        if canonical is not None and canonical != signature:
            raise ValueError("Cannot combine incompatible embedding configurations")
        canonical = signature
        indices = []
        reused = 0
        for local, text in enumerate(texts):
            if text in registry:
                index = registry[text]
                if not np.array_equal(bank[index], vectors[local]):
                    raise ValueError("Identical text has different cached vectors")
                reused += 1
            else:
                index = len(text_list)
                registry[text] = index
                text_list.append(text)
                bank.append(vectors[local].copy())
                roles[index] = set()
            indices.append(index)
        indices = np.asarray(indices, dtype=np.int64)
        dest = out / "cohorts" / cohort
        dest.mkdir(parents=True)
        inputs[cohort] = {str(p.relative_to(source)): sha256_file(p)
                          for p in sorted(source.rglob("*")) if p.is_file()}
        pd.DataFrame({"local_index": np.arange(len(indices)), "embedding_index": indices}).to_parquet(
            dest / "original_indices.parquet", index=False)
        for column, role in [("q_index", "query"), ("r_index", "response")]:
            rows[column] = indices[rows[column].to_numpy()]
            for index in rows[column].unique():
                roles[index].add(role)
        rows.insert(0, "row_id", np.arange(len(rows)))
        if "variant" not in rows:
            rows["variant"] = "original"
        metadata_source = source if cohort != "style" else ROOT / "data/eee-followups"
        run_metadata = json.loads((metadata_source / "run_metadata.json").read_text())
        run_map = {(r["model_id"], r["suite"]): r for r in run_metadata}
        rows["source_run_path"] = [run_map[(m, s)]["path"] for m, s in zip(rows.model_id, rows.suite)]
        rows["evaluation_timestamp"] = [run_map[(m, s)]["evaluation_timestamp"] for m, s in zip(rows.model_id, rows.suite)]
        rows.to_parquet(dest / "rows.parquet", index=False)
        targets = pd.read_parquet(source / "targets.parquet")
        targets = targets[targets.model_id.isin(rows.model_id.unique())].reset_index(drop=True)
        targets.to_parquet(dest / "targets.parquet", index=False)
        save_json(dest / "embedding_config.json", portable_metadata(config))
        for filename in ["dataset_manifest.json", "run_metadata.json", "style_design.json"]:
            path = source / filename
            if path.exists():
                save_json(dest / filename, portable_metadata(json.loads(path.read_text())))
        if (source / "observations.parquet").exists():
            shutil.copyfile(source / "observations.parquet", dest / "observations.parquet")
        summaries[cohort] = dict(source_cache=dirname, rows=len(rows), models=rows.model_id.nunique(),
                                tasks=rows.task_id.nunique(), queries=rows.query_id.nunique(),
                                full_depth_targets=len(targets), local_unique_texts=len(texts),
                                vectors_reused_from_earlier_cohorts=reused,
                                variants=sorted(rows.variant.unique().tolist()))
        checks[cohort] = dict(original_text_order_hash_verified=True, original_rows_verified=True,
                             source_vectors_finite_unit_float32=True, reused_vectors_bitwise_equal=True)
        print(cohort, summaries[cohort], flush=True)
    vectors = np.stack(bank)
    np.save(out / "embeddings.npy", vectors, allow_pickle=False)
    text_frame = pd.DataFrame({"embedding_index": np.arange(len(text_list)),
        "text_sha256": [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in text_list],
        "text": text_list, "is_query": ["query" in roles[i] for i in range(len(text_list))],
        "is_response": ["response" in roles[i] for i in range(len(text_list))]})
    assert not text_frame.text_sha256.duplicated().any()
    text_frame.to_parquet(out / "texts.parquet", index=False)
    assert len(vectors) == 40327, "Update the versioned dataset card for a changed release"
    provenance = out / "provenance"
    provenance.mkdir()
    save_json(provenance / "input_file_hashes.json", inputs)
    prompt = json.loads((ROOT / "data/models/embeddinggemma-2/config_sentence_transformers.json").read_text())["prompts"]["SentenceSimilarity"]
    save_json(provenance / "encoder.json", {**canonical, "prompt_text": prompt,
        "hardware": "Apple M5 / Metal (MPS)", "compute_dtype": "float32", "cpu_fallback_enabled": False,
        "learned_pca": False, "vectors_changed_during_packaging": False,
        "source_code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "packager_source": "code/package_eee_embeddings.py", "source_repository": "https://github.com/evijit/dkps",
        "source_dataset": "evaleval/EEE_datastore", "source_dataset_revision": "8b9632422dd2b3806431efd758c7cdc84044169f"})
    for kind in ["dataset", "model"]:
        shutil.copyfile(ROOT / "data/eee-pkps-hub-sources" / kind / "README.md", provenance / f"upstream_{kind}_card.md")
    code = out / "code"
    code.mkdir()
    shutil.copyfile(Path(__file__), code / Path(__file__).name)
    for filename in ["prepare_eee_followups.py", "embed_eee_followups.py", "embed_stress_extensions.py"]:
        shutil.copyfile(Path(__file__).parent / filename, code / filename)
    for path in ASSETS.iterdir():
        if path.is_file():
            shutil.copyfile(path, out / path.name)
    tables = {}
    for path in sorted(out.rglob("*.parquet")):
        metadata = pq.read_metadata(path)
        tables[str(path.relative_to(out))] = {"rows": metadata.num_rows,
            "columns": [{"name": field.name, "type": str(field.type), "nullable": field.nullable}
                        for field in pq.read_schema(path)]}
    save_json(out / "schemas.json", tables)
    save_json(out / "validation.json", {"checks": checks, "embedding_shape": list(vectors.shape),
        "max_unit_norm_error": float(np.max(abs(np.linalg.norm(vectors, axis=1) - 1))),
        "unique_text_hashes": True, "source_embedding_values_preserved_exactly": True})
    files = [{"path": str(p.relative_to(out)), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
             for p in sorted(out.rglob("*")) if p.is_file()]
    manifest = {"format_version": "1.0.0", "dataset_id": "evaleval/eee-pkps-embeddings",
        "release_date": "2026-10-12", "visibility": "private", "embedding_shape": list(vectors.shape),
        "embedding_dtype": "float32", "index_scope": "one shared bank for every cohort",
        "cohorts": summaries, "files": files,
        "checksum_convention": "MANIFEST.json lists all payload files except itself and SHA256SUMS; SHA256SUMS also covers MANIFEST.json"}
    save_json(out / "MANIFEST.json", manifest)
    paths = sorted(p for p in out.rglob("*") if p.is_file())
    (out / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.relative_to(out)}\n" for p in paths))
    verify(out)
    print("Validated release:", out, "files:", len(paths) + 1, "vectors:", vectors.shape, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/eee-pkps-embeddings")
    build(parser.parse_args().out)
