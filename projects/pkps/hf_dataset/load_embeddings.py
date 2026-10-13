"""Standalone NumPy/Pandas access to evaleval/eee-pkps-embeddings.

This module does not download data or load full-depth targets implicitly.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

COHORTS = ("eee_69", "eee_45", "style")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(root):
    """Verify every release file against MANIFEST.json and SHA256SUMS."""
    root = Path(root)
    manifest = json.loads((root / "MANIFEST.json").read_text())
    for record in manifest["files"]:
        path = root / record["path"]
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"File integrity failure: {record['path']}")
    for line in (root / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if sha256_file(root / name) != expected:
            raise ValueError(f"Checksum failure: {name}")
    return manifest


def load_cohort(root, cohort="eee_69"):
    """Return row metadata and the shared, read-only, memory-mapped vector bank.

    rows.q_index and rows.r_index address the shared bank directly. This function
    deliberately does not load cohorts/<cohort>/targets.parquet.
    """
    if cohort not in COHORTS:
        raise ValueError(f"cohort must be one of {COHORTS}")
    root = Path(root)
    rows = pd.read_parquet(root / "cohorts" / cohort / "rows.parquet")
    vectors = np.load(root / "embeddings.npy", mmap_mode="r", allow_pickle=False)
    if vectors.ndim != 2 or vectors.shape[1] != 768 or vectors.dtype != np.float32:
        raise ValueError("Expected a float32 (n, 768) vector bank")
    indices = rows[["q_index", "r_index"]].to_numpy()
    if indices.min() < 0 or indices.max() >= len(vectors):
        raise ValueError("Cohort embedding index out of bounds")
    return rows, vectors


def export_cache(root, destination, cohort="eee_69"):
    """Restore the original local embedding order and sharded PKPS cache layout.

    The destination must be empty. Targets are copied for compatibility with the
    experiment drivers; consumers must enforce their own label-access protocol.
    """
    rows, vectors = load_cohort(root, cohort)
    root, destination = Path(root), Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Cache export destination must be empty")
    cache = destination / "embeddings"
    cache.mkdir(parents=True, exist_ok=True)
    base = root / "cohorts" / cohort
    mapping = pd.read_parquet(base / "original_indices.parquet").sort_values("local_index")
    expected = np.arange(len(mapping))
    if not np.array_equal(mapping.local_index.to_numpy(), expected):
        raise ValueError("Original index map is not contiguous")
    reverse = dict(zip(mapping.embedding_index, mapping.local_index))
    exported = rows.drop(columns=["row_id", "source_run_path", "evaluation_timestamp"])
    if cohort != "style":
        exported = exported.drop(columns="variant")
    for column in ["q_index", "r_index"]:
        exported[column] = exported[column].map(reverse)
    if exported[["q_index", "r_index"]].isna().any().any():
        raise ValueError("Original index map is incomplete")
    exported.to_parquet(cache / "rows.parquet", index=False)
    texts = pd.read_parquet(root / "texts.parquet").set_index("embedding_index").text
    pool = exported.drop(columns=["q_index", "r_index"])
    pool["query"] = rows.q_index.map(texts)
    pool["response"] = rows.r_index.map(texts)
    pool.to_parquet(destination / "pool.parquet", index=False)
    local_texts = texts.loc[mapping.embedding_index].tolist()
    config = json.loads((base / "embedding_config.json").read_text())
    fingerprint = hashlib.sha256(json.dumps(local_texts, ensure_ascii=False).encode()).hexdigest()
    if fingerprint != config["text_sha256"]:
        raise ValueError("Original ordered-text hash mismatch")
    size = config["chunk_size"]
    for start in range(0, len(mapping), size):
        np.save(cache / f"{start:07d}.npy", vectors[mapping.embedding_index.iloc[start:start + size]],
                allow_pickle=False)
    (cache / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (cache / "complete.json").write_text(json.dumps({"n_texts": len(mapping), "text_sha256": fingerprint}) + "\n")
    for filename in ["targets.parquet", "dataset_manifest.json", "run_metadata.json",
                     "style_design.json", "observations.parquet"]:
        source = base / filename
        if source.exists():
            (destination / filename).write_bytes(source.read_bytes())
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--export", type=Path)
    parser.add_argument("--cohort", choices=COHORTS, default="eee_69")
    args = parser.parse_args()
    if args.verify:
        print("Verified", len(verify(args.root)["files"]), "release files")
    if args.export:
        print("Exported", export_cache(args.root, args.export, args.cohort))
    rows, vectors = load_cohort(args.root, args.cohort)
    print(args.cohort, "rows:", len(rows), "vector bank:", vectors.shape)
