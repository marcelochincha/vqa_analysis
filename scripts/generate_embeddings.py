from __future__ import annotations

import argparse
import json
import pickle
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# Reserved (non-tuple) key under which run metadata is stored inside the cache
# dict. All real entries are 4-tuples, so this string key never collides and is
# transparent to consumers that look up embeddings by key.
META_KEY = "__meta__"


def build_key(agent: str, video: str, question_num: int, repetition: int) -> tuple:
	return (str(agent), str(video), int(question_num), int(repetition))


def _git_commit() -> str | None:
	try:
		root = Path(__file__).resolve().parents[1]
		out = subprocess.run(
			["git", "rev-parse", "HEAD"],
			cwd=root,
			capture_output=True,
			text=True,
			check=True,
		)
		return out.stdout.strip() or None
	except Exception:
		return None


def _library_versions() -> dict:
	versions: dict = {}
	import platform

	versions["python"] = platform.python_version()
	for mod in ("torch", "transformers", "sentence_transformers"):
		try:
			versions[mod] = __import__(mod).__version__
		except Exception:
			versions[mod] = None
	return versions


def resolve_model_meta(model: SentenceTransformer) -> dict:
	"""Capture ground-truth properties from the *loaded* model.

	This is the whole point of the metadata: record what actually happened
	(real dtype, real attention implementation, real dim) rather than what a
	filename or CLI flag claims.
	"""
	meta: dict = {}
	try:
		auto = model[0].auto_model
		meta["resolved_dtype"] = str(getattr(auto, "dtype", None))
		meta["resolved_attn_implementation"] = getattr(
			auto.config, "_attn_implementation", None
		)
	except Exception as exc:  # pragma: no cover - defensive
		meta["resolve_error"] = repr(exc)
	try:
		meta["embedding_dim"] = int(model.get_sentence_embedding_dimension())
	except Exception:
		pass
	try:
		meta["max_seq_length"] = int(model.max_seq_length)
	except Exception:
		pass
	return meta


def build_metadata(
	args: argparse.Namespace,
	data_path: Path,
	n_entries: int,
	model_meta: dict | None,
	prior_meta: dict | None,
) -> dict:
	# Start from prior metadata so ground-truth model fields survive a
	# fully-cached resume run (where the model is never loaded).
	meta: dict = dict(prior_meta) if isinstance(prior_meta, dict) else {}
	meta.update(
		{
			"model": args.model,
			"batch_size": args.batch_size,
			"text_col": args.text_col,
			"normalize": bool(args.normalize),
			"instruction": args.instruction,
			"requested_padding_side": args.padding_side,
			"requested_dtype": args.dtype,
			"requested_max_seq_length": args.max_seq_length,
			"device": args.device,
			"data": str(data_path),
			"n_entries": n_entries,
			"last_run_utc": datetime.now(timezone.utc).isoformat(),
			"git_commit": _git_commit(),
			"versions": _library_versions(),
		}
	)
	if model_meta:
		meta.update(model_meta)
	return meta


def save_meta_sidecar(meta: dict, cache_path: Path) -> Path:
	sidecar = cache_path.with_suffix(cache_path.suffix + ".meta.json")
	sidecar.parent.mkdir(parents=True, exist_ok=True)
	with sidecar.open("w", encoding="utf-8") as handle:
		json.dump(meta, handle, indent=2, sort_keys=True)
	return sidecar


# Short slugs used to derive a default cache filename from --model.
# Falls back to a sanitized last-path-component for anything unmapped.
_MODEL_SLUGS: dict[str, str] = {
	"sentence-transformers/all-mpnet-base-v2": "allmpnet",
	"sentence-transformers/all-minilm-l6-v2": "miniLM",
	"qwen/qwen3-embedding-4b": "qwen3emb4b",
	"qwen/qwen3-embedding-0.6b": "qwen3emb0p6b",
	"qwen/qwen3-embedding-8b": "qwen3emb8b",
}


def model_slug(model: str) -> str:
	key = model.lower()
	if key in _MODEL_SLUGS:
		return _MODEL_SLUGS[key]
	tail = model.rsplit("/", 1)[-1]
	cleaned = "".join(c for c in tail if c.isalnum())
	return cleaned or "model"


def load_cache(path: Path) -> dict:
	if not path.exists():
		return {}
	with path.open("rb") as handle:
		return pickle.load(handle)


def save_cache(cache: dict, path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("wb") as handle:
		pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_npz(keys: list[tuple], cache: dict, path: Path, meta: dict | None = None) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	agents = np.array([k[0] for k in keys], dtype=str)
	videos = np.array([k[1] for k in keys], dtype=str)
	qnums = np.array([k[2] for k in keys], dtype=np.int32)
	reps = np.array([k[3] for k in keys], dtype=np.int32)
	embeddings = np.vstack([cache[k] for k in keys]).astype(np.float32)
	np.savez_compressed(
		path,
		agent=agents,
		video=videos,
		question_num=qnums,
		repetition=reps,
		embedding=embeddings,
		meta_json=np.array(json.dumps(meta or {})),
	)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Generate keyed embedding cache from cleaned CSV")
	parser.add_argument("--data", type=Path, default=None, help="Input CSV (default: data/r2_cleaned.csv)")
	parser.add_argument(
		"--output",
		type=Path,
		default=None,
		help=(
			"Output pickle cache. Default is derived from --model and --batch-size, "
			"e.g. external_embeds/qwen3emb4b_batch1_r2_embeddings_cache_keyed.pkl"
		),
	)
	parser.add_argument("--model", required=True, help="Embedding model name")
	parser.add_argument("--text-col", default="ANSWER", help="Column with text to embed")
	parser.add_argument(
		"--batch-size",
		type=int,
		default=1,
		help=(
			"Encode batch size. Default is 1 for full reproducibility — large models "
			"produce slightly different embeddings depending on batch size due to "
			"non-deterministic accumulation order in fp16/bf16 matmuls. Increase only "
			"if reproducibility across runs is not required."
		),
	)
	parser.add_argument("--checkpoint-size", type=int, default=5000, help="Save every N new embeddings")
	parser.add_argument("--limit", type=int, default=None, help="Limit rows for testing")
	parser.add_argument("--device", default=None, help="Force device (e.g. cpu, cuda)")
	parser.add_argument("--normalize", action="store_true", help="Normalize embeddings")
	parser.add_argument("--trust-remote-code", action="store_true", help="Allow remote code for model")
	parser.add_argument(
		"--dtype",
		choices=["fp32", "fp16", "bf16"],
		default=None,
		help="Model dtype (use fp16/bf16 for large models like Qwen3-Embedding-4B to fit in VRAM)",
	)
	parser.add_argument(
		"--padding-side",
		choices=["left", "right"],
		default=None,
		help="Tokenizer padding side (Qwen3-Embedding requires 'left' for correct last-token pooling)",
	)
	parser.add_argument(
		"--instruction",
		default=None,
		help="Optional task-instruction prefix for query-side encoding (Qwen3-Embedding etc.)",
	)
	parser.add_argument(
		"--max-seq-length",
		type=int,
		default=None,
		help="Override the model's max sequence length (useful when defaults are very high, e.g. 32k for Qwen3-Embedding)",
	)
	parser.add_argument("--npz", type=Path, default=None, help="Optional .npz output")
	parser.add_argument("--resume", action="store_true", help="Resume from existing cache")
	parser.add_argument("--no-resume", dest="resume", action="store_false", help="Do not load existing cache")
	parser.set_defaults(resume=True)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	root = Path(__file__).resolve().parents[1]
	data_path = args.data or root / "data" / "r2_cleaned.csv"

	if args.output:
		output_path = args.output
	else:
		default_name = f"{model_slug(args.model)}_batch{args.batch_size}_r2_embeddings_cache_keyed.pkl"
		output_path = root / "external_embeds" / default_name
		print(f"Output not specified; using derived path: {output_path}")

	df = pd.read_csv(data_path, keep_default_na=False)
	if args.limit:
		df = df.head(args.limit)

	required_cols = ["AGENT", "VIDEO", "QUESTION_NUM", "REPETITION", args.text_col]
	missing = [c for c in required_cols if c not in df.columns]
	if missing:
		raise ValueError(f"Missing columns in {data_path}: {missing}")

	agents = df["AGENT"].astype(str).tolist()
	videos = df["VIDEO"].astype(str).tolist()
	qnums = df["QUESTION_NUM"].astype(int).tolist()
	reps = df["REPETITION"].astype(int).tolist()
	keys = [build_key(a, v, q, r) for a, v, q, r in zip(agents, videos, qnums, reps)]

	cache = load_cache(output_path) if args.resume else {}
	prior_meta = cache.pop(META_KEY, None)
	missing_indices = [i for i, k in enumerate(keys) if k not in cache]
	model_meta: dict | None = None

	print(f"Total rows: {len(keys):,}")
	print(f"Cached: {len(keys) - len(missing_indices):,}")
	print(f"To encode: {len(missing_indices):,}")

	if missing_indices:
		st_kwargs: dict = {}
		model_kwargs: dict = {}
		tokenizer_kwargs: dict = {}

		if args.device:
			st_kwargs["device"] = args.device

		# Auto-configure for Qwen3-Embedding family (decoder LLM with
		# last-token pooling; needs trust_remote_code + left padding).
		is_qwen3_embedding = "qwen3-embedding" in args.model.lower()
		trust_remote_code = args.trust_remote_code or is_qwen3_embedding
		if trust_remote_code:
			st_kwargs["trust_remote_code"] = True

		padding_side = args.padding_side or ("left" if is_qwen3_embedding else None)
		if padding_side:
			tokenizer_kwargs["padding_side"] = padding_side

		if args.dtype:
			import torch
			dtype_map = {
				"fp32": torch.float32,
				"fp16": torch.float16,
				"bf16": torch.bfloat16,
			}
			model_kwargs["torch_dtype"] = dtype_map[args.dtype]

		if model_kwargs:
			st_kwargs["model_kwargs"] = model_kwargs
		if tokenizer_kwargs:
			st_kwargs["tokenizer_kwargs"] = tokenizer_kwargs

		print(f"Loading model: {args.model}")
		print(f"  trust_remote_code={trust_remote_code} padding_side={padding_side} dtype={args.dtype}")
		model = SentenceTransformer(args.model, **st_kwargs)

		if args.max_seq_length:
			model.max_seq_length = args.max_seq_length
			print(f"  max_seq_length={args.max_seq_length}")

		# Ground truth from the actually-loaded model, so the cache can never
		# misreport its real dtype / attention impl the way a filename can.
		model_meta = resolve_model_meta(model)
		model_meta["trust_remote_code"] = trust_remote_code
		model_meta["resolved_padding_side"] = padding_side
		print(
			"  resolved: dtype={} attn={} dim={}".format(
				model_meta.get("resolved_dtype"),
				model_meta.get("resolved_attn_implementation"),
				model_meta.get("embedding_dim"),
			)
		)

		encode_kwargs: dict = {
			"batch_size": args.batch_size,
			"show_progress_bar": True,
			"normalize_embeddings": args.normalize,
		}
		if args.instruction:
			encode_kwargs["prompt"] = f"Instruct: {args.instruction}\nQuery: "
			print(f"  instruction prefix enabled")

		total = len(missing_indices)
		processed = 0
		for start in range(0, total, args.batch_size):
			batch_idx = missing_indices[start : start + args.batch_size]
			batch_texts = df.iloc[batch_idx][args.text_col].astype(str).tolist()
			batch_embeds = model.encode(batch_texts, **encode_kwargs)
			for idx, emb in zip(batch_idx, batch_embeds):
				cache[keys[idx]] = np.asarray(emb, dtype=np.float32)
			processed += len(batch_idx)

			if args.checkpoint_size and (processed % args.checkpoint_size == 0):
				save_cache(cache, output_path)
				print(f"Checkpoint saved: {processed:,}/{total:,} new embeddings")

	n_entries = len(cache)  # tuple entries only; META_KEY was popped on load
	meta = build_metadata(args, data_path, n_entries, model_meta, prior_meta)
	cache[META_KEY] = meta

	save_cache(cache, output_path)
	print(f"Saved cache: {output_path} ({n_entries:,} entries)")

	sidecar = save_meta_sidecar(meta, output_path)
	print(f"Saved metadata: {sidecar}")

	if args.npz:
		save_npz(keys, cache, args.npz, meta)
		print(f"Saved npz: {args.npz}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
