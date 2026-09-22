"""Export and validate a self-contained KEV Qwen3 vLLM artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import safetensors
import torch
import transformers
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer

ARCHITECTURE = "KevQwen3ForDecision"
FORMAT_VERSION = 1
HEAD_KEYS = ("q.weight", "q.bias", "k.weight", "k.bias")
STRUCTURAL_TOKENS = (
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|fim_suffix|>",
)
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
)
REQUIRED_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _artifact_file(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
        raise ValueError(f"invalid artifact filename: {name}")
    return root / relative


def _checked_output_path(
    requested: Path, source: Path, base: Path, provenance: Path
) -> Path:
    absolute = requested.absolute()
    if absolute.is_symlink():
        raise ValueError("output path must not be a symbolic link")
    output = absolute.resolve()
    protected = (source, base, provenance)
    if output == Path(output.anchor) or output == Path.home().resolve():
        raise ValueError("refusing unsafe output path")
    if any(
        output == item or output.is_relative_to(item) or item.is_relative_to(output)
        for item in protected
    ):
        raise ValueError("output path overlaps an input path")
    return output


def _checked_source(source: Path, provenance: dict[str, Any]) -> dict[str, Any]:
    for field in ("resolved_revision", "base_revision", "reference_revision"):
        revision = provenance.get(field)
        if (
            not isinstance(revision, str)
            or REVISION_PATTERN.fullmatch(revision) is None
        ):
            raise ValueError(f"provenance {field} must be an immutable commit SHA")
    expected = provenance["files_sha256"]
    for name in ("adapter_config.json", "adapter_model.safetensors", "head.pt"):
        actual = sha256(source / name)
        if actual != expected.get(name):
            raise ValueError(f"source checksum mismatch: {name}")
    # torch.load is only used after authenticating the fixed source file.
    metadata = torch.load(source / "head.pt", map_location="cpu", weights_only=True)
    if metadata.get("base") != provenance["base_repository"]:
        raise ValueError("head base repository does not match provenance")
    if metadata.get("base_revision") != provenance["base_revision"]:
        raise ValueError("head base revision does not match provenance")
    if metadata.get("option_isolation", False):
        raise ValueError("option-isolation checkpoints are not supported")
    if metadata.get("special_embeddings", False):
        raise ValueError("trained special embeddings are not supported")
    head = metadata.get("head")
    if not isinstance(head, dict) or set(head) != set(HEAD_KEYS):
        raise ValueError("head.pt must contain exactly q/k weight and bias")
    hidden_size = provenance["head"]["hidden_size"]
    head_dim = metadata.get("head_dim")
    if provenance["head"].get("head_dim") != head_dim:
        raise ValueError("head dimension does not match provenance")
    expected_shapes = {
        "q.weight": (head_dim, hidden_size),
        "q.bias": (head_dim,),
        "k.weight": (head_dim, hidden_size),
        "k.bias": (head_dim,),
    }
    for name, tensor in head.items():
        if (
            tensor.dtype != torch.float32
            or tuple(tensor.shape) != expected_shapes[name]
        ):
            raise ValueError(f"invalid head tensor dtype/shape: {name}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"non-finite head tensor: {name}")
    temperature = metadata.get("temperature", 1.0)
    if type(temperature) not in (int, float) or not math.isfinite(temperature):
        raise ValueError("head temperature must be finite")
    if temperature <= 0:
        raise ValueError("head temperature must be positive")
    if provenance["head"].get("temperature") != temperature:
        raise ValueError("head temperature does not match provenance")
    return metadata


def _adapter_pairs(adapter_path: Path) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    with safe_open(adapter_path, framework="pt", device="cpu") as source:
        keys = set(source.keys())
        pairs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for a_name in sorted(name for name in keys if name.endswith(".lora_A.weight")):
            b_name = a_name.replace(".lora_A.weight", ".lora_B.weight")
            if b_name not in keys:
                raise ValueError(f"missing adapter pair: {b_name}")
            if a_name.startswith("base_model.model."):
                base_name = a_name.removeprefix("base_model.model.")
            elif a_name.startswith("base_model."):
                base_name = a_name.removeprefix("base_model.")
            else:
                raise ValueError(f"unsupported adapter tensor name: {a_name}")
            base_name = base_name.replace(".lora_A.weight", ".weight")
            pairs[base_name] = (source.get_tensor(a_name), source.get_tensor(b_name))
        expected_b = {
            name.replace(".lora_A.weight", ".lora_B.weight")
            for name in keys
            if name.endswith(".lora_A.weight")
        }
        if {
            name for name in keys if name.endswith(".lora_B.weight")
        } != expected_b or len(keys) != 2 * len(pairs):
            raise ValueError("adapter contains unsupported or unpaired tensors")
        return pairs


def merge_weight(
    base: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale: float,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    if base.ndim != 2 or a.ndim != 2 or b.ndim != 2:
        raise ValueError("LoRA merge requires matrix tensors")
    if b.shape[1] != a.shape[0] or (b.shape[0], a.shape[1]) != base.shape:
        raise ValueError("LoRA matrices do not match the base weight")
    merged = base.float().addmm(b.float(), a.float(), beta=1.0, alpha=scale)
    if not torch.isfinite(merged).all():
        raise ValueError("LoRA merge produced non-finite values")
    return merged.to(output_dtype)


def _weight_files(base: Path) -> tuple[list[str], dict[str, Any] | None]:
    index_path = base / "model.safetensors.index.json"
    if index_path.exists():
        index = read_json(index_path)
        return sorted(set(index["weight_map"].values())), index
    single = base / "model.safetensors"
    if single.exists():
        return [single.name], None
    raise ValueError("base snapshot has no safetensors weights")


def export(args: argparse.Namespace) -> None:
    source = args.source.resolve()
    base = args.base.resolve()
    provenance_path = args.provenance.resolve()
    output = _checked_output_path(args.output, source, base, provenance_path)
    if output.exists() and not args.force:
        raise FileExistsError(f"output already exists: {output}")
    if output.exists() and not output.is_dir():
        raise ValueError("output path exists and is not a directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.export-", dir=output.parent)
    )
    try:
        _export_to(args, source, base, provenance_path, staging)
        if output.exists():
            if not args.force:
                raise FileExistsError(f"output appeared during export: {output}")
            shutil.rmtree(output)
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _export_to(
    args: argparse.Namespace,
    source: Path,
    base: Path,
    provenance_path: Path,
    output: Path,
) -> None:
    provenance = read_json(provenance_path)
    metadata = _checked_source(source, provenance)
    adapter_config = read_json(source / "adapter_config.json")
    if adapter_config.get("base_model_name_or_path") != provenance["base_repository"]:
        raise ValueError("adapter base repository does not match provenance")
    if adapter_config.get("peft_type") != "LORA":
        raise ValueError("only standard LoRA adapters are supported")
    unsupported_lora_modes = (
        "fan_in_fan_out",
        "use_dora",
        "use_qalora",
        "use_rslora",
    )
    enabled_modes = [
        name for name in unsupported_lora_modes if adapter_config.get(name)
    ]
    if enabled_modes:
        raise ValueError(f"unsupported LoRA modes: {enabled_modes}")
    if adapter_config.get("trainable_token_indices"):
        raise ValueError("adapters with trained token embeddings are unsupported")
    rank = adapter_config.get("r")
    alpha = adapter_config.get("lora_alpha")
    if (
        type(rank) is not int
        or rank <= 0
        or type(alpha) not in (int, float)
        or not math.isfinite(alpha)
        or alpha <= 0
    ):
        raise ValueError("invalid LoRA rank or alpha")
    if adapter_config.get("rank_pattern") or adapter_config.get("alpha_pattern"):
        raise ValueError("per-module LoRA rank/alpha patterns are unsupported")
    pairs = _adapter_pairs(source / "adapter_model.safetensors")
    if not pairs:
        raise ValueError("adapter contains no LoRA tensors")
    if any(a.shape[0] != rank for a, _ in pairs.values()):
        raise ValueError("adapter tensor rank does not match adapter config")

    output_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    scale = float(alpha) / rank
    weight_files, _ = _weight_files(base)
    used: set[str] = set()
    weight_map: dict[str, str] = {}
    total_size = 0
    for filename in weight_files:
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(base / filename, framework="pt", device="cpu") as shard:
            for name in shard.keys():  # noqa: SIM118 - safe_open is not iterable
                tensor = shard.get_tensor(name)
                if name in pairs:
                    a, b = pairs[name]
                    tensor = merge_weight(tensor, a, b, scale, output_dtype)
                    used.add(name)
                elif tensor.is_floating_point():
                    tensor = tensor.to(output_dtype)
                if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                    raise ValueError(f"non-finite backbone tensor: {name}")
                tensors[name] = tensor
                weight_map[name] = filename
                total_size += tensor.numel() * tensor.element_size()
        save_file(tensors, output / filename)
    if used != set(pairs):
        missing = sorted(set(pairs) - used)
        raise ValueError(f"adapter targets absent from base model: {missing[:3]}")

    head_tensors = {
        f"pooler.{name}": tensor.contiguous()
        for name, tensor in metadata["head"].items()
    }
    head_file = "kev-head.safetensors"
    save_file(head_tensors, output / head_file)
    for name, tensor in head_tensors.items():
        weight_map[name] = head_file
        total_size += tensor.numel() * tensor.element_size()
    write_json(
        output / "model.safetensors.index.json",
        {"metadata": {"total_size": total_size}, "weight_map": weight_map},
    )

    config = read_json(base / "config.json")
    if (
        config.get("model_type") != "qwen3"
        or config.get("hidden_size") != provenance["head"]["hidden_size"]
    ):
        raise ValueError("base config is not the fixed Qwen3 architecture")
    if type(args.max_branch_tokens) is not int or args.max_branch_tokens <= 0:
        raise ValueError("max branch tokens must be positive")
    config.update(
        {
            "architectures": [ARCHITECTURE],
            "kev_backbone_dtype": args.dtype,
            "kev_head_dim": metadata["head_dim"],
            "kev_max_branch_tokens": args.max_branch_tokens,
            "kev_temperature": float(metadata.get("temperature", 1.0)),
            "torch_dtype": args.dtype,
        }
    )
    write_json(output / "config.json", config)
    missing_tokenizer_files = [
        name for name in REQUIRED_TOKENIZER_FILES if not (source / name).is_file()
    ]
    if missing_tokenizer_files:
        raise ValueError(f"source tokenizer is incomplete: {missing_tokenizer_files}")
    for name in TOKENIZER_FILES:
        if (source / name).exists():
            shutil.copy2(source / name, output / name)

    tokenizer = AutoTokenizer.from_pretrained(output, local_files_only=True)
    vocabulary = tokenizer.get_vocab()
    structural_ids = {
        token: tokenizer.convert_tokens_to_ids(token) for token in STRUCTURAL_TOKENS
    }
    if (
        any(token not in vocabulary for token in STRUCTURAL_TOKENS)
        or any(type(value) is not int or value < 0 for value in structural_ids.values())
        or len(set(structural_ids.values())) != len(structural_ids)
    ):
        raise ValueError("tokenizer does not define distinct KEV structural tokens")

    manifest = {
        "format_version": FORMAT_VERSION,
        "architecture": ARCHITECTURE,
        "model_id": args.model_id,
        "sources": {
            "checkpoint_repository": provenance["repository"],
            "checkpoint_revision": provenance["resolved_revision"],
            "base_repository": provenance["base_repository"],
            "base_revision": provenance["base_revision"],
            "reference_repository": provenance["reference_repository"],
            "reference_revision": provenance["reference_revision"],
            "files_sha256": {
                name: provenance["files_sha256"][name]
                for name in (
                    "adapter_config.json",
                    "adapter_model.safetensors",
                    "head.pt",
                )
            },
        },
        "merge": {"method": "fp32_lora", "rank": rank, "alpha": alpha},
        "backbone_dtype": args.dtype,
        "head_dtype": "float32",
        "head_dim": metadata["head_dim"],
        "temperature": float(metadata.get("temperature", 1.0)),
        "option_isolation": False,
        "structural_token_ids": structural_ids,
        "export": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "safetensors": safetensors.__version__,
            "transformers": transformers.__version__,
            "parameters": {
                "dtype": args.dtype,
                "max_branch_tokens": args.max_branch_tokens,
                "model_id": args.model_id,
            },
        },
    }
    manifest["files_sha256"] = {
        path.relative_to(output).as_posix(): sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "kev_manifest.json"
    }
    write_json(output / "kev_manifest.json", manifest)
    validate_artifact(output)


def validate_artifact(path: Path) -> None:
    path = path.resolve()
    if any(entry.is_symlink() or not entry.is_file() for entry in path.iterdir()):
        raise ValueError("artifact must contain only regular files")
    manifest = read_json(path / "kev_manifest.json")
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported KEV artifact format")
    if manifest.get("architecture") != ARCHITECTURE:
        raise ValueError("invalid KEV architecture")
    if not isinstance(manifest.get("model_id"), str) or not manifest["model_id"]:
        raise ValueError("invalid KEV model ID")
    if manifest.get("option_isolation") is not False:
        raise ValueError("option-isolation artifacts are not supported")
    if manifest.get("backbone_dtype") not in ("bfloat16", "float16"):
        raise ValueError("invalid backbone dtype")
    if manifest.get("head_dtype") != "float32":
        raise ValueError("invalid pointer head dtype")
    if type(manifest.get("head_dim")) is not int or manifest["head_dim"] <= 0:
        raise ValueError("invalid pointer head dimension")
    merge = manifest.get("merge")
    if (
        not isinstance(merge, dict)
        or merge.get("method") != "fp32_lora"
        or type(merge.get("rank")) is not int
        or merge["rank"] <= 0
        or type(merge.get("alpha")) not in (int, float)
        or not math.isfinite(merge["alpha"])
        or merge["alpha"] <= 0
    ):
        raise ValueError("invalid LoRA merge metadata")
    temperature = manifest.get("temperature")
    if (
        type(temperature) not in (int, float)
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("temperature must be finite and positive")
    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("artifact manifest has no source provenance")  # noqa: TRY004
    for field in ("checkpoint_revision", "base_revision", "reference_revision"):
        revision = sources.get(field)
        if (
            not isinstance(revision, str)
            or REVISION_PATTERN.fullmatch(revision) is None
        ):
            raise ValueError(f"source {field} is not an immutable commit SHA")
    source_checksums = sources.get("files_sha256")
    if not isinstance(source_checksums, dict) or set(source_checksums) != {
        "adapter_config.json",
        "adapter_model.safetensors",
        "head.pt",
    }:
        raise ValueError("artifact source checksums are incomplete")
    if any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in source_checksums.values()
    ):
        raise ValueError("artifact source checksum is invalid")
    structural_ids = manifest.get("structural_token_ids")
    if (
        not isinstance(structural_ids, dict)
        or set(structural_ids) != set(STRUCTURAL_TOKENS)
        or any(type(value) is not int or value < 0 for value in structural_ids.values())
        or len(set(structural_ids.values())) != len(STRUCTURAL_TOKENS)
    ):
        raise ValueError("invalid structural token ID manifest")
    checksums = manifest.get("files_sha256")
    if not isinstance(checksums, dict) or not checksums:
        raise ValueError("artifact manifest has no file checksums")
    for name in checksums:
        _artifact_file(path, name)
    actual_files = {
        file.name
        for file in path.iterdir()
        if file.is_file() and file.name != "kev_manifest.json"
    }
    if set(checksums) != actual_files:
        raise ValueError("artifact checksum inventory does not match its files")
    for name, expected in checksums.items():
        file = _artifact_file(path, name)
        if (
            not isinstance(expected, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected) is None
            or not file.is_file()
            or sha256(file) != expected
        ):
            raise ValueError(f"artifact checksum mismatch: {name}")
    config = read_json(path / "config.json")
    if config.get("model_type") != "qwen3" or config.get("architectures") != [
        ARCHITECTURE
    ]:
        raise ValueError("config architecture does not match manifest")
    for field in ("kev_head_dim", "kev_temperature", "kev_backbone_dtype"):
        expected = manifest[field.removeprefix("kev_")]
        if config.get(field) != expected:
            raise ValueError(f"config/manifest mismatch: {field}")
    if (
        type(config.get("kev_max_branch_tokens")) is not int
        or config["kev_max_branch_tokens"] <= 0
    ):
        raise ValueError("max branch tokens must be positive")
    index = read_json(path / "model.safetensors.index.json")
    if not isinstance(index.get("weight_map"), dict) or not index["weight_map"]:
        raise ValueError("artifact has no weight map")
    head_names = {f"pooler.{name}" for name in HEAD_KEYS}
    weight_files = set(index["weight_map"].values())
    actual_total_size = 0
    for filename in weight_files:
        file = _artifact_file(path, filename)
        if filename not in checksums or not file.is_file():
            raise ValueError(f"unverified or missing weight file: {filename}")
        expected_names = {
            name
            for name, mapped_file in index["weight_map"].items()
            if mapped_file == filename
        }
        with safe_open(file, framework="pt", device="cpu") as tensors:
            if set(tensors.keys()) != expected_names:
                raise ValueError(f"weight index does not match shard: {filename}")
            for name in tensors.keys():  # noqa: SIM118 - safe_open is not iterable
                tensor = tensors.get_tensor(name)
                actual_total_size += tensor.numel() * tensor.element_size()
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"non-finite artifact tensor: {name}")
                if name not in head_names and tensor.is_floating_point():
                    expected_dtype = {
                        "bfloat16": torch.bfloat16,
                        "float16": torch.float16,
                    }[manifest["backbone_dtype"]]
                    if tensor.dtype != expected_dtype:
                        raise ValueError(f"invalid backbone tensor dtype: {name}")
    if index.get("metadata", {}).get("total_size") != actual_total_size:
        raise ValueError("weight index total_size does not match tensors")
    if not head_names.issubset(index["weight_map"]):
        raise ValueError("artifact is missing pointer head parameters")
    head_file = _artifact_file(path, index["weight_map"]["pooler.q.weight"])
    with safe_open(head_file, framework="pt", device="cpu") as tensors:
        if not head_names.issubset(tensors.keys()):
            raise ValueError("artifact head file is missing pointer parameters")
        hidden_size = config["hidden_size"]
        head_dim = config["kev_head_dim"]
        expected_shapes = {
            "pooler.q.weight": (head_dim, hidden_size),
            "pooler.q.bias": (head_dim,),
            "pooler.k.weight": (head_dim, hidden_size),
            "pooler.k.bias": (head_dim,),
        }
        for name, shape in expected_shapes.items():
            tensor = tensors.get_tensor(name)
            if tensor.dtype != torch.float32 or tuple(tensor.shape) != shape:
                raise ValueError(f"invalid artifact head tensor: {name}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"non-finite artifact head tensor: {name}")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    vocabulary = tokenizer.get_vocab()
    actual_ids = {
        token: tokenizer.convert_tokens_to_ids(token) for token in STRUCTURAL_TOKENS
    }
    if (
        any(token not in vocabulary for token in STRUCTURAL_TOKENS)
        or len(set(actual_ids.values())) != len(actual_ids)
        or actual_ids != structural_ids
    ):
        raise ValueError("tokenizer structural IDs do not match manifest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    exporter = subparsers.add_parser("export")
    exporter.add_argument("--source", type=Path, required=True)
    exporter.add_argument("--base", type=Path, required=True)
    exporter.add_argument("--output", type=Path, required=True)
    exporter.add_argument("--provenance", type=Path, required=True)
    exporter.add_argument("--model-id", default="kev-qwen3-4b")
    exporter.add_argument("--dtype", choices=("bfloat16", "float16"), required=True)
    exporter.add_argument("--max-branch-tokens", type=int, default=8192)
    exporter.add_argument("--force", action="store_true")
    validator = subparsers.add_parser("validate")
    validator.add_argument("artifact", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "export":
        export(args)
    else:
        validate_artifact(args.artifact)


if __name__ == "__main__":
    main()
