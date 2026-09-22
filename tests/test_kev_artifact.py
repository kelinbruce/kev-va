import argparse
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tools import kev_artifact
from tools.kev_artifact import STRUCTURAL_TOKENS, merge_weight


class FakeTokenizer:
    def __init__(self):
        self.vocab = {
            token: 100 + index for index, token in enumerate(STRUCTURAL_TOKENS)
        }

    def get_vocab(self):
        return self.vocab

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, 0)


@pytest.fixture
def synthetic_export(tmp_path, monkeypatch):
    source = tmp_path / "source"
    base = tmp_path / "base"
    output = tmp_path / "output"
    source.mkdir()
    base.mkdir()
    (source / "tokenizer.json").write_text("{}")
    (source / "tokenizer_config.json").write_text("{}")
    adapter_config = {
        "base_model_name_or_path": "Qwen/Qwen3-4B-Base",
        "peft_type": "LORA",
        "r": 1,
        "lora_alpha": 2,
        "rank_pattern": {},
        "alpha_pattern": {},
    }
    (source / "adapter_config.json").write_text(json.dumps(adapter_config))
    save_file(
        {
            "base_model.model.model.layers.0.q_proj.lora_A.weight": torch.tensor(
                [[0.25, -0.5]]
            ),
            "base_model.model.model.layers.0.q_proj.lora_B.weight": torch.tensor(
                [[2.0], [-1.0]]
            ),
        },
        source / "adapter_model.safetensors",
    )
    torch.save(
        {
            "base": "Qwen/Qwen3-4B-Base",
            "base_revision": "2" * 40,
            "option_isolation": False,
            "special_embeddings": False,
            "head_dim": 1,
            "temperature": 2.5,
            "head": {
                "q.weight": torch.ones(1, 2),
                "q.bias": torch.zeros(1),
                "k.weight": torch.full((1, 2), 2.0),
                "k.bias": torch.ones(1),
            },
        },
        source / "head.pt",
    )
    original = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    save_file(
        {
            "model.layers.0.q_proj.weight": original,
            "model.embed_tokens.weight": torch.ones(3, 2, dtype=torch.float32),
        },
        base / "model.safetensors",
    )
    (base / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "hidden_size": 2})
    )
    provenance = {
        "repository": "jaredpalmer/kev-4b",
        "resolved_revision": "1" * 40,
        "base_repository": "Qwen/Qwen3-4B-Base",
        "base_revision": "2" * 40,
        "reference_repository": "jaredpalmer/kev",
        "reference_revision": "3" * 40,
        "head": {"hidden_size": 2, "head_dim": 1, "temperature": 2.5},
        "files_sha256": {
            name: kev_artifact.sha256(source / name)
            for name in (
                "adapter_config.json",
                "adapter_model.safetensors",
                "head.pt",
            )
        },
    }
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_text(json.dumps(provenance))
    monkeypatch.setattr(
        kev_artifact.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: FakeTokenizer(),
    )
    args = argparse.Namespace(
        source=source,
        base=base,
        output=output,
        provenance=provenance_path,
        model_id="test-kev",
        dtype="bfloat16",
        max_branch_tokens=128,
        force=False,
    )
    return args, original, provenance


def test_fp32_lora_merge_matches_reference_before_dtype_conversion():
    base = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    a = torch.tensor([[0.25, -0.5]], dtype=torch.float32)
    b = torch.tensor([[2.0], [-1.0]], dtype=torch.float32)
    expected = base.float() + 2.0 * (b @ a)

    merged = merge_weight(base, a, b, 2.0, torch.float32)
    torch.testing.assert_close(merged, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "a,b",
    [
        (torch.ones(2), torch.ones(2, 1)),
        (torch.ones(1, 3), torch.ones(2, 1)),
    ],
)
def test_lora_merge_rejects_invalid_shapes(a, b):
    with pytest.raises(ValueError):
        merge_weight(torch.ones(2, 2), a, b, 1.0, torch.float32)


def test_script_is_importable_from_workspace():
    assert Path(__file__).parents[1].joinpath("tools/kev_artifact.py").is_file()


def test_export_merges_standard_peft_names_and_builds_valid_artifact(
    synthetic_export,
):
    args, original, provenance = synthetic_export
    kev_artifact.export(args)

    with safe_open(args.output / "model.safetensors", framework="pt") as weights:
        merged = weights.get_tensor("model.layers.0.q_proj.weight")
        expected = original + 2.0 * torch.tensor([[2.0], [-1.0]]) @ torch.tensor(
            [[0.25, -0.5]]
        )
        torch.testing.assert_close(merged.float(), expected, atol=0.02, rtol=0)
        assert merged.dtype == torch.bfloat16
        assert weights.get_tensor("model.embed_tokens.weight").dtype == torch.bfloat16

    manifest = kev_artifact.read_json(args.output / "kev_manifest.json")
    assert manifest["temperature"] == 2.5
    assert manifest["sources"]["files_sha256"] == provenance["files_sha256"]
    assert manifest["export"]["parameters"]["max_branch_tokens"] == 128
    kev_artifact.validate_artifact(args.output)


def test_validate_rejects_corrupted_file(synthetic_export):
    args, _, _ = synthetic_export
    kev_artifact.export(args)
    with (args.output / "config.json").open("a") as stream:
        stream.write(" ")

    with pytest.raises(ValueError, match="checksum mismatch: config.json"):
        kev_artifact.validate_artifact(args.output)


def test_validate_rejects_path_traversal_in_manifest(synthetic_export):
    args, _, _ = synthetic_export
    kev_artifact.export(args)
    manifest_path = args.output / "kev_manifest.json"
    manifest = kev_artifact.read_json(manifest_path)
    manifest["files_sha256"]["../outside"] = "0" * 64
    kev_artifact.write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="invalid artifact filename"):
        kev_artifact.validate_artifact(args.output)


def test_export_rejects_source_checksum_mismatch(synthetic_export):
    args, _, _ = synthetic_export
    with (args.source / "adapter_config.json").open("a") as stream:
        stream.write(" ")

    with pytest.raises(ValueError, match="source checksum mismatch"):
        kev_artifact.export(args)


def test_export_rejects_option_isolation(synthetic_export):
    args, _, provenance = synthetic_export
    metadata = torch.load(args.source / "head.pt", weights_only=True)
    metadata["option_isolation"] = True
    torch.save(metadata, args.source / "head.pt")
    provenance["files_sha256"]["head.pt"] = kev_artifact.sha256(args.source / "head.pt")
    args.provenance.write_text(json.dumps(provenance))

    with pytest.raises(ValueError, match="option-isolation"):
        kev_artifact.export(args)


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("missing_bias", "exactly q/k weight and bias"),
        ("wrong_shape", "dtype/shape"),
        ("nonfinite", "non-finite"),
        ("zero_temperature", "temperature must be positive"),
    ],
)
def test_export_rejects_invalid_head(synthetic_export, failure, message):
    args, _, provenance = synthetic_export
    metadata = torch.load(args.source / "head.pt", weights_only=True)
    if failure == "missing_bias":
        del metadata["head"]["q.bias"]
    elif failure == "wrong_shape":
        metadata["head"]["q.weight"] = torch.ones(1, 3)
    elif failure == "nonfinite":
        metadata["head"]["q.bias"][0] = torch.nan
    else:
        metadata["temperature"] = 0
    torch.save(metadata, args.source / "head.pt")
    provenance["files_sha256"]["head.pt"] = kev_artifact.sha256(args.source / "head.pt")
    args.provenance.write_text(json.dumps(provenance))

    with pytest.raises(ValueError, match=message):
        kev_artifact.export(args)


def test_export_rejects_mutable_provenance_revision(synthetic_export):
    args, _, provenance = synthetic_export
    provenance["base_revision"] = "main"
    args.provenance.write_text(json.dumps(provenance))

    with pytest.raises(ValueError, match="immutable commit SHA"):
        kev_artifact.export(args)


def test_export_rejects_qwen35_config(synthetic_export):
    args, _, _ = synthetic_export
    args.base.joinpath("config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "hidden_size": 2})
    )

    with pytest.raises(ValueError, match="fixed Qwen3 architecture"):
        kev_artifact.export(args)


def test_export_rejects_missing_structural_token(synthetic_export, monkeypatch):
    args, _, _ = synthetic_export
    tokenizer = FakeTokenizer()
    del tokenizer.vocab[STRUCTURAL_TOKENS[-1]]
    monkeypatch.setattr(
        kev_artifact.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: tokenizer,
    )

    with pytest.raises(ValueError, match="distinct KEV structural tokens"):
        kev_artifact.export(args)


def test_force_export_refuses_to_delete_input_directory(synthetic_export):
    args, _, _ = synthetic_export
    args.output = args.base
    args.force = True

    with pytest.raises(ValueError, match="overlaps an input path"):
        kev_artifact.export(args)
    assert args.base.joinpath("model.safetensors").is_file()


def test_failed_force_export_preserves_existing_artifact(synthetic_export, monkeypatch):
    args, _, _ = synthetic_export
    args.output.mkdir()
    marker = args.output / "existing-artifact"
    marker.write_text("keep")
    args.force = True
    tokenizer = FakeTokenizer()
    del tokenizer.vocab[STRUCTURAL_TOKENS[-1]]
    monkeypatch.setattr(
        kev_artifact.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: tokenizer,
    )

    with pytest.raises(ValueError, match="distinct KEV structural tokens"):
        kev_artifact.export(args)
    assert marker.read_text() == "keep"


def test_validate_rejects_unverified_weight_file(synthetic_export):
    args, _, _ = synthetic_export
    kev_artifact.export(args)
    manifest_path = args.output / "kev_manifest.json"
    manifest = kev_artifact.read_json(manifest_path)
    del manifest["files_sha256"]["model.safetensors"]
    kev_artifact.write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="checksum inventory"):
        kev_artifact.validate_artifact(args.output)


def test_validate_rejects_missing_head_parameter(synthetic_export):
    args, _, _ = synthetic_export
    kev_artifact.export(args)
    head_path = args.output / "kev-head.safetensors"
    with safe_open(head_path, framework="pt") as source:
        tensors = {
            name: source.get_tensor(name)
            for name in source.keys()  # noqa: SIM118 - safe_open is not iterable
            if name != "pooler.q.bias"
        }
    save_file(tensors, head_path)
    manifest_path = args.output / "kev_manifest.json"
    manifest = kev_artifact.read_json(manifest_path)
    manifest["files_sha256"][head_path.name] = kev_artifact.sha256(head_path)
    kev_artifact.write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="weight index does not match shard"):
        kev_artifact.validate_artifact(args.output)
