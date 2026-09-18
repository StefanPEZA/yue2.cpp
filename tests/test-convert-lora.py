# test-convert-lora.py: convert-lora.py parity and rejection harness.
# Builds synthetic PEFT adapters on disk, converts them, reads the GGUF back.

import json
import os
import struct
import sys

import numpy as np
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ENGINE_DIR)

import gguf

convert_lora = __import__("convert-lora")


def write_safetensors(path, tensors):
    """Minimal F32 safetensors writer, mirroring convert.py's reader."""
    header, offset, blobs = {}, 0, []
    for name in sorted(tensors):
        arr = np.ascontiguousarray(tensors[name], dtype=np.float32)
        blobs.append(arr.tobytes())
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(blobs[-1])],
        }
        offset += len(blobs[-1])
    raw = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for b in blobs:
            f.write(b)


def make_adapter(tmp_path, tensors, **config):
    cfg = {"r": 4, "lora_alpha": 8, "peft_type": "LORA",
           "target_modules": ["q_proj"], "use_rslora": False, "use_dora": False}
    cfg.update(config)
    with open(os.path.join(str(tmp_path), "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    write_safetensors(os.path.join(str(tmp_path), "adapter_model.safetensors"), tensors)
    return str(tmp_path)


def read_gguf(path):
    reader = gguf.GGUFReader(path)
    return {t.name: t for t in reader.tensors}, reader


def test_names_map_onto_base_tensor_names(tmp_path):
    a = np.arange(4 * 8, dtype=np.float32).reshape(4, 8)
    b = np.arange(8 * 4, dtype=np.float32).reshape(8, 4)
    src = make_adapter(tmp_path, {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": a,
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": b,
    })
    out = str(tmp_path / "out.gguf")
    convert_lora.convert_lora(src, out)

    tensors, _ = read_gguf(out)
    assert "model.layers.0.self_attn.q_proj.weight.lora_a" in tensors
    assert "model.layers.0.self_attn.q_proj.weight.lora_b" in tensors


def test_nar_targets_survive(tmp_path):
    a = np.zeros((4, 8), dtype=np.float32)
    b = np.zeros((8, 4), dtype=np.float32)
    src = make_adapter(tmp_path, {
        "base_model.model.model.layers.3.nar_self_attn.v_proj.lora_A.weight": a,
        "base_model.model.model.layers.3.nar_self_attn.v_proj.lora_B.weight": b,
    })
    out = str(tmp_path / "out.gguf")
    convert_lora.convert_lora(src, out)

    tensors, _ = read_gguf(out)
    assert "model.layers.3.nar_self_attn.v_proj.weight.lora_a" in tensors


def test_alpha_over_r_is_folded_into_b(tmp_path):
    a = np.ones((4, 8), dtype=np.float32)
    b = np.ones((8, 4), dtype=np.float32)
    src = make_adapter(tmp_path, {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": a,
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": b,
    }, r=4, lora_alpha=8)
    out = str(tmp_path / "out.gguf")
    convert_lora.convert_lora(src, out)

    tensors, _ = read_gguf(out)
    got_a = np.array(tensors["model.layers.0.self_attn.q_proj.weight.lora_a"].data)
    got_b = np.array(tensors["model.layers.0.self_attn.q_proj.weight.lora_b"].data)
    # alpha/r == 2.0 lands in B; A is untouched.
    assert np.allclose(got_a.astype(np.float32), 1.0)
    assert np.allclose(got_b.astype(np.float32), 2.0)


def test_rslora_uses_sqrt_r(tmp_path):
    a = np.ones((4, 8), dtype=np.float32)
    b = np.ones((8, 4), dtype=np.float32)
    src = make_adapter(tmp_path, {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": a,
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": b,
    }, r=4, lora_alpha=8, use_rslora=True)
    out = str(tmp_path / "out.gguf")
    convert_lora.convert_lora(src, out)

    tensors, _ = read_gguf(out)
    got_b = np.array(tensors["model.layers.0.self_attn.q_proj.weight.lora_b"].data)
    assert np.allclose(got_b.astype(np.float32), 8.0 / np.sqrt(4.0), atol=1e-3)


def test_rank_is_recorded(tmp_path):
    a = np.zeros((4, 8), dtype=np.float32)
    b = np.zeros((8, 4), dtype=np.float32)
    src = make_adapter(tmp_path, {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": a,
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": b,
    }, r=4)
    out = str(tmp_path / "out.gguf")
    convert_lora.convert_lora(src, out)

    _, reader = read_gguf(out)
    field = reader.get_field("yue2-lora.rank")
    assert int(field.parts[field.data[0]][0]) == 4


@pytest.mark.parametrize("config,message", [
    ({"use_dora": True}, "DoRA"),
    ({"modules_to_save": ["norm"]}, "modules_to_save"),
])
def test_unsupported_configs_are_rejected(tmp_path, config, message):
    a = np.zeros((4, 8), dtype=np.float32)
    b = np.zeros((8, 4), dtype=np.float32)
    src = make_adapter(tmp_path, {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": a,
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": b,
    }, **config)
    with pytest.raises(SystemExit, match=message):
        convert_lora.convert_lora(src, str(tmp_path / "out.gguf"))


def test_head_targets_are_rejected(tmp_path):
    a = np.zeros((4, 8), dtype=np.float32)
    b = np.zeros((8, 4), dtype=np.float32)
    src = make_adapter(tmp_path, {
        "base_model.model.lm_head.lora_A.weight": a,
        "base_model.model.lm_head.lora_B.weight": b,
    })
    with pytest.raises(SystemExit, match="lm_head"):
        convert_lora.convert_lora(src, str(tmp_path / "out.gguf"))


def test_unpaired_tensor_is_rejected(tmp_path):
    a = np.zeros((4, 8), dtype=np.float32)
    src = make_adapter(tmp_path, {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": a,
    })
    with pytest.raises(SystemExit, match="lora_B"):
        convert_lora.convert_lora(src, str(tmp_path / "out.gguf"))
