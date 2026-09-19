#!/usr/bin/env python3
# Convert a LoRA adapter to GGUF for yue2.cpp.
#
# The backbone conversion streams its tensors unrenamed, so an engine tensor
# name is the HuggingFace name. A PEFT key is therefore the base tensor name
# wrapped in a prefix and a suffix, and mapping one to the other is string
# surgery with no table:
#
#   base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
#   -> model.layers.0.self_attn.q_proj.weight.lora_a
#
# That also carries the nar_* weight set with no extra cases.
#
# A and B keep PEFT's own shapes, (r, in) and (out, r). gguf reverses the axes
# on write, so ggml reads A as [in, r] and B as [r, out], which is what
# ggml_mul_mat wants. Nothing is transposed anywhere in the pipeline.
#
# PEFT's alpha/r scaling is folded into B here so the engine applies only the
# strength the user asked for.
#
# A bare adapter - a lone safetensors with no config beside it - is the same
# surgery under a different wrapper:
#
#   layers.0.self_attn.q_proj.lora_A
#   -> model.layers.0.self_attn.q_proj.weight.lora_a
#
# It states its rank in A's own shape, and its exporter has normally already
# folded alpha into B, so the scale is 1.0 unless __metadata__ says otherwise.

import os
import sys
import json
import struct
import argparse
import numpy as np
import gguf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from convert import read_sf_header, log

PEFT_PREFIX = "base_model.model."
A_SUFFIX = ".lora_A.weight"
B_SUFFIX = ".lora_B.weight"

# A bare adapter is a single safetensors with no config beside it: keys are
# already rooted at the layer list and carry no ".weight", so the mapping is
# the PEFT one with the prefix and suffix swapped.
BARE_PREFIX = "layers."
BARE_A_SUFFIX = ".lora_A"
BARE_B_SUFFIX = ".lora_B"

# Targets the engine cannot apply a delta to. The LM head is read through a
# row window (qw3lm_head_rows), so a delta on it needs its own design.
REJECTED_TARGETS = ("embed_tokens", "lm_head")

def base_name(peft_key):
    """PEFT key -> engine base tensor name, or None if it is not an A/B key."""
    if not peft_key.startswith(PEFT_PREFIX):
        return None
    stem = peft_key[len(PEFT_PREFIX):]
    for suffix in (A_SUFFIX, B_SUFFIX):
        if stem.endswith(suffix):
            return stem[: -len(suffix)] + ".weight"
    return None

# F16 halves the adapter's VRAM and sits far below the backbone's own
# quantization error; F32 is there for parity work against peft.
DTYPES = {
    "f16": (np.float16, gguf.GGMLQuantizationType.F16),
    "f32": (np.float32, gguf.GGMLQuantizationType.F32),
}

def adapter_stem(src):
    """The adapter's name: a PEFT directory's basename, a bare file's without its extension."""
    stem = os.path.basename(os.path.normpath(src))
    if stem.endswith(".safetensors"):
        stem = stem[: -len(".safetensors")]
    return stem

def default_outfile(src):
    return os.path.join(os.path.dirname(os.path.normpath(src)), adapter_stem(src) + ".gguf")

def bare_base_name(key):
    """Bare adapter key -> engine base tensor name, or None if it is not an A/B key."""
    if not key.startswith(BARE_PREFIX):
        return None
    for suffix in (BARE_A_SUFFIX, BARE_B_SUFFIX):
        if key.endswith(suffix):
            return "model." + key[: -len(suffix)] + ".weight"
    return None

def group_pairs(meta, to_base, a_suffix):
    """A/B keys grouped by the base tensor they target.

    Everything the engine has no delta path for is rejected here, before a single
    byte of tensor data is read.
    """
    pairs = {}
    for key in sorted(meta):
        base = to_base(key)
        if base is None:
            raise SystemExit("unexpected key in adapter: %s" % key)
        for bad in REJECTED_TARGETS:
            if bad in base:
                raise SystemExit("adapters on %s are not supported (%s)" % (bad, key))
        slot = "a" if key.endswith(a_suffix) else "b"
        pairs.setdefault(base, {})[slot] = key

    if not pairs:
        raise SystemExit("adapter contains no LoRA pairs")
    for base in sorted(pairs):
        slots = pairs[base]
        if "a" not in slots:
            raise SystemExit("%s has a lora_B without a lora_A" % base)
        if "b" not in slots:
            raise SystemExit("%s has a lora_A without a lora_B" % base)
    return pairs

def check_config(cfg):
    if cfg.get("use_dora"):
        raise SystemExit("DoRA adapters are not supported: the magnitude vector "
                         "has no equivalent in the engine's delta path")
    if cfg.get("modules_to_save"):
        raise SystemExit("modules_to_save is not supported: %s are full "
                         "replacement tensors, not deltas"
                         % ", ".join(cfg["modules_to_save"]))

def scaling(cfg):
    r = int(cfg["r"])
    alpha = float(cfg["lora_alpha"])
    if r <= 0:
        raise SystemExit("adapter_config.json has a non-positive r")
    return (alpha / np.sqrt(r)) if cfg.get("use_rslora") else (alpha / r), r

def peft_adapter(adapter_dir):
    """PEFT directory -> (sf_path, meta, data_start, pairs, scale, rank)."""
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    check_config(cfg)
    scale, rank = scaling(cfg)

    sf_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.exists(sf_path):
        raise SystemExit("no adapter_model.safetensors in %s" % adapter_dir)
    meta, data_start = read_sf_header(sf_path)
    return sf_path, meta, data_start, group_pairs(meta, base_name, A_SUFFIX), scale, rank

def md_flag(md, key):
    """A metadata flag. Values are always strings here, so "false" must not read as true."""
    return str(md.get(key, "")).strip().lower() in ("1", "true", "yes")

def read_sf_metadata(path):
    """The __metadata__ block read_sf_header drops; {} when the file has none."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header.get("__metadata__") or {}

def bare_adapter(sf_path):
    """Bare adapter file -> (sf_path, meta, data_start, pairs, scale, rank)."""
    meta, data_start = read_sf_header(sf_path)
    if any(key.startswith(PEFT_PREFIX) for key in meta):
        raise SystemExit("%s is a PEFT adapter; point at its directory instead, so "
                         "adapter_config.json is read with it" % sf_path)
    pairs = group_pairs(meta, bare_base_name, BARE_A_SUFFIX)
    # A is (r, in), so the tensors state the rank with no config to consult.
    rank = int(meta[pairs[min(pairs)]["a"]]["shape"][0])

    md = read_sf_metadata(sf_path)
    check_config({"use_dora": md_flag(md, "use_dora"),
                  "modules_to_save": md.get("modules_to_save")})
    if "rank" in md and int(md["rank"]) != rank:
        raise SystemExit("metadata claims rank %d, but lora_A is rank %d"
                         % (int(md["rank"]), rank))

    # No alpha means the exporter already folded it into B, which is the whole
    # reason a bare adapter can travel without a config. An alpha that is there
    # is folded through the same path PEFT takes, rslora included.
    alpha = md.get("lora_alpha", md.get("alpha"))
    if alpha is None:
        scale = 1.0
    else:
        scale, _ = scaling({"r": rank, "lora_alpha": alpha,
                            "use_rslora": md_flag(md, "use_rslora")})
    return sf_path, meta, data_start, pairs, scale, rank

def convert_lora(src, out_path, dtype="f16"):
    """A PEFT directory or a bare adapter file -> out_path, A/B keyed by base name."""
    np_type, raw_type = DTYPES[dtype]
    read = peft_adapter if os.path.isdir(src) else bare_adapter
    sf_path, meta, data_start, pairs, scale, rank = read(src)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    w = gguf.GGUFWriter(out_path, arch="yue2-lora")
    w.add_name(adapter_stem(src))
    w.add_uint32("yue2-lora.rank", rank)

    with open(sf_path, "rb") as f:
        for base in sorted(pairs):
            slots = pairs[base]
            for slot in ("a", "b"):
                t = meta[slots[slot]]
                f.seek(data_start + t["data_offsets"][0])
                raw = f.read(t["data_offsets"][1] - t["data_offsets"][0])
                arr = np.frombuffer(raw, dtype=np.float32).reshape(t["shape"])
                # The strength lives in B so the engine multiplies once.
                if slot == "b":
                    arr = arr * scale
                w.add_tensor("%s.lora_%s" % (base, slot),
                             np.ascontiguousarray(arr, dtype=np_type),
                             raw_dtype=raw_type)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    log("lora", "wrote %s (%d pairs, r=%d, scale=%.4f, %s)"
        % (out_path, len(pairs), rank, scale, dtype))

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Convert a LoRA adapter for the YuE2 backbone to GGUF.")
    p.add_argument("adapter",
                   help="a PEFT directory holding adapter_config.json and "
                        "adapter_model.safetensors, or a bare adapter .safetensors")
    p.add_argument("--outfile",
                   help="where to write; the adapter's own name with .gguf by default")
    p.add_argument("--dtype", choices=sorted(DTYPES), default="f16",
                   help="tensor type to emit (default f16)")
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    out = args.outfile or default_outfile(args.adapter)
    convert_lora(args.adapter, out, args.dtype)
    return 0

if __name__ == "__main__":
    sys.exit(main())
