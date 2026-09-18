#!/usr/bin/env python3
# Convert a PEFT LoRA adapter to GGUF for yue2.cpp.
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

import os
import sys
import json
import argparse
import numpy as np
import gguf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from convert import read_sf_header, log

PEFT_PREFIX = "base_model.model."
A_SUFFIX = ".lora_A.weight"
B_SUFFIX = ".lora_B.weight"

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

def convert_lora(adapter_dir, out_path):
    """adapter_dir (PEFT) -> out_path (yue2-lora GGUF), A/B keyed by base name."""
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    check_config(cfg)
    scale, rank = scaling(cfg)

    sf_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.exists(sf_path):
        raise SystemExit("no adapter_model.safetensors in %s" % adapter_dir)
    meta, data_start = read_sf_header(sf_path)

    # Group the A/B keys by the base tensor they target, rejecting anything
    # the engine has no delta path for before reading a single byte.
    pairs = {}
    for key in sorted(meta):
        base = base_name(key)
        if base is None:
            raise SystemExit("unexpected key in adapter: %s" % key)
        for bad in REJECTED_TARGETS:
            if bad in base:
                raise SystemExit("adapters on %s are not supported (%s)" % (bad, key))
        slot = "a" if key.endswith(A_SUFFIX) else "b"
        pairs.setdefault(base, {})[slot] = key

    if not pairs:
        raise SystemExit("adapter contains no LoRA pairs")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    w = gguf.GGUFWriter(out_path, arch="yue2-lora")
    w.add_name(os.path.basename(os.path.normpath(adapter_dir)))
    w.add_uint32("yue2-lora.rank", rank)

    F16 = gguf.GGMLQuantizationType.F16
    with open(sf_path, "rb") as f:
        for base in sorted(pairs):
            slots = pairs[base]
            if "a" not in slots:
                raise SystemExit("%s has a lora_B without a lora_A" % base)
            if "b" not in slots:
                raise SystemExit("%s has a lora_A without a lora_B" % base)
            for slot in ("a", "b"):
                t = meta[slots[slot]]
                f.seek(data_start + t["data_offsets"][0])
                raw = f.read(t["data_offsets"][1] - t["data_offsets"][0])
                arr = np.frombuffer(raw, dtype=np.float32).reshape(t["shape"])
                # The strength lives in B so the engine multiplies once.
                if slot == "b":
                    arr = arr * scale
                w.add_tensor("%s.lora_%s" % (base, slot),
                             np.ascontiguousarray(arr, dtype=np.float16),
                             raw_dtype=F16)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    log("lora", "wrote %s (%d pairs, r=%d, scale=%.4f)"
        % (out_path, len(pairs), rank, scale))

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Convert a PEFT LoRA adapter for the YuE2 backbone to GGUF.")
    p.add_argument("adapter_dir",
                   help="directory holding adapter_config.json and "
                        "adapter_model.safetensors")
    p.add_argument("--outfile",
                   help="where to write; <adapter_dir>.gguf by default")
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    out = args.outfile or (os.path.normpath(args.adapter_dir) + ".gguf")
    convert_lora(args.adapter_dir, out)
    return 0

if __name__ == "__main__":
    sys.exit(main())
