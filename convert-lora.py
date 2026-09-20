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
import collections
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

# Weights an adapter may replace outright rather than decorate with a delta.
# The NAR-branch adapters retrain the VAE/LLM projections whole, and ship them
# beside the pairs. Deliberately a whitelist: a replacement the engine has no
# override path for must still fail as an unexpected key.
FULL_WEIGHTS = ("vae2llm.weight", "vae2llm.bias", "llm2vae.weight", "llm2vae.bias")
FULL_SUFFIX = ".full"

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
    """A/B keys grouped by the base tensor they target, plus whole-weight keys.

    Everything the engine has no delta path for is rejected here, before a single
    byte of tensor data is read.
    """
    pairs, fulls = {}, {}
    for key in sorted(meta):
        if key in FULL_WEIGHTS:
            fulls[key] = key
            continue
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
    return pairs, fulls

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

Adapter = collections.namedtuple(
    "Adapter", "src sf_path meta data_start pairs fulls scale rank")

def peft_adapter(adapter_dir):
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    check_config(cfg)
    scale, rank = scaling(cfg)

    sf_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.exists(sf_path):
        raise SystemExit("no adapter_model.safetensors in %s" % adapter_dir)
    meta, data_start = read_sf_header(sf_path)
    pairs, fulls = group_pairs(meta, base_name, A_SUFFIX)
    return Adapter(adapter_dir, sf_path, meta, data_start, pairs, fulls, scale, rank)

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
    meta, data_start = read_sf_header(sf_path)
    if any(key.startswith(PEFT_PREFIX) for key in meta):
        raise SystemExit("%s is a PEFT adapter; point at its directory instead, so "
                         "adapter_config.json is read with it" % sf_path)
    pairs, fulls = group_pairs(meta, bare_base_name, BARE_A_SUFFIX)
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
    return Adapter(sf_path, sf_path, meta, data_start, pairs, fulls, scale, rank)

def read_adapter(src):
    return (peft_adapter if os.path.isdir(src) else bare_adapter)(src)

def read_tensor(f, ad, key):
    """One tensor out of the adapter, F32 only.

    convert.py names the dtype it cannot handle; without the same check here a
    BF16 adapter - what most exporters write - dies inside numpy's reshape.
    """
    t = ad.meta[key]
    if t["dtype"] != "F32":
        raise SystemExit("unexpected dtype %s for %s in %s"
                         % (t["dtype"], key, ad.sf_path))
    f.seek(ad.data_start + t["data_offsets"][0])
    raw = f.read(t["data_offsets"][1] - t["data_offsets"][0])
    return np.frombuffer(raw, dtype=np.float32).reshape(t["shape"])

def merge_targets(adapters):
    """Every base tensor the adapters write, rejecting any two that collide.

    Last-wins would silently drop half of one adapter, which is exactly the
    quiet half-application the engine's load-time target check exists to stop.
    """
    seen = {}
    for ad in adapters:
        for name in list(ad.pairs) + list(ad.fulls):
            if name in seen:
                raise SystemExit("%s and %s both target %s"
                                 % (adapter_stem(seen[name]), adapter_stem(ad.src), name))
            seen[name] = ad.src
    return seen

def convert_lora(src, out_path, dtype="f16"):
    """One or more PEFT directories / bare adapter files -> out_path.

    Several sources merge into one GGUF: the AR and NAR halves of a pairing are
    bound from a single LoraSet, so the engine needs them in a single file.
    """
    np_type, raw_type = DTYPES[dtype]
    sources = [src] if isinstance(src, str) else list(src)
    adapters = [read_adapter(s) for s in sources]
    merge_targets(adapters)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    w = gguf.GGUFWriter(out_path, arch="yue2-lora")
    w.add_name("+".join(adapter_stem(s) for s in sources))
    # Advisory, and only meaningful when the sources agree - a merged AR/NAR
    # pairing routinely mixes ranks. lora.h reads a missing key as 0.
    ranks = {ad.rank for ad in adapters}
    if len(ranks) == 1:
        w.add_uint32("yue2-lora.rank", ranks.pop())

    n_pairs = n_fulls = 0
    for ad in adapters:
        with open(ad.sf_path, "rb") as f:
            for base in sorted(ad.pairs):
                for slot in ("a", "b"):
                    arr = read_tensor(f, ad, ad.pairs[base][slot])
                    # The strength lives in B so the engine multiplies once.
                    if slot == "b":
                        arr = arr * ad.scale
                    w.add_tensor("%s.lora_%s" % (base, slot),
                                 np.ascontiguousarray(arr, dtype=np_type),
                                 raw_dtype=raw_type)
                n_pairs += 1
            # A replacement weight is not a delta: alpha/r does not apply to it.
            for base in sorted(ad.fulls):
                arr = read_tensor(f, ad, ad.fulls[base])
                w.add_tensor(base + FULL_SUFFIX,
                             np.ascontiguousarray(arr, dtype=np_type),
                             raw_dtype=raw_type)
                n_fulls += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    log("lora", "wrote %s (%d pairs, %d replacements, r=%s, %s)"
        % (out_path, n_pairs, n_fulls,
           ",".join(str(ad.rank) for ad in adapters), dtype))

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Convert a LoRA adapter for the YuE2 backbone to GGUF.")
    p.add_argument("adapter", nargs="+",
                   help="a PEFT directory holding adapter_config.json and "
                        "adapter_model.safetensors, or a bare adapter .safetensors. "
                        "Several merge into one GGUF, which is how an AR and a NAR "
                        "adapter are applied together")
    p.add_argument("--outfile",
                   help="where to write; the adapter's own name with .gguf by "
                        "default, required when merging")
    p.add_argument("--dtype", choices=sorted(DTYPES), default="f16",
                   help="tensor type to emit (default f16)")
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    if not args.outfile and len(args.adapter) > 1:
        raise SystemExit("--outfile is required when merging several adapters")
    out = args.outfile or default_outfile(args.adapter[0])
    convert_lora(args.adapter, out, args.dtype)
    return 0

if __name__ == "__main__":
    sys.exit(main())
