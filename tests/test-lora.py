#!/usr/bin/env python3
# test-lora.py: torch reference for the LoRA delta path.
#
# Builds a small random adapter over the projections named on the command line,
# converts it, runs the engine harness with and without it, and compares the
# engine's LoRA delta against peft's.
#
# The comparison is on the delta, not on the logits: subtracting each side's
# own base isolates the contribution this code is responsible for.
#
# Run it against the BF16 backbone. Against Q8_0 the base logits sit over a
# percent away from a torch fp32 forward, and a perturbation injected into 28
# layers diverges by about as much again, which buries the thing being measured.
# BF16 halves that, and the assertion is then on the delta's direction and
# magnitude rather than its absolute agreement - see the note by the checks at
# the bottom for why that is the discriminating test.

import argparse
import os
import subprocess
import sys

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ENGINE_DIR)
sys.path.insert(0, SCRIPT_DIR)


def engine_logits(harness, gguf, adapter, scale, prefix, ids):
    subprocess.run([harness, gguf, adapter, str(scale), prefix] + [str(i) for i in ids],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return np.fromfile(prefix + ".bin", dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=os.path.join(ENGINE_DIR, "checkpoints", "YuE2-3B"))
    p.add_argument("--gguf", required=True, help="the converted backbone GGUF")
    p.add_argument("--harness", required=True, help="path to test-lora[.exe]")
    p.add_argument("--targets", nargs="+", default=["q_proj", "v_proj"])
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--alpha", type=float, default=8.0)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--ids", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--workdir", default=os.path.join(ENGINE_DIR, "build", "loraref"))
    p.add_argument("--b-std", type=float, default=0.1, help="std of the randomised B")
    p.add_argument("--tol", type=float, default=0.05, help="tolerance on the delta magnitude ratio")
    p.add_argument("--min-cos", type=float, default=0.995, help="minimum cosine with the reference delta")
    args = p.parse_args()

    os.makedirs(args.workdir, exist_ok=True)

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, dtype=torch.float32, trust_remote_code=True)
    model.eval()
    cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha,
                     target_modules=args.targets, lora_dropout=0.0, bias="none")
    peft_model = get_peft_model(model, cfg)

    # peft initialises B to zero, which would make the delta vanish and the
    # comparison vacuous. Randomise it.
    #
    # Only on the AR half. peft matches target_modules by suffix, so "o_proj"
    # also catches nar_self_attn.o_proj, and the reference would then carry NAR
    # deltas that the causal forward may or may not use. The engine's LM binds
    # model.layers.N.self_attn.* and nothing else, so the reference has to be
    # held to the same set. The NAR pairs still travel into the GGUF, as zeros.
    for name, param in peft_model.named_parameters():
        if "lora_B" in name and "nar_" not in name:
            torch.nn.init.normal_(param, std=args.b_std)

    peft_model.save_pretrained(args.workdir, safe_serialization=True)
    adapter_dir = args.workdir
    if not os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        adapter_dir = os.path.join(args.workdir, "default")

    out_gguf = os.path.join(args.workdir, "adapter.gguf")
    convert_lora = __import__("convert-lora")
    convert_lora.convert_lora(adapter_dir, out_gguf)

    ids = torch.tensor([args.ids], dtype=torch.long)
    with torch.no_grad():
        adapted = peft_model(ids).logits[0, -1].float().numpy()
        with peft_model.disable_adapter():
            base = peft_model(ids).logits[0, -1].float().numpy()
    torch_delta = (adapted - base) * args.scale

    eng_base = engine_logits(args.harness, args.gguf, "none", 1.0,
                             os.path.join(args.workdir, "engine_base"), args.ids)
    eng_adapted = engine_logits(args.harness, args.gguf, out_gguf, args.scale,
                                os.path.join(args.workdir, "engine_lora"), args.ids)
    engine_delta = eng_adapted - eng_base

    ref_norm = np.linalg.norm(torch_delta)
    eng_norm = np.linalg.norm(engine_delta)

    # The engine's own numerics - an f16 KV cache, flash attention, ggml's
    # accumulation order - put its base logits about half a percent away from
    # a torch fp32 forward, and a perturbation injected into 28 layers diverges
    # by a comparable amount. That is a floor on the absolute agreement, and it
    # does not shrink as the delta grows. So the assertion is on direction and
    # magnitude, which a real defect destroys and numerical noise does not: a
    # wrong row offset, a missed transpose or a dropped alpha/r all send the
    # cosine far from 1, while the floor only blunts it in the fourth decimal.
    base_gap = np.linalg.norm(eng_base - base)
    cos = float(np.dot(engine_delta, torch_delta) / max(eng_norm * ref_norm, 1e-12))
    ratio = eng_norm / max(ref_norm, 1e-12)
    rel = np.linalg.norm(engine_delta - torch_delta) / max(ref_norm, 1e-12)

    print("targets      : %s" % " ".join(args.targets))
    print("base gap     : %.3f  (engine vs torch, the noise floor)" % base_gap)
    print("delta norm   : torch %.3f, engine %.3f  (ratio %.4f)" % (ref_norm, eng_norm, ratio))
    print("cosine       : %.6f" % cos)
    print("rel error    : %.3f%%" % (100.0 * rel))

    if ref_norm < 1e-6:
        print("FAIL: the reference delta is zero, the comparison proves nothing")
        return 1
    if ref_norm < 4.0 * base_gap:
        print("FAIL: delta is not clear of the noise floor; raise --b-std")
        return 1
    if cos < args.min_cos:
        print("FAIL: cosine %.6f below %.6f - the delta points the wrong way" % (cos, args.min_cos))
        return 1
    if abs(ratio - 1.0) > args.tol:
        print("FAIL: delta magnitude off by %.2f%%" % (100.0 * abs(ratio - 1.0)))
        return 1

    # The harness rebound the same adapter at half strength and decoded the
    # same token again, from the same cache at the same position, so every key
    # of the batched decode graph was unchanged.
    #
    # The assertion is exact non-identity, and only that. A cache that ignores
    # the binding replays the graph it already built and returns the
    # full-strength logits bit for bit; the engine is deterministic, so any
    # difference at all means the graph was rebuilt.
    #
    # The magnitude is deliberately not asserted. Only the last token is
    # re-decoded, while the prefilled prefix stays in the cache at full
    # strength, so the resulting logit delta is not a monotonic function of the
    # last token's strength - it grows for some target sets and shrinks for
    # others. Anything stronger than non-identity here would be a guess.
    half = np.fromfile(os.path.join(args.workdir, "engine_lora.half.bin"), dtype=np.float32)
    print("half strength: bit-identical %s, delta ratio %.4f"
          % (np.array_equal(half, eng_adapted),
             np.linalg.norm(half - eng_base) / max(eng_norm, 1e-12)))
    if np.array_equal(half, eng_adapted):
        print("FAIL: rebinding at half strength changed nothing - stale graph cache")
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
