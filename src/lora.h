#pragma once
// lora.h: LoRA adapter weights for the Qwen3 stacks
//
// An adapter GGUF written by convert-lora.py holds one A/B pair per base
// tensor it targets, named "<base tensor name>.lora_a" and ".lora_b". A
// LoraSet is that file resident on a backend, looked up by base tensor name.
//
// The set is keyed by its own path, not by the module it decorates: the AR
// half and the NAR half read pairs out of the same set, and it outlives the
// eviction of either, the way the KV cache does.
//
// Usage:
//   LoraSet set;
//   if (!lora_load(&set, "style.gguf", "YuE2-3B-Q8_0.gguf")) { error; }
//   LoraPair p = lora_find(&set, "model.layers.0.self_attn.q_proj.weight");
//   lora_free(&set);

#include "backend.h"
#include "gguf-weights.h"
#include "weight-ctx.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>

#define LORA_A_SUFFIX ".lora_a"
#define LORA_B_SUFFIX ".lora_b"

// One delta factorisation. a is [in, r], b is [r, out]: the shapes PEFT
// already stores, so B(Ax) needs no transpose on either side.
struct LoraPair {
    struct ggml_tensor * a = nullptr;
    struct ggml_tensor * b = nullptr;
};

struct LoraSet {
    std::string    path;
    int            rank = 0;
    WeightCtx      wctx;
    ggml_backend_t backend     = nullptr;
    ggml_backend_t cpu_backend = nullptr;

    std::unordered_map<std::string, LoraPair> pairs;  // keyed by base tensor name
};

static void lora_free(LoraSet * s) {
    wctx_free(&s->wctx);
    if (s->backend) {
        backend_release(s->backend, s->cpu_backend);
    }
    s->pairs.clear();
    s->backend     = nullptr;
    s->cpu_backend = nullptr;
    s->rank        = 0;
}

// Every pair must name a tensor the backbone actually has. A typo'd or
// mismatched adapter has to fail loudly here: applying the half that matched
// would quietly produce something between the two models.
static bool lora_check_targets(const LoraSet * s, const char * base_path) {
    GGUFModel base;
    if (!gf_load(&base, base_path)) {
        return false;
    }
    bool ok = true;
    for (const auto & kv : s->pairs) {
        if (gguf_find_tensor(base.gguf, kv.first.c_str()) < 0) {
            fprintf(stderr, "[LoRA] FATAL: adapter targets '%s', which %s does not have\n", kv.first.c_str(),
                    base_path);
            ok = false;
        }
    }
    gf_close(&base);
    return ok;
}

static bool lora_ends_with(const std::string & name, const char * suffix) {
    size_t n = strlen(suffix);
    return name.size() > n && name.compare(name.size() - n, n, suffix) == 0;
}

static bool lora_load(LoraSet * s, const char * lora_path, const char * base_path) {
    GGUFModel gf;
    if (!gf_load(&gf, lora_path)) {
        return false;
    }

    s->path        = lora_path;
    BackendPair bp = backend_init("LoRA");
    s->backend     = bp.backend;
    s->cpu_backend = bp.cpu_backend;

    int64_t n = gguf_get_n_tensors(gf.gguf);
    wctx_init(&s->wctx, (int) n);

    int64_t rank_key = gguf_find_key(gf.gguf, "yue2-lora.rank");
    s->rank          = rank_key < 0 ? 0 : (int) gguf_get_val_u32(gf.gguf, rank_key);

    for (int64_t i = 0; i < n; i++) {
        std::string name = gguf_get_tensor_name(gf.gguf, i);
        bool        is_a = lora_ends_with(name, LORA_A_SUFFIX);
        bool        is_b = lora_ends_with(name, LORA_B_SUFFIX);
        if (!is_a && !is_b) {
            fprintf(stderr, "[LoRA] FATAL: '%s' is neither a lora_a nor a lora_b\n", name.c_str());
            gf_close(&gf);
            lora_free(s);
            return false;
        }
        std::string          base = name.substr(0, name.size() - strlen(is_a ? LORA_A_SUFFIX : LORA_B_SUFFIX));
        struct ggml_tensor * t    = gf_load_tensor(&s->wctx, gf, name);
        if (is_a) {
            s->pairs[base].a = t;
        } else {
            s->pairs[base].b = t;
        }
    }

    for (const auto & kv : s->pairs) {
        if (!kv.second.a || !kv.second.b) {
            fprintf(stderr, "[LoRA] FATAL: '%s' is missing its %s\n", kv.first.c_str(),
                    kv.second.a ? "lora_b" : "lora_a");
            gf_close(&gf);
            lora_free(s);
            return false;
        }
    }

    if (!lora_check_targets(s, base_path)) {
        gf_close(&gf);
        lora_free(s);
        return false;
    }

    if (!wctx_alloc(&s->wctx, s->backend)) {
        gf_close(&gf);
        lora_free(s);
        return false;
    }
    gf_close(&gf);

    fprintf(stderr, "[LoRA] %s: %zu pairs, rank %d\n", lora_path, s->pairs.size(), s->rank);
    return true;
}

// Zeroed pair when the adapter does not cover this tensor: a partial adapter
// (the usual q_proj + v_proj case) is normal, not an error.
static LoraPair lora_find(const LoraSet * s, const std::string & base_name) {
    if (!s) {
        return {};
    }
    auto it = s->pairs.find(base_name);
    return it == s->pairs.end() ? LoraPair{} : it->second;
}

static size_t lora_bytes(const LoraSet * s) {
    return s && s->wctx.buffer ? ggml_backend_buffer_get_size(s->wctx.buffer) : 0;
}
