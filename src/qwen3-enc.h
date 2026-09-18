// qwen3-enc.h: Qwen3 transformer encoder via ggml
//
// Generic Qwen3 backbone used by:
//   YuE2 AR path: 28L, H=2048, causal, GQA 16/8, vocab lookup
//
// Architecture per layer:
//   RMSNorm -> Q/K/V proj -> QK-Norm -> RoPE -> GQA -> O proj -> +residual
//   RMSNorm -> gate/up proj -> SwiGLU -> down proj -> +residual
// Final: RMSNorm

#pragma once
#include "backend.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf-weights.h"
#include "lora.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#define QWEN3_MAX_LAYERS 32

// Config
struct Qwen3Config {
    int   hidden_size;        // H
    int   intermediate_size;  // FFN inner dim
    int   n_heads;            // Nh (query heads)
    int   n_kv_heads;         // Nkv (key/value heads, for GQA)
    int   head_dim;           // D = H / Nh
    int   n_layers;
    float rope_theta;
    float rms_norm_eps;
    bool  is_causal;  // true for text encoder, false for lyric/timbre
};

// The LoRA contributions to one weight slot. A fused weight holds up to
// three, each writing the rows of the constituent it belongs to; row0 is
// where those rows start in the fused output. A part with a null pair is an
// adapter that does not cover that constituent, which is the usual case:
// PEFT commonly targets q_proj and v_proj and leaves k_proj alone.
struct LoraDelta {
    LoraPair part[3];
    int64_t  row0[3] = { 0, 0, 0 };
    int      n       = 0;
    float    scale   = 1.0f;
    bool     fused   = false;
};

// Per-layer weights
struct Qwen3Layer {
    struct ggml_tensor * input_layernorm;      // [H]
    struct ggml_tensor * post_attn_layernorm;  // [H]

    // Attention (fused or separate)
    struct ggml_tensor * qkv;     // [H, (Nh+2*Nkv)*D] full fused (or NULL)
    struct ggml_tensor * qk;      // [H, (Nh+Nkv)*D] Q+K fused (or NULL)
    struct ggml_tensor * q_proj;  // [H, Nh*D]  (NULL when fused)
    struct ggml_tensor * k_proj;  // [H, Nkv*D] (NULL when fused)
    struct ggml_tensor * v_proj;  // [H, Nkv*D] (NULL when QKV fused)
    struct ggml_tensor * o_proj;  // [Nh*D, H]
    struct ggml_tensor * q_norm;  // [D]
    struct ggml_tensor * k_norm;  // [D]

    // MLP (fused or separate)
    struct ggml_tensor * gate_up;    // [H, 2*FFN] fused (or NULL)
    struct ggml_tensor * gate_proj;  // [H, FFN] (NULL when fused)
    struct ggml_tensor * up_proj;    // [H, FFN] (NULL when fused)
    struct ggml_tensor * down_proj;  // [FFN, H]

    // LoRA deltas, one per weight slot above, bound per generate and cleared
    // by qwen3_unbind_lora. All zeroed means no adapter.
    LoraDelta lora_qkv, lora_qk, lora_q, lora_k, lora_v, lora_o;
    LoraDelta lora_gate_up, lora_gate, lora_up, lora_down;
};

// Helpers (pure graph ops, no side effects)
static struct ggml_tensor * qwen3_f32(struct ggml_context * ctx, struct ggml_tensor * t) {
    if (t->type == GGML_TYPE_F32) {
        return t;
    }
    return ggml_cast(ctx, t, GGML_TYPE_F32);
}

// y = Wx, plus scale * B(Ax) for every LoRA part bound to this slot.
//
// An unfused weight adds its single delta outright. A fused weight accs each
// part into the row range of its constituent, which keeps a partial adapter
// free of any zero-fill: the parts it does not cover simply emit no nodes.
// ggml_acc is cheap here because the copy scales with the sequence length and
// only the one prefill forward has a long one.
static struct ggml_tensor * qwen3_linear(struct ggml_context * ctx,
                                         struct ggml_tensor *  w,
                                         struct ggml_tensor *  x,
                                         const LoraDelta *     ld = nullptr) {
    struct ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    if (!ld || ld->n == 0) {
        return y;
    }
    for (int i = 0; i < ld->n; i++) {
        if (!ld->part[i].a) {
            continue;
        }
        struct ggml_tensor * d = ggml_mul_mat(ctx, ld->part[i].b, ggml_mul_mat(ctx, ld->part[i].a, x));
        if (ld->scale != 1.0f) {
            d = ggml_scale(ctx, d, ld->scale);
        }
        y = ld->fused ? ggml_acc(ctx, y, d, y->nb[1], y->nb[2], y->nb[3], (size_t) ld->row0[i] * y->nb[0]) :
                        ggml_add(ctx, y, d);
    }
    return y;
}

// F32 manual attention (fallback when flash_attn_ext is disabled).
// Works for 3D [D, S, X] and 4D [D, S, X, N] inputs.
// Returns same layout as flash_attn_ext: dims 1 and 2 swapped vs input.
static struct ggml_tensor * qwen3_attn_f32(struct ggml_context * ctx,
                                           struct ggml_tensor *  q,
                                           struct ggml_tensor *  k,
                                           struct ggml_tensor *  v,
                                           struct ggml_tensor *  mask,
                                           float                 scale) {
    struct ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
    scores                      = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);
    struct ggml_tensor * vt     = ggml_cont(ctx, ggml_transpose(ctx, v));
    struct ggml_tensor * out    = ggml_mul_mat(ctx, vt, scores);
    return ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
}

static struct ggml_tensor * qwen3_rms_norm(struct ggml_context * ctx,
                                           struct ggml_tensor *  x,
                                           struct ggml_tensor *  w,
                                           float                 eps) {
    struct ggml_tensor * n = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, n, qwen3_f32(ctx, w));
}

// Graph builders
// These build sub-graphs and return output tensors.
// They operate on ggml layout: [H, S] for hidden states.

// MLP: SwiGLU (fused gate+up or separate)
static struct ggml_tensor * qwen3_build_mlp(struct ggml_context * ctx,
                                            Qwen3Layer *          ly,
                                            struct ggml_tensor *  x,  // [H, S]
                                            int                   S) {
    (void) S;
    struct ggml_tensor * ff;
    if (ly->gate_up) {
        struct ggml_tensor * gu = qwen3_linear(ctx, ly->gate_up, x, &ly->lora_gate_up);
        ff                      = ggml_swiglu(ctx, gu);
    } else {
        struct ggml_tensor * gate = qwen3_linear(ctx, ly->gate_proj, x, &ly->lora_gate);
        struct ggml_tensor * up   = qwen3_linear(ctx, ly->up_proj, x, &ly->lora_up);
        ff                        = ggml_swiglu_split(ctx, gate, up);
    }
    return qwen3_linear(ctx, ly->down_proj, ff, &ly->lora_down);
}

// Loading
static void qwen3_load_layer(WeightCtx *         wctx,
                             const GGUFModel &   gf,
                             Qwen3Layer *        ly,
                             const std::string & prefix,
                             int                 layer_idx = -1) {
    ly->input_layernorm     = gf_load_tensor_f32(wctx, gf, prefix + ".input_layernorm.weight");
    ly->post_attn_layernorm = gf_load_tensor_f32(wctx, gf, prefix + ".post_attention_layernorm.weight");

    // Attention: try Q+K+V fused, then Q+K partial, then separate
    ly->qkv = gf_load_qkv_fused(wctx, gf, prefix + ".self_attn.q_proj.weight", prefix + ".self_attn.k_proj.weight",
                                prefix + ".self_attn.v_proj.weight");
    if (!ly->qkv) {
        ly->qk = gf_load_pair_fused(wctx, gf, prefix + ".self_attn.q_proj.weight", prefix + ".self_attn.k_proj.weight");
        if (ly->qk) {
            ly->v_proj = gf_load_tensor(wctx, gf, prefix + ".self_attn.v_proj.weight");
            if (layer_idx == 0) {
                fprintf(stderr, "[Qwen3] Attn: Q+K fused, V separate\n");
            }
        } else {
            ly->q_proj = gf_load_tensor(wctx, gf, prefix + ".self_attn.q_proj.weight");
            ly->k_proj = gf_load_tensor(wctx, gf, prefix + ".self_attn.k_proj.weight");
            ly->v_proj = gf_load_tensor(wctx, gf, prefix + ".self_attn.v_proj.weight");
            if (layer_idx == 0) {
                fprintf(stderr, "[Qwen3] Attn: all separate\n");
            }
        }
    } else {
        if (layer_idx == 0) {
            fprintf(stderr, "[Qwen3] Attn: Q+K+V fused\n");
        }
    }

    ly->o_proj = gf_load_tensor(wctx, gf, prefix + ".self_attn.o_proj.weight");
    ly->q_norm = gf_load_tensor_f32(wctx, gf, prefix + ".self_attn.q_norm.weight");
    ly->k_norm = gf_load_tensor_f32(wctx, gf, prefix + ".self_attn.k_norm.weight");

    // MLP: try gate+up fused, then separate
    ly->gate_up = gf_load_pair_fused(wctx, gf, prefix + ".mlp.gate_proj.weight", prefix + ".mlp.up_proj.weight");
    if (ly->gate_up) {
        if (layer_idx == 0) {
            fprintf(stderr, "[Qwen3] MLP: gate+up fused\n");
        }
    } else {
        ly->gate_proj = gf_load_tensor(wctx, gf, prefix + ".mlp.gate_proj.weight");
        ly->up_proj   = gf_load_tensor(wctx, gf, prefix + ".mlp.up_proj.weight");
        if (layer_idx == 0) {
            fprintf(stderr, "[Qwen3] MLP: gate+up separate\n");
        }
    }
    ly->down_proj = gf_load_tensor(wctx, gf, prefix + ".mlp.down_proj.weight");
}

// Fill one delta from the set. Absent in the adapter leaves it at n = 0,
// which makes qwen3_linear take its original path.
static void qwen3_bind_one(LoraDelta * ld, const LoraSet * set, const std::string & base, float scale) {
    *ld         = {};
    ld->part[0] = lora_find(set, base);
    ld->n       = ld->part[0].a ? 1 : 0;
    ld->scale   = scale;
    ld->fused   = false;
}

// Bind every slot this layer actually loaded. The fused slots stay unbound
// here; qwen3_bind_lora_fused knows their row geometry.
static void qwen3_bind_lora(Qwen3Layer * ly, const LoraSet * set, const std::string & prefix, float scale) {
    if (ly->q_proj) {
        qwen3_bind_one(&ly->lora_q, set, prefix + ".self_attn.q_proj.weight", scale);
    }
    if (ly->k_proj) {
        qwen3_bind_one(&ly->lora_k, set, prefix + ".self_attn.k_proj.weight", scale);
    }
    if (ly->v_proj) {
        qwen3_bind_one(&ly->lora_v, set, prefix + ".self_attn.v_proj.weight", scale);
    }
    qwen3_bind_one(&ly->lora_o, set, prefix + ".self_attn.o_proj.weight", scale);
    if (ly->gate_proj) {
        qwen3_bind_one(&ly->lora_gate, set, prefix + ".mlp.gate_proj.weight", scale);
    }
    if (ly->up_proj) {
        qwen3_bind_one(&ly->lora_up, set, prefix + ".mlp.up_proj.weight", scale);
    }
    qwen3_bind_one(&ly->lora_down, set, prefix + ".mlp.down_proj.weight", scale);
}

// The NAR weight set of the same layers. nar_load_layer names its tensors
// nar_self_attn / nar_mlp, and tries only the fused-or-separate pair, so the
// qk slot never comes up here.
static void qwen3_bind_lora_nar(Qwen3Layer * ly, const LoraSet * set, const std::string & prefix, float scale) {
    if (!ly->qkv) {
        qwen3_bind_one(&ly->lora_q, set, prefix + ".nar_self_attn.q_proj.weight", scale);
        qwen3_bind_one(&ly->lora_k, set, prefix + ".nar_self_attn.k_proj.weight", scale);
        qwen3_bind_one(&ly->lora_v, set, prefix + ".nar_self_attn.v_proj.weight", scale);
    }
    qwen3_bind_one(&ly->lora_o, set, prefix + ".nar_self_attn.o_proj.weight", scale);
    if (!ly->gate_up) {
        qwen3_bind_one(&ly->lora_gate, set, prefix + ".nar_mlp.gate_proj.weight", scale);
        qwen3_bind_one(&ly->lora_up, set, prefix + ".nar_mlp.up_proj.weight", scale);
    }
    qwen3_bind_one(&ly->lora_down, set, prefix + ".nar_mlp.down_proj.weight", scale);
}

static void qwen3_unbind_lora(Qwen3Layer * ly) {
    ly->lora_qkv = ly->lora_qk = ly->lora_q = ly->lora_k = ly->lora_v = ly->lora_o = {};
    ly->lora_gate_up = ly->lora_gate = ly->lora_up = ly->lora_down = {};
}
