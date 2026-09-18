#pragma once
// model-store.h: centralised ownership of GGML modules
//
// VRAM policy doctrine. READ THIS BEFORE CHANGING ANYTHING IN THIS FILE.
//
//   --keep-loaded (EVICT_NEVER)
//       Everything stays in VRAM. No reload, ever. The user is telling us
//       they have the budget for the full working set. Do not second-guess
//       them by adding smart eviction rules.
//
//   default (EVICT_STRICT)
//       Maximum VRAM optimisation. At most one coexistence group resident
//       at a time. The AR weights and the NAR weights never coexist by
//       construction, because they live in different groups.
//
//   coexistence groups
//       The pipeline interleaves some modules at a granularity where
//       evicting between them would thrash, so eviction operates on
//       groups, not single modules:
//         AR         = { LM }        the score and semantic stages
//         SYNTH      = { NAR, VAE }  the flow matching and the decode, per song
//         TRANSCRIBE = { SS2 }       the audio to score transcriber
//       Modules in the same group coexist freely. A require from another
//       group evicts every idle module of the resident group.
//
//   the KV cache is not a module
//       The AR fills it and the NAR reads it, so it belongs to the pipeline
//       and survives every eviction. That is what lets the two halves of one
//       GGUF trade places in VRAM around it.
//
//   the LoRA adapter is not a module either
//       Both halves of the backbone read pairs out of the same adapter, so it
//       cannot live in either coexistence group without the other group's
//       require evicting it mid generate. It gets its own slot with its own
//       rule: at most one resident, and requiring a different path replaces
//       it. Like the cache, it survives the eviction of what it decorates.
//
//   invariant held under BOTH policies
//       Exactly ONE instance per ModelKey for the whole process. Two
//       requires with the same key return the same pointer.
//
// A ModelStore holds the GGML module instances that the pipeline needs
// (AR half, NAR half, VAE). The pipeline asks the store for a module by key
// and returns it when done. The store decides what stays in VRAM and what
// gets evicted, following the policy above set at creation time.
//
// Keys
//   A module is uniquely identified by (kind, path). Two requires with the
//   same key return the same instance.
//
// Refcounting
//   Each module has a refcount. require increments it, release decrements.
//   In EVICT_STRICT, a module with refcount > 0 in another group cannot be
//   evicted: a conflicting require is a programming error (aborts). This
//   catches accidental overlap between groups that must not coexist.
//
// Threading
//   The store has no lock: a single worker thread owns it. The server
//   serializes GPU work on one worker; HTTP handlers never touch the
//   store. Adding a second worker requires adding a mutex here first.

#include "bpe.h"
#include "lora.h"
#include "nar.h"
#include "qwen3-lm.h"
#include "sheetsage.h"
#include "vae.h"

#include <cstddef>
#include <string>

struct ModelStore;

enum ModelKind {
    MODEL_LM,   // Qwen3LM  the AR half of the backbone GGUF
    MODEL_NAR,  // Yue2NAR  the NAR half of the same GGUF
    MODEL_VAE,  // VAEGGML  from the VAE GGUF
    MODEL_SS2,  // SheetSage2 from the transcriber GGUF
};

struct ModelKey {
    ModelKind   kind;
    std::string path;  // GGUF path the module is loaded from
};

enum EvictPolicy {
    EVICT_STRICT,  // default: at most one coexistence group resident
    EVICT_NEVER,   // --keep-loaded: never evict, accumulate
};

ModelStore * store_create(EvictPolicy policy);
void         store_free(ModelStore * s);
EvictPolicy  store_policy(const ModelStore * s);

// Typed GPU module accessors. Each returns a pointer owned by the store;
// never free it yourself. Returns NULL on load failure.
//
// After require, the module stays resident with a refcount > 0 until the
// matching release. In EVICT_STRICT, require evicts every module outside
// its coexistence group whose refcount is zero; if any conflicting module
// has refcount > 0 the store aborts (a programming error in the caller).
Qwen3LM *    store_require_lm(ModelStore * s, const ModelKey & k);
Yue2NAR *    store_require_nar(ModelStore * s, const ModelKey & k);
VAEGGML *    store_require_vae(ModelStore * s, const ModelKey & k);
SheetSage2 * store_require_ss2(ModelStore * s, const ModelKey & k);

// Release decrements the refcount for the module behind this handle.
// Pass exactly the pointer returned by require. After release, the pointer
// must not be used: in EVICT_STRICT it may be unloaded immediately.
void store_release(ModelStore * s, void * handle);

// CPU-resident accessor. Loaded on first call, kept forever, never
// evicted: the tokenizer travels with the backbone GGUF metadata (a few
// MB). Returns NULL on load failure.
BPETokenizer * store_bpe(ModelStore * s, const char * lm_path);

// Adapter accessor. Loads on first require of a path and keeps it resident
// while the refcount is above zero. Requiring a different path with the
// current one idle replaces it; requiring one while another is still held is a
// programming error and aborts. base_path is the backbone the adapter is
// validated against. Returns NULL on load failure, and for an empty path,
// which is how "no adapter" travels.
const LoraSet * store_require_lora(ModelStore * s, const char * lora_path, const char * base_path);
void            store_release_lora(ModelStore * s, const LoraSet * set);

// Observability: sum of currently resident GPU module weight buffers, and
// the count of loaded GPU modules.
size_t store_vram_bytes(const ModelStore * s);
int    store_gpu_module_count(const ModelStore * s);

// RAII helper. Builds on top of store_release, nothing else.
struct ModelHandle {
    ModelStore * store;
    void *       ptr;

    ModelHandle(ModelStore * s, void * p) : store(s), ptr(p) {}

    ~ModelHandle() {
        if (store && ptr) {
            store_release(store, ptr);
        }
    }

    // non-copyable, movable
    ModelHandle(const ModelHandle &)             = delete;
    ModelHandle & operator=(const ModelHandle &) = delete;

    ModelHandle(ModelHandle && o) noexcept : store(o.store), ptr(o.ptr) {
        o.store = nullptr;
        o.ptr   = nullptr;
    }
};
