// test-lora.cpp: LoRA delta parity harness
//
// Forwards one token sequence with an adapter bound at a given strength and
// dumps the logits of the last position, so the torch reference can be diffed
// against it. With "none" as the adapter path it dumps the base logits, which
// is how the harness proves a bound adapter changes the output and an unbound
// one does not.

#include "lora.h"
#include "qwen3-lm.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static bool dump(const std::string & path, const std::vector<float> & data) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f || fwrite(data.data(), sizeof(float), data.size(), f) != data.size()) {
        fprintf(stderr, "[Test-LoRA] cannot write %s\n", path.c_str());
        return false;
    }
    fclose(f);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s lm.gguf adapter.gguf|none scale out_prefix id0 [id1 ...]\n", argv[0]);
        return 1;
    }
    const char * lm_path      = argv[1];
    const char * adapter_path = argv[2];
    float        scale        = (float) atof(argv[3]);
    std::string  out_prefix   = argv[4];

    std::vector<int> ids;
    for (int i = 5; i < argc; i++) {
        ids.push_back(atoi(argv[i]));
    }

    Qwen3LM lm;
    if (!qw3lm_load(&lm, lm_path)) {
        return 1;
    }

    LoraSet set;
    bool    has_lora = strcmp(adapter_path, "none") != 0;
    if (has_lora) {
        if (!lora_load(&set, adapter_path, lm_path)) {
            return 1;
        }
        qw3lm_bind_lora(&lm, &set, scale);
    }

    // One sequence in one KV set, forwarded over the whole vocabulary: the
    // same shape test-lm.cpp uses for its prefill pass.
    const int    V = lm.cfg.vocab_size;
    Qw3lmKvCache kv;
    qw3lm_kv_init(&kv, lm.cfg, lm.backend);
    if (!qw3lm_kv_sets(&kv, 1)) {
        return 1;
    }
    std::vector<float> logits((size_t) V);
    qw3lm_forward(&lm, &kv, ids.data(), (int) ids.size(), 0, logits.data(), 0, V);
    if (!dump(out_prefix + ".bin", logits)) {
        return 1;
    }

    qw3lm_kv_free(&kv);
    if (has_lora) {
        qw3lm_unbind_lora(&lm);
        lora_free(&set);
    }
    qw3lm_free(&lm);
    return 0;
}
