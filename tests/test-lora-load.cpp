// test-lora-load.cpp: adapter GGUF load harness
//
// Loads an adapter against a backbone and prints one line per pair,
// "name a_ne0 a_ne1 b_ne0 b_ne1", so the python side can diff it against the
// file it wrote. Exits 1 when the adapter names a tensor the backbone does
// not have.

#include "lora.h"

#include <cstdio>

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s base.gguf adapter.gguf\n", argv[0]);
        return 1;
    }

    LoraSet set;
    if (!lora_load(&set, argv[2], argv[1])) {
        fprintf(stderr, "[Test-LoRA] load failed\n");
        return 1;
    }

    printf("rank %d\n", set.rank);
    printf("pairs %zu\n", set.pairs.size());
    for (const auto & kv : set.pairs) {
        printf("%s %lld %lld %lld %lld\n", kv.first.c_str(), (long long) kv.second.a->ne[0],
               (long long) kv.second.a->ne[1], (long long) kv.second.b->ne[0], (long long) kv.second.b->ne[1]);
    }

    lora_free(&set);
    return 0;
}
