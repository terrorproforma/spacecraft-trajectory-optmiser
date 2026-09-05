// Linux diagnostic interposer. Preload before libqoco.so to track only the four
// host allocations returned by construct_kkt. CUDA memcheck cannot see this leak.
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
extern "C" {
#include "kkt.h"
void __libc_free(void*);
}
struct Allocation { std::atomic<void*> pointer{}; std::atomic<size_t> bytes{}; };
static Allocation allocations[256];
static std::atomic<size_t> created{}, freed{}, matrices{};
static void remember(void* pointer, size_t bytes) {
    if (!pointer) return;
    for (auto& allocation : allocations) {
        if (!allocation.pointer.load(std::memory_order_acquire)) {
            allocation.bytes = bytes;
            allocation.pointer.store(pointer, std::memory_order_release);
            created.fetch_add(bytes); return;
        }
    }
    std::fputs("KKT host allocation tracker capacity exceeded\n", stderr); std::abort();
}
extern "C" void free(void* pointer) noexcept {
    if (pointer) for (auto& allocation : allocations) {
        if (allocation.pointer.load(std::memory_order_acquire) != pointer) continue;
        const auto bytes = allocation.bytes.load();
        void* expected = pointer;
        if (allocation.pointer.compare_exchange_strong(expected, nullptr)) {
            freed.fetch_add(bytes); break;
        }
    }
    __libc_free(pointer);
}
extern "C" QOCOCscMatrix* construct_kkt(QOCOCscMatrix* P, QOCOCscMatrix* A,
    QOCOCscMatrix* G, QOCOCscMatrix* At, QOCOCscMatrix* Gt, QOCOFloat reg,
    QOCOInt n, QOCOInt m, QOCOInt p, QOCOInt l, QOCOInt nsoc, QOCOInt* q,
    QOCOInt* pm, QOCOInt* am, QOCOInt* gm, QOCOInt* nt, QOCOInt* ntd, QOCOInt wnnz) {
    const auto actual = reinterpret_cast<decltype(&construct_kkt)>(dlsym(RTLD_NEXT, "construct_kkt"));
    if (!actual) { std::fputs("KKT tracker could not resolve constructor\n", stderr); std::abort(); }
    auto* result = actual(P, A, G, At, Gt, reg, n, m, p, l, nsoc, q, pm, am, gm, nt, ntd, wnnz);
    remember(result->x, result->nnz * sizeof(QOCOFloat));
    remember(result->i, result->nnz * sizeof(QOCOInt));
    remember(result->p, (result->n + 1) * sizeof(QOCOInt));
    remember(result, sizeof(QOCOCscMatrix)); matrices.fetch_add(1);
    return result;
}
__attribute__((destructor)) static void report() {
    std::fprintf(stderr, "{\"case\":\"qoco_host_kkt_lifetime\",\"matrices\":%zu,\"created_bytes\":%zu,\"freed_bytes\":%zu,\"live_bytes\":%zu}\n",
        matrices.load(), created.load(), freed.load(), created.load() - freed.load());
}
