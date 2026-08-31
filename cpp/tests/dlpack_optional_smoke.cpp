#include "spacepdhcg/dlpack_adapter.hpp"

int main() {
    static_assert(SPACEPDHCG_HAS_DLPACK == 0 || SPACEPDHCG_HAS_DLPACK == 1);
    return 0;
}
