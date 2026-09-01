#include "spacepdhcg/cuda/orbitweaver_g3_adapter.hpp"

#include <type_traits>

namespace g7 = spacepdhcg::orbitweaver::g7;

static_assert(std::is_base_of_v<g7::ArcBatchBackend, g7::G3PersistentTrajectoryAdapter>);
static_assert(std::has_virtual_destructor_v<g7::G3ArcBinding>);

int main() {
    // Compile/link contract only.  Device execution is intentionally deferred to the
    // dedicated one-GPU route-correctness campaign.
    return 0;
}
