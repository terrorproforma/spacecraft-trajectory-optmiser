#include "spacepdhcg/c_api.h"

#include <cstdlib>
#include <vector>

void require(const bool condition) {
    if (!condition) {
        std::abort();
    }
}

int main() {
    spacepdhcg_lambert_family_request request{};
    request.deterministic_id = 9U;
    request.departure_position[0] = 7.0e6;
    request.arrival_position[1] = 8.0e6;
    request.time_of_flight = 3'600.0;
    request.gravitational_parameter = 3.986004418e14;
    request.time_tolerance = 1.0e-8;
    request.maximum_iterations = 256U;
    request.maximum_revolutions = 1U;
    request.scan_samples_per_band = 512U;
    request.include_short_way = 1;
    const auto stride = spacepdhcg_lambert_family_result_stride(1U);
    require(stride == 6U);
    std::vector<spacepdhcg_lambert_family_result> results(stride);
    require(
        spacepdhcg_lambert_family_batch_cpu(
            &request, 1U, 1U, results.data(), results.size()
        )
        == SPACEPDHCG_STATUS_OK
    );
    require(results[0].deterministic_id == 9U);
    require(results[0].status == SPACEPDHCG_LAMBERT_FAMILY_FEASIBLE);
    require(results[3].status == SPACEPDHCG_LAMBERT_FAMILY_UNSUPPORTED);
    return 0;
}
