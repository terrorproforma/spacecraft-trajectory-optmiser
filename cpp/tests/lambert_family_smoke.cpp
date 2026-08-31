#include "spacepdhcg/orbitweaver/lambert_family.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

int main() {
    using spacepdhcg::orbitweaver::LambertParameterBranch;
    using spacepdhcg::orbitweaver::Vector3;
    using spacepdhcg::orbitweaver::enumerate_lambert_families;
    using spacepdhcg::orbitweaver::solve_lambert_revolution_family;

    constexpr double gravitational_parameter = 398'600.4418;
    constexpr double radius = 7'000.0;
    constexpr double time_of_flight = 10'000.0;
    constexpr double cosine_sixty = 0.5;
    constexpr double sine_sixty = 0.86602540378443864676;
    const Vector3 departure{radius, 0.0, 0.0};
    const Vector3 arrival{
        radius * cosine_sixty,
        radius * sine_sixty,
        0.0,
    };

    const auto short_way = solve_lambert_revolution_family(
        departure,
        arrival,
        time_of_flight,
        gravitational_parameter,
        2U,
        false,
        1.0e-8,
        256U,
        8'192U
    );
    if (short_way.size() != 5U) {
        return 1;
    }
    const auto count_revolutions = [&short_way](const std::size_t revolutions) {
        return static_cast<std::size_t>(std::count_if(
            short_way.begin(),
            short_way.end(),
            [revolutions](const auto& member) {
                return member.revolutions == revolutions;
            }
        ));
    };
    if (count_revolutions(0U) != 1U || count_revolutions(1U) != 2U
        || count_revolutions(2U) != 2U) {
        return 2;
    }
    for (const auto& member : short_way) {
        if (member.long_way || !std::isfinite(member.solution.universal_parameter)
            || std::abs(member.solution.time_of_flight_residual) > 1.0e-6) {
            return 3;
        }
        for (const auto velocity : member.solution.departure_velocity) {
            if (!std::isfinite(velocity)) {
                return 4;
            }
        }
        for (const auto velocity : member.solution.arrival_velocity) {
            if (!std::isfinite(velocity)) {
                return 5;
            }
        }
    }
    const auto first_one_revolution = std::find_if(
        short_way.begin(),
        short_way.end(),
        [](const auto& member) {
            return member.revolutions == 1U;
        }
    );
    if (first_one_revolution == short_way.end()
        || first_one_revolution->branch != LambertParameterBranch::lower_parameter
        || (first_one_revolution + 1)->branch
               != LambertParameterBranch::higher_parameter
        || first_one_revolution->solution.universal_parameter
               >= (first_one_revolution + 1)->solution.universal_parameter) {
        return 6;
    }

    const auto all_directions = enumerate_lambert_families(
        departure,
        arrival,
        time_of_flight,
        gravitational_parameter,
        1U,
        true,
        true
    );
    if (all_directions.size() < short_way.size() - 2U
        || std::none_of(
            all_directions.begin(),
            all_directions.end(),
            [](const auto& member) { return member.long_way; }
        )) {
        return 7;
    }
    return 0;
}
