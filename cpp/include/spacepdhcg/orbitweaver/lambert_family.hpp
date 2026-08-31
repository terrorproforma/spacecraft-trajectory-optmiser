#pragma once

#include "spacepdhcg/orbitweaver/lambert.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace spacepdhcg::orbitweaver {

enum class LambertParameterBranch {
    unique,
    lower_parameter,
    higher_parameter,
};

struct LambertFamilyMember {
    LambertSolution solution{};
    std::size_t revolutions{0U};
    bool long_way{false};
    LambertParameterBranch branch{LambertParameterBranch::unique};
};

namespace lambert_family_detail {

inline constexpr double pi = 3.141592653589793238462643383279502884;

struct Geometry {
    double radius_one{0.0};
    double radius_two{0.0};
    double cosine{0.0};
    double geometry_a{0.0};
    double transfer_angle{0.0};
};

inline Geometry geometry(
    const Vector3& departure_position,
    const Vector3& arrival_position,
    const bool long_way
) {
    detail::validate_vector(departure_position, "departure position must be finite");
    detail::validate_vector(arrival_position, "arrival position must be finite");
    const auto radius_one = detail::norm(departure_position);
    const auto radius_two = detail::norm(arrival_position);
    if (radius_one <= 0.0 || radius_two <= 0.0) {
        throw std::invalid_argument("Lambert endpoint radii must be positive");
    }
    const auto cosine = std::clamp(
        detail::dot(departure_position, arrival_position) / (radius_one * radius_two),
        -1.0,
        1.0
    );
    auto sine = std::sqrt(std::max(0.0, 1.0 - cosine * cosine));
    if (long_way) {
        sine = -sine;
    }
    const auto denominator = 1.0 - cosine;
    if (denominator <= 1.0e-14 || std::abs(sine) <= 1.0e-14) {
        throw std::invalid_argument("collinear Lambert endpoints require a specialised solver");
    }
    const auto geometry_a = sine * std::sqrt(radius_one * radius_two / denominator);
    if (std::abs(geometry_a) <= 1.0e-14) {
        throw std::invalid_argument("Lambert geometry is singular");
    }
    auto transfer_angle = std::acos(cosine);
    if (long_way) {
        transfer_angle = 2.0 * pi - transfer_angle;
    }
    return Geometry{
        radius_one,
        radius_two,
        cosine,
        geometry_a,
        transfer_angle,
    };
}

inline LambertSolution solution_from_root(
    const Vector3& departure_position,
    const Vector3& arrival_position,
    const Geometry& geometry_value,
    const double root,
    const double time_of_flight,
    const double gravitational_parameter,
    const std::size_t iterations
) {
    const auto evaluation = detail::evaluate_universal(
        root,
        geometry_value.radius_one,
        geometry_value.radius_two,
        geometry_value.geometry_a,
        time_of_flight,
        gravitational_parameter
    );
    if (!evaluation.has_value()) {
        throw std::runtime_error(
            "multi-revolution Lambert root produced an invalid universal state"
        );
    }
    const auto y = evaluation->y;
    const auto f = 1.0 - y / geometry_value.radius_one;
    const auto g = geometry_value.geometry_a
                   * std::sqrt(y / gravitational_parameter);
    const auto g_dot = 1.0 - y / geometry_value.radius_two;
    if (!std::isfinite(g) || std::abs(g) <= 1.0e-14) {
        throw std::runtime_error("multi-revolution Lambert g coefficient is singular");
    }
    const auto departure_velocity = detail::scale(
        detail::subtract(
            arrival_position,
            detail::scale(departure_position, f)
        ),
        1.0 / g
    );
    const auto arrival_velocity = detail::scale(
        detail::subtract(
            detail::scale(arrival_position, g_dot),
            departure_position
        ),
        1.0 / g
    );
    return LambertSolution{
        departure_velocity,
        arrival_velocity,
        root,
        geometry_value.transfer_angle,
        iterations,
        evaluation->residual,
    };
}

struct Root {
    double parameter{0.0};
    std::size_t iterations{0U};
};

inline Root bisect_root(
    double lower,
    double upper,
    const Geometry& geometry_value,
    const double time_of_flight,
    const double gravitational_parameter,
    const double time_tolerance,
    const std::size_t maximum_iterations
) {
    auto lower_evaluation = detail::evaluate_universal(
        lower,
        geometry_value.radius_one,
        geometry_value.radius_two,
        geometry_value.geometry_a,
        time_of_flight,
        gravitational_parameter
    );
    auto upper_evaluation = detail::evaluate_universal(
        upper,
        geometry_value.radius_one,
        geometry_value.radius_two,
        geometry_value.geometry_a,
        time_of_flight,
        gravitational_parameter
    );
    if (!lower_evaluation.has_value() || !upper_evaluation.has_value()
        || lower_evaluation->residual * upper_evaluation->residual > 0.0) {
        throw std::runtime_error("multi-revolution Lambert bracket is invalid");
    }
    double root = 0.5 * (lower + upper);
    std::size_t iterations{0U};
    for (; iterations < maximum_iterations; ++iterations) {
        root = 0.5 * (lower + upper);
        const auto middle = detail::evaluate_universal(
            root,
            geometry_value.radius_one,
            geometry_value.radius_two,
            geometry_value.geometry_a,
            time_of_flight,
            gravitational_parameter
        );
        if (!middle.has_value()) {
            throw std::runtime_error(
                "multi-revolution Lambert bisection entered an invalid interval"
            );
        }
        if (std::abs(middle->residual) <= time_tolerance
            || std::abs(upper - lower) <= 1.0e-13) {
            return Root{root, iterations + 1U};
        }
        if (lower_evaluation->residual * middle->residual <= 0.0) {
            upper = root;
            upper_evaluation = middle;
        } else {
            lower = root;
            lower_evaluation = middle;
        }
    }
    throw std::runtime_error("multi-revolution Lambert bisection reached its limit");
}

inline std::vector<Root> roots_in_revolution_band(
    const Geometry& geometry_value,
    const double time_of_flight,
    const double gravitational_parameter,
    const std::size_t revolutions,
    const double time_tolerance,
    const std::size_t maximum_iterations,
    const std::size_t scan_samples
) {
    if (revolutions == 0U) {
        throw std::invalid_argument("multi-revolution band index must be positive");
    }
    const auto lower_singularity =
        4.0 * static_cast<double>(revolutions * revolutions) * pi * pi;
    const auto next = revolutions + 1U;
    const auto upper_singularity =
        4.0 * static_cast<double>(next * next) * pi * pi;
    const auto margin = 1.0e-9 * std::max(1.0, upper_singularity);
    const auto lower = lower_singularity + margin;
    const auto upper = upper_singularity - margin;

    std::vector<Root> roots{};
    std::pair<double, detail::UniversalEvaluation> previous{};
    bool has_previous = false;
    for (std::size_t sample = 0; sample <= scan_samples; ++sample) {
        const auto fraction = static_cast<double>(sample)
                              / static_cast<double>(scan_samples);
        const auto parameter = lower + fraction * (upper - lower);
        const auto evaluation = detail::evaluate_universal(
            parameter,
            geometry_value.radius_one,
            geometry_value.radius_two,
            geometry_value.geometry_a,
            time_of_flight,
            gravitational_parameter
        );
        if (!evaluation.has_value()) {
            has_previous = false;
            continue;
        }
        if (std::abs(evaluation->residual) <= time_tolerance) {
            roots.push_back(Root{parameter, 0U});
            previous = std::make_pair(parameter, *evaluation);
            has_previous = true;
            continue;
        }
        if (has_previous
            && previous.second.residual * evaluation->residual < 0.0) {
            roots.push_back(bisect_root(
                previous.first,
                parameter,
                geometry_value,
                time_of_flight,
                gravitational_parameter,
                time_tolerance,
                maximum_iterations
            ));
        }
        previous = std::make_pair(parameter, *evaluation);
        has_previous = true;
    }
    std::sort(roots.begin(), roots.end(), [](const Root& left, const Root& right) {
        return left.parameter < right.parameter;
    });
    roots.erase(
        std::unique(
            roots.begin(),
            roots.end(),
            [](const Root& left, const Root& right) {
                return std::abs(left.parameter - right.parameter)
                       <= 1.0e-9
                              * std::max(
                                  {1.0,
                                   std::abs(left.parameter),
                                   std::abs(right.parameter)}
                              );
            }
        ),
        roots.end()
    );
    return roots;
}

inline void validate_solver_inputs(
    const double time_of_flight,
    const double gravitational_parameter,
    const double time_tolerance,
    const std::size_t maximum_iterations,
    const std::size_t scan_samples
) {
    if (!std::isfinite(time_of_flight) || time_of_flight <= 0.0
        || !std::isfinite(gravitational_parameter) || gravitational_parameter <= 0.0
        || !std::isfinite(time_tolerance) || time_tolerance <= 0.0
        || maximum_iterations == 0U || scan_samples < 16U) {
        throw std::invalid_argument("multi-revolution Lambert solver inputs are invalid");
    }
}

}  // namespace lambert_family_detail

/// Enumerate every universal-variable root through `maximum_revolutions` for one transfer
/// direction. Revolution-zero contributes one root. Each positive-revolution band may
/// contribute a lower-parameter and higher-parameter branch, or none when the requested
/// time of flight lies below that band's minimum.
[[nodiscard]] inline std::vector<LambertFamilyMember> solve_lambert_revolution_family(
    const Vector3& departure_position,
    const Vector3& arrival_position,
    const double time_of_flight,
    const double gravitational_parameter,
    const std::size_t maximum_revolutions,
    const bool long_way = false,
    const double time_tolerance = 1.0e-8,
    const std::size_t maximum_iterations = 256U,
    const std::size_t scan_samples_per_band = 8'192U
) {
    lambert_family_detail::validate_solver_inputs(
        time_of_flight,
        gravitational_parameter,
        time_tolerance,
        maximum_iterations,
        scan_samples_per_band
    );
    const auto geometry_value = lambert_family_detail::geometry(
        departure_position,
        arrival_position,
        long_way
    );
    std::vector<LambertFamilyMember> result{};
    result.push_back(LambertFamilyMember{
        solve_lambert_zero_revolution(
            departure_position,
            arrival_position,
            time_of_flight,
            gravitational_parameter,
            long_way,
            time_tolerance,
            maximum_iterations
        ),
        0U,
        long_way,
        LambertParameterBranch::unique,
    });
    for (std::size_t revolutions = 1U;
         revolutions <= maximum_revolutions;
         ++revolutions) {
        const auto roots = lambert_family_detail::roots_in_revolution_band(
            geometry_value,
            time_of_flight,
            gravitational_parameter,
            revolutions,
            time_tolerance,
            maximum_iterations,
            scan_samples_per_band
        );
        for (std::size_t branch = 0; branch < roots.size(); ++branch) {
            result.push_back(LambertFamilyMember{
                lambert_family_detail::solution_from_root(
                    departure_position,
                    arrival_position,
                    geometry_value,
                    roots[branch].parameter,
                    time_of_flight,
                    gravitational_parameter,
                    roots[branch].iterations
                ),
                revolutions,
                long_way,
                roots.size() == 1U
                    ? LambertParameterBranch::unique
                    : branch == 0U
                          ? LambertParameterBranch::lower_parameter
                          : LambertParameterBranch::higher_parameter,
            });
        }
    }
    return result;
}

/// Enumerate short-way and/or long-way Lambert families and sort them deterministically by
/// direction, revolution count, then universal parameter.
[[nodiscard]] inline std::vector<LambertFamilyMember> enumerate_lambert_families(
    const Vector3& departure_position,
    const Vector3& arrival_position,
    const double time_of_flight,
    const double gravitational_parameter,
    const std::size_t maximum_revolutions,
    const bool include_short_way = true,
    const bool include_long_way = true,
    const double time_tolerance = 1.0e-8,
    const std::size_t maximum_iterations = 256U,
    const std::size_t scan_samples_per_band = 8'192U
) {
    if (!include_short_way && !include_long_way) {
        throw std::invalid_argument("at least one Lambert transfer direction is required");
    }
    std::vector<LambertFamilyMember> result{};
    if (include_short_way) {
        auto family = solve_lambert_revolution_family(
            departure_position,
            arrival_position,
            time_of_flight,
            gravitational_parameter,
            maximum_revolutions,
            false,
            time_tolerance,
            maximum_iterations,
            scan_samples_per_band
        );
        result.insert(
            result.end(),
            std::make_move_iterator(family.begin()),
            std::make_move_iterator(family.end())
        );
    }
    if (include_long_way) {
        auto family = solve_lambert_revolution_family(
            departure_position,
            arrival_position,
            time_of_flight,
            gravitational_parameter,
            maximum_revolutions,
            true,
            time_tolerance,
            maximum_iterations,
            scan_samples_per_band
        );
        result.insert(
            result.end(),
            std::make_move_iterator(family.begin()),
            std::make_move_iterator(family.end())
        );
    }
    std::sort(
        result.begin(),
        result.end(),
        [](const LambertFamilyMember& left, const LambertFamilyMember& right) {
            if (left.long_way != right.long_way) {
                return !left.long_way;
            }
            if (left.revolutions != right.revolutions) {
                return left.revolutions < right.revolutions;
            }
            return left.solution.universal_parameter
                   < right.solution.universal_parameter;
        }
    );
    return result;
}

}  // namespace spacepdhcg::orbitweaver
