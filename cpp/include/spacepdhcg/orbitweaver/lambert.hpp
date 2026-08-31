#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace spacepdhcg::orbitweaver {

using Vector3 = std::array<double, 3U>;

struct LambertSolution {
    Vector3 departure_velocity{};
    Vector3 arrival_velocity{};
    double universal_parameter{0.0};
    double transfer_angle_radians{0.0};
    std::size_t iterations{0U};
    double time_of_flight_residual{0.0};
};

struct HohmannTransfer {
    double departure_delta_v{0.0};
    double arrival_delta_v{0.0};
    double total_delta_v{0.0};
    double transfer_time{0.0};
};

namespace detail {

inline double dot(const Vector3& left, const Vector3& right) noexcept {
    return left[0U] * right[0U] + left[1U] * right[1U] + left[2U] * right[2U];
}

inline double norm(const Vector3& vector) noexcept { return std::sqrt(dot(vector, vector)); }

inline Vector3 subtract(const Vector3& left, const Vector3& right) noexcept {
    return Vector3{
        left[0U] - right[0U],
        left[1U] - right[1U],
        left[2U] - right[2U],
    };
}

inline Vector3 scale(const Vector3& vector, double factor) noexcept {
    return Vector3{factor * vector[0U], factor * vector[1U], factor * vector[2U]};
}

inline std::pair<double, double> stumpff(double z) {
    if (z > 1.0e-8) {
        const auto root = std::sqrt(z);
        return {
            (1.0 - std::cos(root)) / z,
            (root - std::sin(root)) / (root * root * root),
        };
    }
    if (z < -1.0e-8) {
        const auto root = std::sqrt(-z);
        return {
            (std::cosh(root) - 1.0) / (-z),
            (std::sinh(root) - root) / (root * root * root),
        };
    }
    const auto z2 = z * z;
    const auto z3 = z2 * z;
    return {
        0.5 - z / 24.0 + z2 / 720.0 - z3 / 40'320.0,
        1.0 / 6.0 - z / 120.0 + z2 / 5'040.0 - z3 / 362'880.0,
    };
}

struct UniversalEvaluation {
    double residual{0.0};
    double y{0.0};
};

inline std::optional<UniversalEvaluation> evaluate_universal(
    double z,
    double radius_one,
    double radius_two,
    double geometry_a,
    double time_of_flight,
    double gravitational_parameter
) {
    const auto [c, s] = stumpff(z);
    if (!std::isfinite(c) || !std::isfinite(s) || c <= 0.0) {
        return std::nullopt;
    }
    const auto root_c = std::sqrt(c);
    const auto y = radius_one + radius_two + geometry_a * (z * s - 1.0) / root_c;
    if (!std::isfinite(y) || y < 0.0) {
        return std::nullopt;
    }
    const auto x = std::sqrt(y / c);
    const auto computed_time =
        (x * x * x * s + geometry_a * std::sqrt(y)) / std::sqrt(gravitational_parameter);
    if (!std::isfinite(computed_time)) {
        return std::nullopt;
    }
    return UniversalEvaluation{computed_time - time_of_flight, y};
}

inline void validate_vector(const Vector3& vector, const char* name) {
    for (const auto value : vector) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(name);
        }
    }
}

}  // namespace detail

/// Solve the classical two-body, zero-revolution Lambert problem.
///
/// `long_way=false` selects an angle in (0, pi); `long_way=true` selects the
/// complementary angle in (pi, 2*pi). Collinear endpoint geometries are rejected.
inline LambertSolution solve_lambert_zero_revolution(
    const Vector3& departure_position,
    const Vector3& arrival_position,
    double time_of_flight,
    double gravitational_parameter,
    bool long_way = false,
    double time_tolerance = 1.0e-8,
    std::size_t maximum_iterations = 256U
) {
    detail::validate_vector(departure_position, "departure position must be finite");
    detail::validate_vector(arrival_position, "arrival position must be finite");
    if (!std::isfinite(time_of_flight) || time_of_flight <= 0.0) {
        throw std::invalid_argument("Lambert time of flight must be finite and positive");
    }
    if (!std::isfinite(gravitational_parameter) || gravitational_parameter <= 0.0) {
        throw std::invalid_argument("gravitational parameter must be finite and positive");
    }
    if (!std::isfinite(time_tolerance) || time_tolerance <= 0.0) {
        throw std::invalid_argument("Lambert time tolerance must be finite and positive");
    }
    if (maximum_iterations == 0U) {
        throw std::invalid_argument("Lambert iteration limit must be positive");
    }

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

    constexpr double pi = 3.141592653589793238462643383279502884;
    constexpr std::size_t scan_samples = 8'192U;
    const auto lower_scan = -4.0 * pi * pi;
    const auto upper_scan = 4.0 * pi * pi - 1.0e-8;
    std::optional<std::pair<double, detail::UniversalEvaluation>> previous{};
    std::optional<double> exact_root{};
    double lower{0.0};
    double upper{0.0};
    bool bracketed{false};

    for (std::size_t sample = 0; sample <= scan_samples; ++sample) {
        const auto fraction = static_cast<double>(sample) / static_cast<double>(scan_samples);
        const auto z = lower_scan + fraction * (upper_scan - lower_scan);
        const auto evaluation = detail::evaluate_universal(
            z,
            radius_one,
            radius_two,
            geometry_a,
            time_of_flight,
            gravitational_parameter
        );
        if (!evaluation.has_value()) {
            continue;
        }
        if (std::abs(evaluation->residual) <= time_tolerance) {
            exact_root = z;
            break;
        }
        if (previous.has_value()) {
            const auto previous_residual = previous->second.residual;
            if ((previous_residual < 0.0 && evaluation->residual > 0.0)
                || (previous_residual > 0.0 && evaluation->residual < 0.0)) {
                lower = previous->first;
                upper = z;
                bracketed = true;
                break;
            }
        }
        previous = std::make_pair(z, *evaluation);
    }

    if (!exact_root.has_value() && !bracketed) {
        throw std::runtime_error(
            "zero-revolution Lambert root was not bracketed for the requested geometry"
        );
    }

    double root = exact_root.value_or(0.5 * (lower + upper));
    std::size_t iterations{0U};
    if (!exact_root.has_value()) {
        auto lower_evaluation = detail::evaluate_universal(
            lower,
            radius_one,
            radius_two,
            geometry_a,
            time_of_flight,
            gravitational_parameter
        );
        if (!lower_evaluation.has_value()) {
            throw std::runtime_error("Lambert lower bracket became invalid");
        }
        for (; iterations < maximum_iterations; ++iterations) {
            root = 0.5 * (lower + upper);
            const auto middle_evaluation = detail::evaluate_universal(
                root,
                radius_one,
                radius_two,
                geometry_a,
                time_of_flight,
                gravitational_parameter
            );
            if (!middle_evaluation.has_value()) {
                lower = root;
                continue;
            }
            if (std::abs(middle_evaluation->residual) <= time_tolerance
                || std::abs(upper - lower) <= 1.0e-13) {
                break;
            }
            if ((lower_evaluation->residual < 0.0 && middle_evaluation->residual > 0.0)
                || (lower_evaluation->residual > 0.0 && middle_evaluation->residual < 0.0)) {
                upper = root;
            } else {
                lower = root;
                lower_evaluation = middle_evaluation;
            }
        }
        if (iterations == maximum_iterations) {
            throw std::runtime_error("Lambert bisection reached its iteration limit");
        }
    }

    const auto final_evaluation = detail::evaluate_universal(
        root,
        radius_one,
        radius_two,
        geometry_a,
        time_of_flight,
        gravitational_parameter
    );
    if (!final_evaluation.has_value()) {
        throw std::runtime_error("Lambert root produced an invalid universal-variable state");
    }
    const auto y = final_evaluation->y;
    const auto f = 1.0 - y / radius_one;
    const auto g = geometry_a * std::sqrt(y / gravitational_parameter);
    const auto g_dot = 1.0 - y / radius_two;
    if (!std::isfinite(g) || std::abs(g) <= 1.0e-14) {
        throw std::runtime_error("Lambert Lagrange coefficient g is singular");
    }
    const auto departure_velocity = detail::scale(
        detail::subtract(arrival_position, detail::scale(departure_position, f)),
        1.0 / g
    );
    const auto arrival_velocity = detail::scale(
        detail::subtract(detail::scale(arrival_position, g_dot), departure_position),
        1.0 / g
    );
    auto transfer_angle = std::acos(cosine);
    if (long_way) {
        transfer_angle = 2.0 * pi - transfer_angle;
    }
    return LambertSolution{
        departure_velocity,
        arrival_velocity,
        root,
        transfer_angle,
        iterations,
        final_evaluation->residual,
    };
}

inline HohmannTransfer hohmann_transfer(
    double initial_radius,
    double final_radius,
    double gravitational_parameter
) {
    if (!std::isfinite(initial_radius) || initial_radius <= 0.0
        || !std::isfinite(final_radius) || final_radius <= 0.0
        || !std::isfinite(gravitational_parameter) || gravitational_parameter <= 0.0) {
        throw std::invalid_argument("Hohmann radii and gravitational parameter must be positive");
    }
    const auto semimajor = 0.5 * (initial_radius + final_radius);
    const auto circular_one = std::sqrt(gravitational_parameter / initial_radius);
    const auto circular_two = std::sqrt(gravitational_parameter / final_radius);
    const auto transfer_one = std::sqrt(
        gravitational_parameter * (2.0 / initial_radius - 1.0 / semimajor)
    );
    const auto transfer_two = std::sqrt(
        gravitational_parameter * (2.0 / final_radius - 1.0 / semimajor)
    );
    const auto departure = std::abs(transfer_one - circular_one);
    const auto arrival = std::abs(circular_two - transfer_two);
    constexpr double pi = 3.141592653589793238462643383279502884;
    const auto transfer_time = pi * std::sqrt(semimajor * semimajor * semimajor / gravitational_parameter);
    return HohmannTransfer{departure, arrival, departure + arrival, transfer_time};
}

inline double propellant_required(
    double initial_mass,
    double delta_v,
    double specific_impulse_seconds,
    double standard_gravity = 9.80665
) {
    if (!std::isfinite(initial_mass) || initial_mass <= 0.0
        || !std::isfinite(delta_v) || delta_v < 0.0
        || !std::isfinite(specific_impulse_seconds) || specific_impulse_seconds <= 0.0
        || !std::isfinite(standard_gravity) || standard_gravity <= 0.0) {
        throw std::invalid_argument("rocket-equation inputs are invalid");
    }
    const auto final_mass = initial_mass
                            / std::exp(delta_v / (specific_impulse_seconds * standard_gravity));
    return initial_mass - final_mass;
}

}  // namespace spacepdhcg::orbitweaver
