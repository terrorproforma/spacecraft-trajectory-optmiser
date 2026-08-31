#pragma once

#include "spacepdhcg/orbitweaver/oracle.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace spacepdhcg::orbitweaver {

using Vector3 = std::array<double, 3>;

[[nodiscard]] inline double dot(const Vector3& left, const Vector3& right) noexcept {
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

[[nodiscard]] inline Vector3 cross(const Vector3& left, const Vector3& right) noexcept {
    return Vector3{
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    };
}

[[nodiscard]] inline double norm(const Vector3& vector) noexcept {
    return std::sqrt(dot(vector, vector));
}

[[nodiscard]] inline Vector3 subtract(const Vector3& left, const Vector3& right) noexcept {
    return Vector3{left[0] - right[0], left[1] - right[1], left[2] - right[2]};
}

[[nodiscard]] inline Vector3 scale(const Vector3& vector, const double factor) noexcept {
    return Vector3{factor * vector[0], factor * vector[1], factor * vector[2]};
}

[[nodiscard]] inline Vector3 circular_position(
    const CircularOrbitTarget& target,
    const double epoch
) {
    target.validate();
    const double angle = target.phase(epoch);
    return Vector3{
        target.radius * std::cos(angle),
        target.radius * std::sin(angle),
        0.0,
    };
}

[[nodiscard]] inline Vector3 circular_velocity(
    const CircularOrbitTarget& target,
    const double epoch
) {
    target.validate();
    const double angle = target.phase(epoch);
    const double speed = std::sqrt(target.gravitational_parameter / target.radius);
    return Vector3{
        -speed * std::sin(angle),
        speed * std::cos(angle),
        0.0,
    };
}

struct LambertOptions {
    bool prograde{true};
    double time_tolerance{1.0e-8};
    double universal_tolerance{1.0e-12};
    std::size_t maximum_iterations{200};
    std::size_t bracket_samples{1'024};
    double minimum_transfer_angle{1.0e-8};

    void validate() const {
        if (!std::isfinite(time_tolerance) || time_tolerance <= 0.0 ||
            !std::isfinite(universal_tolerance) || universal_tolerance <= 0.0 ||
            !std::isfinite(minimum_transfer_angle) || minimum_transfer_angle <= 0.0 ||
            maximum_iterations == 0U || bracket_samples < 8U) {
            throw std::invalid_argument("Lambert solver options are invalid");
        }
    }
};

struct LambertSolution {
    Vector3 departure_velocity{};
    Vector3 arrival_velocity{};
    double transfer_angle{0.0};
    double universal_variable{0.0};
    double auxiliary_y{0.0};
    double time_of_flight{0.0};
    double time_residual{std::numeric_limits<double>::infinity()};
    std::size_t iterations{0};
};

namespace detail {

[[nodiscard]] inline double stumpff_c(const double z) {
    if (z > 1.0e-8) {
        const double root = std::sqrt(z);
        return (1.0 - std::cos(root)) / z;
    }
    if (z < -1.0e-8) {
        const double root = std::sqrt(-z);
        return (std::cosh(root) - 1.0) / (-z);
    }
    const double z2 = z * z;
    const double z3 = z2 * z;
    return 0.5 - z / 24.0 + z2 / 720.0 - z3 / 40'320.0;
}

[[nodiscard]] inline double stumpff_s(const double z) {
    if (z > 1.0e-8) {
        const double root = std::sqrt(z);
        return (root - std::sin(root)) / (root * root * root);
    }
    if (z < -1.0e-8) {
        const double root = std::sqrt(-z);
        return (std::sinh(root) - root) / (root * root * root);
    }
    const double z2 = z * z;
    const double z3 = z2 * z;
    return 1.0 / 6.0 - z / 120.0 + z2 / 5'040.0 - z3 / 362'880.0;
}

struct UniversalEvaluation {
    double z{0.0};
    double y{0.0};
    double time_of_flight{0.0};
    double residual{0.0};
};

[[nodiscard]] inline std::optional<UniversalEvaluation> evaluate_universal(
    const double z,
    const double radius_one,
    const double radius_two,
    const double a_parameter,
    const double gravitational_parameter,
    const double requested_time
) {
    const double c = stumpff_c(z);
    const double s = stumpff_s(z);
    if (!std::isfinite(c) || !std::isfinite(s) || c <= 0.0) {
        return std::nullopt;
    }
    const double y = radius_one + radius_two +
        a_parameter * (z * s - 1.0) / std::sqrt(c);
    if (!std::isfinite(y) || y <= 0.0) {
        return std::nullopt;
    }
    const double x = std::sqrt(y / c);
    const double time =
        (x * x * x * s + a_parameter * std::sqrt(y)) /
        std::sqrt(gravitational_parameter);
    if (!std::isfinite(time) || time <= 0.0) {
        return std::nullopt;
    }
    return UniversalEvaluation{z, y, time, time - requested_time};
}

}  // namespace detail

[[nodiscard]] inline LambertSolution solve_lambert_zero_revolution(
    const Vector3& departure_position,
    const Vector3& arrival_position,
    const double time_of_flight,
    const double gravitational_parameter,
    LambertOptions options = {}
) {
    options.validate();
    if (!std::isfinite(time_of_flight) || time_of_flight <= 0.0 ||
        !std::isfinite(gravitational_parameter) || gravitational_parameter <= 0.0) {
        throw std::invalid_argument("Lambert time and gravitational parameter must be positive");
    }
    const double radius_one = norm(departure_position);
    const double radius_two = norm(arrival_position);
    if (!std::isfinite(radius_one) || !std::isfinite(radius_two) ||
        radius_one <= 0.0 || radius_two <= 0.0) {
        throw std::invalid_argument("Lambert endpoint radii must be finite and positive");
    }
    const double cosine = std::clamp(
        dot(departure_position, arrival_position) / (radius_one * radius_two),
        -1.0,
        1.0
    );
    double transfer_angle = std::acos(cosine);
    const double cross_z = cross(departure_position, arrival_position)[2];
    if ((options.prograde && cross_z < 0.0) || (!options.prograde && cross_z >= 0.0)) {
        transfer_angle = 2.0 * std::numbers::pi - transfer_angle;
    }
    const double sine = std::sin(transfer_angle);
    const double one_minus_cosine = 1.0 - std::cos(transfer_angle);
    if (transfer_angle < options.minimum_transfer_angle ||
        2.0 * std::numbers::pi - transfer_angle < options.minimum_transfer_angle ||
        std::abs(sine) < options.minimum_transfer_angle ||
        one_minus_cosine <= 0.0) {
        throw std::invalid_argument("Lambert transfer angle is singular or nearly collinear");
    }
    const double a_parameter = sine *
        std::sqrt(radius_one * radius_two / one_minus_cosine);
    if (!std::isfinite(a_parameter) || std::abs(a_parameter) < 1.0e-14) {
        throw std::invalid_argument("Lambert geometry produced a singular A parameter");
    }

    const double edge = 4.0 * std::numbers::pi * std::numbers::pi - 1.0e-8;
    std::optional<detail::UniversalEvaluation> lower;
    std::optional<detail::UniversalEvaluation> upper;
    std::optional<detail::UniversalEvaluation> previous;
    for (std::size_t sample = 0; sample <= options.bracket_samples; ++sample) {
        const double fraction = static_cast<double>(sample) /
            static_cast<double>(options.bracket_samples);
        const double z = -edge + 2.0 * edge * fraction;
        const auto current = detail::evaluate_universal(
            z,
            radius_one,
            radius_two,
            a_parameter,
            gravitational_parameter,
            time_of_flight
        );
        if (!current.has_value()) {
            continue;
        }
        if (std::abs(current->residual) <= options.time_tolerance) {
            lower = current;
            upper = current;
            break;
        }
        if (previous.has_value() &&
            std::signbit(previous->residual) != std::signbit(current->residual)) {
            lower = previous;
            upper = current;
            break;
        }
        previous = current;
    }
    if (!lower.has_value() || !upper.has_value()) {
        throw std::runtime_error("Lambert zero-revolution branch could not bracket a solution");
    }

    detail::UniversalEvaluation root = *lower;
    std::size_t iterations = 0U;
    if (lower->z != upper->z) {
        for (; iterations < options.maximum_iterations; ++iterations) {
            const double midpoint = 0.5 * (lower->z + upper->z);
            const auto candidate = detail::evaluate_universal(
                midpoint,
                radius_one,
                radius_two,
                a_parameter,
                gravitational_parameter,
                time_of_flight
            );
            if (!candidate.has_value()) {
                lower->z = midpoint;
                continue;
            }
            root = *candidate;
            if (std::abs(root.residual) <= options.time_tolerance ||
                std::abs(upper->z - lower->z) <= options.universal_tolerance) {
                ++iterations;
                break;
            }
            if (std::signbit(lower->residual) != std::signbit(root.residual)) {
                upper = root;
            } else {
                lower = root;
            }
        }
    }
    if (std::abs(root.residual) > std::max(options.time_tolerance, 1.0e-10 * time_of_flight)) {
        throw std::runtime_error("Lambert solve did not meet the requested time tolerance");
    }

    const double f = 1.0 - root.y / radius_one;
    const double g = a_parameter * std::sqrt(root.y / gravitational_parameter);
    const double g_dot = 1.0 - root.y / radius_two;
    if (!std::isfinite(g) || std::abs(g) < 1.0e-12) {
        throw std::runtime_error("Lambert solution produced a singular Lagrange g coefficient");
    }
    Vector3 departure_velocity{};
    Vector3 arrival_velocity{};
    for (std::size_t component = 0; component < 3U; ++component) {
        departure_velocity[component] =
            (arrival_position[component] - f * departure_position[component]) / g;
        arrival_velocity[component] =
            (g_dot * arrival_position[component] - departure_position[component]) / g;
    }
    return LambertSolution{
        departure_velocity,
        arrival_velocity,
        transfer_angle,
        root.z,
        root.y,
        root.time_of_flight,
        root.residual,
        iterations,
    };
}

struct LambertScreeningOptions {
    std::size_t departure_samples{12};
    std::size_t arrival_samples{24};
    double minimum_flight_time{1.0};
    LambertOptions solver{};

    void validate() const {
        solver.validate();
        if (departure_samples == 0U || arrival_samples == 0U ||
            !std::isfinite(minimum_flight_time) || minimum_flight_time <= 0.0) {
            throw std::invalid_argument("Lambert screening options are invalid");
        }
    }
};

class LambertCircularOracle final : public ArcOracle {
  public:
    explicit LambertCircularOracle(LambertScreeningOptions options = {})
        : options_(options) {
        options_.validate();
    }

    [[nodiscard]] ArcResult evaluate(const ArcRequest& request) const override {
        request.validate();
        ArcResult best;
        best.fidelity = ArcFidelity::analytical;
        best.source_id = request.source.id;
        best.target_id = request.target.id;
        best.lower_bound = hohmann_transfer(
            request.source.radius,
            request.target.radius,
            request.source.gravitational_parameter
        ).total_delta_v();
        if (request.fidelity != ArcFidelity::analytical) {
            best.status = "Lambert screening is an analytical-fidelity oracle";
            return best;
        }
        if (request.source.id == request.target.id) {
            best.status = "source and target are identical";
            return best;
        }

        const auto departures = sample_window(request.departure_window, options_.departure_samples);
        const auto arrivals = sample_window(request.arrival_window, options_.arrival_samples);
        for (const double departure : departures) {
            const Vector3 source_position = circular_position(request.source, departure);
            const Vector3 source_velocity = circular_velocity(request.source, departure);
            for (const double arrival : arrivals) {
                const double flight_time = arrival - departure;
                if (flight_time < options_.minimum_flight_time) {
                    continue;
                }
                try {
                    const Vector3 target_position = circular_position(request.target, arrival);
                    const Vector3 target_velocity = circular_velocity(request.target, arrival);
                    const auto transfer = solve_lambert_zero_revolution(
                        source_position,
                        target_position,
                        flight_time,
                        request.source.gravitational_parameter,
                        options_.solver
                    );
                    const double departure_delta_v = norm(subtract(
                        transfer.departure_velocity,
                        source_velocity
                    ));
                    const double arrival_delta_v = norm(subtract(
                        target_velocity,
                        transfer.arrival_velocity
                    ));
                    const double total_delta_v = departure_delta_v + arrival_delta_v;
                    const double propellant =
                        request.spacecraft.propellant_for_delta_v(total_delta_v);
                    if (total_delta_v < best.delta_v &&
                        total_delta_v <= request.spacecraft.available_delta_v() &&
                        propellant <= request.spacecraft.propellant_mass) {
                        best.feasible = true;
                        best.departure_epoch = departure;
                        best.arrival_epoch = arrival;
                        best.flight_time = flight_time;
                        best.delta_v = total_delta_v;
                        best.propellant_required = propellant;
                        best.phase_error = 0.0;
                        best.status = "Lambert zero-revolution transfer found";
                        best.warm_start_token = "lambert:" + request.source.id + ":" +
                            request.target.id + ":" + std::to_string(departure) + ":" +
                            std::to_string(arrival);
                    }
                } catch (const std::runtime_error&) {
                    continue;
                } catch (const std::invalid_argument&) {
                    continue;
                }
            }
        }
        if (!best.feasible) {
            best.status = "no feasible zero-revolution Lambert sample in the epoch windows";
        }
        return best;
    }

  private:
    [[nodiscard]] static std::vector<double> sample_window(
        const EpochWindow& window,
        const std::size_t samples
    ) {
        window.validate();
        if (samples == 1U || window.earliest == window.latest) {
            return {window.earliest};
        }
        std::vector<double> epochs(samples);
        for (std::size_t index = 0; index < samples; ++index) {
            const double fraction = static_cast<double>(index) /
                static_cast<double>(samples - 1U);
            epochs[index] = window.earliest + fraction * (window.latest - window.earliest);
        }
        return epochs;
    }

    LambertScreeningOptions options_;
};

}  // namespace spacepdhcg::orbitweaver
