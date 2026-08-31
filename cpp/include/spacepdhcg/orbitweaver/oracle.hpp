#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace spacepdhcg::orbitweaver {

inline constexpr double standard_gravity = 9.80665;

enum class ArcFidelity {
    analytical,
    coarse_convex,
    refined_scvx,
    robust_scvx,
    certified,
};

struct EpochWindow {
    double earliest{0.0};
    double latest{0.0};

    void validate() const {
        if (!std::isfinite(earliest) || !std::isfinite(latest) || earliest > latest) {
            throw std::invalid_argument("epoch window must be finite and ordered");
        }
    }

    [[nodiscard]] std::optional<EpochWindow> intersect(const EpochWindow& other) const {
        validate();
        other.validate();
        const EpochWindow result{
            std::max(earliest, other.earliest),
            std::min(latest, other.latest),
        };
        if (result.earliest > result.latest) {
            return std::nullopt;
        }
        return result;
    }
};

struct CircularOrbitTarget {
    std::string id;
    double radius{0.0};
    double phase_at_epoch_zero{0.0};
    double gravitational_parameter{0.0};

    void validate() const {
        if (id.empty()) {
            throw std::invalid_argument("orbit target id may not be empty");
        }
        if (!std::isfinite(radius) || radius <= 0.0 ||
            !std::isfinite(phase_at_epoch_zero) ||
            !std::isfinite(gravitational_parameter) || gravitational_parameter <= 0.0) {
            throw std::invalid_argument("circular orbit target parameters are invalid");
        }
    }

    [[nodiscard]] double mean_motion() const {
        validate();
        return std::sqrt(gravitational_parameter / (radius * radius * radius));
    }

    [[nodiscard]] double phase(const double epoch) const {
        if (!std::isfinite(epoch)) {
            throw std::invalid_argument("orbit epoch must be finite");
        }
        return phase_at_epoch_zero + mean_motion() * epoch;
    }
};

struct SpacecraftResources {
    double dry_mass{0.0};
    double propellant_mass{0.0};
    double specific_impulse{0.0};

    void validate() const {
        if (!std::isfinite(dry_mass) || dry_mass <= 0.0 ||
            !std::isfinite(propellant_mass) || propellant_mass < 0.0 ||
            !std::isfinite(specific_impulse) || specific_impulse <= 0.0) {
            throw std::invalid_argument("spacecraft resource parameters are invalid");
        }
    }

    [[nodiscard]] double wet_mass() const {
        validate();
        return dry_mass + propellant_mass;
    }

    [[nodiscard]] double available_delta_v() const {
        validate();
        if (propellant_mass == 0.0) {
            return 0.0;
        }
        return standard_gravity * specific_impulse *
            std::log(wet_mass() / dry_mass);
    }

    [[nodiscard]] double propellant_for_delta_v(const double delta_v) const {
        validate();
        if (!std::isfinite(delta_v) || delta_v < 0.0) {
            throw std::invalid_argument("delta-v must be finite and non-negative");
        }
        if (delta_v == 0.0) {
            return 0.0;
        }
        return wet_mass() *
            (1.0 - std::exp(-delta_v / (standard_gravity * specific_impulse)));
    }
};

struct ArcRequest {
    CircularOrbitTarget source;
    CircularOrbitTarget target;
    EpochWindow departure_window;
    EpochWindow arrival_window;
    SpacecraftResources spacecraft;
    ArcFidelity fidelity{ArcFidelity::analytical};
    double phase_tolerance{1.0e-8};

    void validate() const {
        source.validate();
        target.validate();
        departure_window.validate();
        arrival_window.validate();
        spacecraft.validate();
        if (source.gravitational_parameter != target.gravitational_parameter) {
            throw std::invalid_argument("analytical circular transfer requires one central body");
        }
        if (!std::isfinite(phase_tolerance) || phase_tolerance <= 0.0 ||
            phase_tolerance >= std::numbers::pi) {
            throw std::invalid_argument("phase tolerance must lie in (0,pi)");
        }
    }
};

struct ArcResult {
    bool feasible{false};
    ArcFidelity fidelity{ArcFidelity::analytical};
    std::string source_id;
    std::string target_id;
    double departure_epoch{std::numeric_limits<double>::quiet_NaN()};
    double arrival_epoch{std::numeric_limits<double>::quiet_NaN()};
    double flight_time{std::numeric_limits<double>::quiet_NaN()};
    double delta_v{std::numeric_limits<double>::infinity()};
    double propellant_required{std::numeric_limits<double>::infinity()};
    double lower_bound{std::numeric_limits<double>::infinity()};
    double phase_error{std::numeric_limits<double>::infinity()};
    std::string status;
    std::string warm_start_token;
};

class ArcOracle {
  public:
    ArcOracle(const ArcOracle&) = delete;
    ArcOracle& operator=(const ArcOracle&) = delete;
    ArcOracle(ArcOracle&&) = delete;
    ArcOracle& operator=(ArcOracle&&) = delete;
    virtual ~ArcOracle() = default;

    [[nodiscard]] virtual ArcResult evaluate(const ArcRequest& request) const = 0;

    [[nodiscard]] virtual std::vector<ArcResult> evaluate_batch(
        std::span<const ArcRequest> requests
    ) const {
        std::vector<ArcResult> results;
        results.reserve(requests.size());
        for (const auto& request : requests) {
            results.push_back(evaluate(request));
        }
        return results;
    }

  protected:
    ArcOracle() = default;
};

struct HohmannTransfer {
    double flight_time{0.0};
    double departure_delta_v{0.0};
    double arrival_delta_v{0.0};

    [[nodiscard]] double total_delta_v() const noexcept {
        return departure_delta_v + arrival_delta_v;
    }
};

[[nodiscard]] inline HohmannTransfer hohmann_transfer(
    const double source_radius,
    const double target_radius,
    const double gravitational_parameter
) {
    if (!std::isfinite(source_radius) || source_radius <= 0.0 ||
        !std::isfinite(target_radius) || target_radius <= 0.0 ||
        !std::isfinite(gravitational_parameter) || gravitational_parameter <= 0.0) {
        throw std::invalid_argument("Hohmann transfer parameters must be finite and positive");
    }
    if (source_radius == target_radius) {
        return HohmannTransfer{};
    }
    const double semi_major_axis = 0.5 * (source_radius + target_radius);
    const double source_circular_speed = std::sqrt(gravitational_parameter / source_radius);
    const double target_circular_speed = std::sqrt(gravitational_parameter / target_radius);
    const double transfer_source_speed = std::sqrt(
        gravitational_parameter * (2.0 / source_radius - 1.0 / semi_major_axis)
    );
    const double transfer_target_speed = std::sqrt(
        gravitational_parameter * (2.0 / target_radius - 1.0 / semi_major_axis)
    );
    return HohmannTransfer{
        std::numbers::pi *
            std::sqrt(semi_major_axis * semi_major_axis * semi_major_axis /
                      gravitational_parameter),
        std::abs(transfer_source_speed - source_circular_speed),
        std::abs(target_circular_speed - transfer_target_speed),
    };
}

[[nodiscard]] inline double wrap_to_pi(const double angle) noexcept {
    double wrapped = std::remainder(angle, 2.0 * std::numbers::pi);
    if (wrapped <= -std::numbers::pi) {
        wrapped += 2.0 * std::numbers::pi;
    }
    return wrapped;
}

class AnalyticalCircularOracle final : public ArcOracle {
  public:
    [[nodiscard]] ArcResult evaluate(const ArcRequest& request) const override {
        request.validate();
        ArcResult result;
        result.fidelity = ArcFidelity::analytical;
        result.source_id = request.source.id;
        result.target_id = request.target.id;

        if (request.fidelity != ArcFidelity::analytical) {
            result.status = "requested fidelity requires a numerical trajectory backend";
            return result;
        }
        if (request.source.id == request.target.id) {
            result.status = "source and target are identical";
            return result;
        }
        if (request.source.radius == request.target.radius) {
            result.status = "equal-radius phasing is not represented by the Hohmann lower bound";
            result.lower_bound = 0.0;
            return result;
        }

        const auto transfer = hohmann_transfer(
            request.source.radius,
            request.target.radius,
            request.source.gravitational_parameter
        );
        result.flight_time = transfer.flight_time;
        result.delta_v = transfer.total_delta_v();
        result.lower_bound = result.delta_v;
        result.propellant_required = request.spacecraft.propellant_for_delta_v(result.delta_v);
        if (result.propellant_required > request.spacecraft.propellant_mass) {
            result.status = "insufficient propellant for analytical transfer";
            return result;
        }

        const EpochWindow arrival_implied_departure{
            request.arrival_window.earliest - transfer.flight_time,
            request.arrival_window.latest - transfer.flight_time,
        };
        const auto feasible_departure_window =
            request.departure_window.intersect(arrival_implied_departure);
        if (!feasible_departure_window.has_value()) {
            result.status = "departure and arrival windows do not overlap after flight time";
            return result;
        }

        const auto phased_departure = find_phased_departure(
            request.source,
            request.target,
            transfer.flight_time,
            *feasible_departure_window,
            request.phase_tolerance
        );
        if (!phased_departure.has_value()) {
            result.status = "no phase-compatible departure exists inside the epoch windows";
            return result;
        }

        result.departure_epoch = *phased_departure;
        result.arrival_epoch = result.departure_epoch + transfer.flight_time;
        result.phase_error = phase_error(
            request.source,
            request.target,
            result.departure_epoch,
            transfer.flight_time
        );
        result.feasible = std::abs(result.phase_error) <= request.phase_tolerance;
        result.status = result.feasible ? "feasible analytical Hohmann arc" : "phase residual exceeded tolerance";
        result.warm_start_token = make_token(request, result);
        return result;
    }

  private:
    [[nodiscard]] static double phase_error(
        const CircularOrbitTarget& source,
        const CircularOrbitTarget& target,
        const double departure_epoch,
        const double flight_time
    ) {
        return wrap_to_pi(
            target.phase(departure_epoch + flight_time) -
            source.phase(departure_epoch) - std::numbers::pi
        );
    }

    [[nodiscard]] static std::optional<double> find_phased_departure(
        const CircularOrbitTarget& source,
        const CircularOrbitTarget& target,
        const double flight_time,
        const EpochWindow& window,
        const double tolerance
    ) {
        window.validate();
        const double relative_rate = target.mean_motion() - source.mean_motion();
        const double offset = target.phase_at_epoch_zero + target.mean_motion() * flight_time -
            source.phase_at_epoch_zero - std::numbers::pi;
        if (std::abs(relative_rate) <= 1.0e-15) {
            return std::abs(wrap_to_pi(offset)) <= tolerance
                ? std::optional<double>{window.earliest}
                : std::nullopt;
        }

        const double endpoint_a = offset + relative_rate * window.earliest;
        const double endpoint_b = offset + relative_rate * window.latest;
        const double minimum = std::min(endpoint_a, endpoint_b);
        const double maximum = std::max(endpoint_a, endpoint_b);
        const auto first_revolution = static_cast<long long>(
            std::ceil(minimum / (2.0 * std::numbers::pi) - 1.0e-12)
        );
        const auto last_revolution = static_cast<long long>(
            std::floor(maximum / (2.0 * std::numbers::pi) + 1.0e-12)
        );
        std::optional<double> best;
        for (long long revolution = first_revolution; revolution <= last_revolution; ++revolution) {
            const double departure =
                (2.0 * std::numbers::pi * static_cast<double>(revolution) - offset) /
                relative_rate;
            if (departure < window.earliest - tolerance || departure > window.latest + tolerance) {
                continue;
            }
            if (!best.has_value() || departure < *best) {
                best = std::clamp(departure, window.earliest, window.latest);
            }
        }
        return best;
    }

    [[nodiscard]] static std::string make_token(
        const ArcRequest& request,
        const ArcResult& result
    ) {
        std::uint64_t hash = 1469598103934665603ULL;
        const auto mix = [&hash](std::string_view text) {
            for (const unsigned char value : text) {
                hash ^= value;
                hash *= 1099511628211ULL;
            }
        };
        mix(request.source.id);
        mix("->");
        mix(request.target.id);
        mix(std::to_string(result.departure_epoch));
        mix(std::to_string(result.arrival_epoch));
        mix(std::to_string(result.delta_v));
        return "analytical-" + std::to_string(hash);
    }
};

}  // namespace spacepdhcg::orbitweaver
