#include "spacepdhcg/native/persistent_session.hpp"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace native = spacepdhcg::native;

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct Counters {
    std::size_t created{0};
    std::size_t updates{0};
    std::size_t warm_starts{0};
    std::size_t solves{0};
    std::size_t clears{0};
};

class FakeWorkspace final : public native::CqpWorkspace {
  public:
    FakeWorkspace(native::OwnedCqp initial, std::shared_ptr<Counters> counters)
        : current_(std::move(initial)), counters_(std::move(counters)) {
        ++counters_->created;
    }

    [[nodiscard]] std::string backend_name() const override { return "fake-persistent"; }
    [[nodiscard]] bool persistent() const noexcept override { return true; }

    void update(const native::OwnedCqp& problem) override {
        current_ = problem;
        ++counters_->updates;
    }

    void warm_start(
        std::span<const double> primal,
        std::span<const double> dual
    ) override {
        require(primal.size() == static_cast<std::size_t>(current_.variables()),
                "fake primal warm start has the wrong dimension");
        require(dual.size() == static_cast<std::size_t>(
                    current_.scalar_constraint.rows + current_.affine_cone.rows
                ),
                "fake dual warm start has the wrong dimension");
        ++counters_->warm_starts;
    }

    void clear_warm_start() override { ++counters_->clears; }

    [[nodiscard]] native::CqpSolveResult solve(
        const native::CqpSolveOptions& options
    ) override {
        options.validate();
        ++counters_->solves;
        std::vector<double> primal(static_cast<std::size_t>(current_.variables()), 0.0);
        std::vector<double> dual(
            static_cast<std::size_t>(
                current_.scalar_constraint.rows + current_.affine_cone.rows
            ),
            0.0
        );
        return native::CqpSolveResult{
            native::CqpSolveStatus::solved,
            std::move(primal),
            std::move(dual),
            0.0,
            0.0,
            0.0,
            1,
            0.0,
            0.0,
            "fake exact solve",
        };
    }

  private:
    native::OwnedCqp current_{};
    std::shared_ptr<Counters> counters_{};
};

native::OwnedCqp make_zero_problem() {
    native::CscBuilder quadratic(2, 2);
    quadratic.add(0, 0, 1.0);
    quadratic.add(1, 1, 1.0);
    native::CscBuilder scalar(1, 2);
    scalar.add(0, 0, 1.0);
    scalar.add(0, 1, 1.0);
    native::CscBuilder affine(0, 2);
    return native::OwnedCqp{
        quadratic.build(),
        scalar.build(),
        affine.build(),
        {0.0, 0.0},
        {0.0},
        {0.0},
        {},
        {-1.0, -1.0},
        {1.0, 1.0},
        {},
        {},
    };
}

void test_persistent_session() {
    auto counters = std::make_shared<Counters>();
    auto problem = make_zero_problem();
    native::PersistentCqpSession session(
        problem,
        [counters](const native::OwnedCqp& initial) {
            return std::make_unique<FakeWorkspace>(initial, counters);
        }
    );
    require(session.backend_name() == "fake-persistent", "session backend name is wrong");
    require(session.backend_is_persistent(), "session lost the persistent capability flag");

    native::CqpSolveOptions options{};
    options.optimality_tolerance = 1.0e-8;
    options.feasibility_tolerance = 1.0e-8;
    const auto first = session.solve(problem, options);
    require(first.acceptable(1.0e-8), "first session result failed independent checks");
    require(!first.warm_started, "first solve should not have a previous warm start");

    problem.linear[0] = 0.25;
    const auto second = session.solve(problem, options);
    require(second.acceptable(1.0e-8), "second session result failed independent checks");
    require(second.warm_started, "second solve did not reuse the prior primal-dual point");
    require(session.update_count() == 2 && session.solve_count() == 2,
            "session lifecycle counters are wrong");
    require(counters->created == 1 && counters->updates == 2 && counters->solves == 2,
            "backend workspace was reconstructed or skipped");
    require(counters->warm_starts == 1, "backend did not receive exactly one warm start");

    session.clear_warm_start();
    require(!session.has_warm_start(), "session warm start was not cleared");
    require(counters->clears == 1, "backend warm-start clear was not propagated");
}

}  // namespace

int main() {
    test_persistent_session();
    return 0;
}
