#include "spacepdhcg/persistent_cqp.hpp"

#include <array>
#include <type_traits>

using spacepdhcg::ConeBlockDescriptor;
using spacepdhcg::ConeKind;
using spacepdhcg::HostConstSpan;
using spacepdhcg::Index;
using spacepdhcg::SparseFormat;
using spacepdhcg::SparsePatternView;
using spacepdhcg::StructureDescriptor;

static_assert(!std::is_copy_constructible_v<spacepdhcg::PersistentCQP>);
static_assert(std::is_trivially_copyable_v<SparsePatternView>);
static_assert(std::is_trivially_copyable_v<ConeBlockDescriptor>);

int main() {
    constexpr std::array<Index, 3> offsets{0, 1, 2};
    constexpr std::array<Index, 2> indices{0, 1};
    constexpr std::array<ConeBlockDescriptor, 1> cones{
        ConeBlockDescriptor{ConeKind::second_order, 0, 2, 0.0},
    };

    constexpr SparsePatternView identity{
        SparseFormat::csc,
        2,
        2,
        HostConstSpan<Index>{offsets.data(), offsets.size()},
        HostConstSpan<Index>{indices.data(), indices.size()},
    };
    static_assert(identity.well_formed());
    static_assert(identity.nonzeros() == 2);

    const StructureDescriptor structure{
        2,
        identity,
        identity,
        SparsePatternView{},
        HostConstSpan<ConeBlockDescriptor>{cones.data(), cones.size()},
        HostConstSpan<ConeBlockDescriptor>{},
    };

    return structure.variables == 2 ? 0 : 1;
}
