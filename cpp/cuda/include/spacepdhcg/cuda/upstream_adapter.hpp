/*
 * Linked-reference metadata for pinned PDHCG.
 *
 * The persistent hot path does not invoke the one-shot API. Tests link the
 * same pinned target and use this metadata in evidence.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <string_view>

namespace spacepdhcg::cuda {

inline constexpr std::string_view pinned_pdhcg_commit =
    "167c8b72b4b96d2f94d405b8763e485514192b81";
inline constexpr std::string_view pinned_pdhcg_tree =
    "62b05e6c1bedd385f6c267af3645ae4aae0421b4";
inline constexpr std::string_view pinned_pdhcg_patch_set =
    "0001-free-quadratic-state.patch:"
    "7f212ac5ef6afa96b7084092bfcff80602c2c976e1a7a9f3305d1817cb7554f4";
inline constexpr std::string_view persistent_integration_strategy =
    "linked_internal_adapter";

}  // namespace spacepdhcg::cuda
