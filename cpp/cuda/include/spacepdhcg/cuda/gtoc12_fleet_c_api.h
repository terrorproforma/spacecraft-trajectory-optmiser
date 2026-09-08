#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    double value, mass;
    int32_t ships, reserved;
    int64_t identifier;
} spacepdhcg_gtoc12_fleet_column;
typedef struct {
    double objective, upper_bound, greedy_objective;
    uint64_t nodes;
    int32_t exhaustive, tasks;
} spacepdhcg_gtoc12_fleet_report;
typedef struct {
    uint64_t proposals;
    int32_t moves, rounds;
} spacepdhcg_gtoc12_fleet_exchange_report;
/* Conflicts and each foreign requirement's alternative providers use CSR.
 * requirement_offsets maps columns to requirement groups; provider_offsets maps
 * groups to providers. All arrays are host resident. Output is one byte/column.
 * Inputs describe already certified columns, with the same physics/bonus table.
 * No CPU optimisation or LP fallback is performed. Returns 0/invalid=1/CUDA=2.
 */
int spacepdhcg_gtoc12_fleet_search_host(
    int32_t columns, int32_t max_ships, int32_t prefix_bits, uint64_t node_cap,
    const spacepdhcg_gtoc12_fleet_column* data,
    const int32_t* conflict_offsets, const int32_t* conflicts,
    const int32_t* requirement_offsets, const int32_t* provider_offsets,
    const int32_t* providers, const uint8_t* incumbent,
    uint8_t* selected, spacepdhcg_gtoc12_fleet_report* report);
/* Versioned extension: GPU single-column additions, removals and exchanges
 * improve the best seed before branch-and-bound. The legacy ABI is unchanged.
 * proposals counts screened exchange combinations separately from B&B nodes.
 */
int spacepdhcg_gtoc12_fleet_search_v2_host(
    int32_t columns, int32_t max_ships, int32_t prefix_bits, uint64_t node_cap,
    const spacepdhcg_gtoc12_fleet_column* data,
    const int32_t* conflict_offsets, const int32_t* conflicts,
    const int32_t* requirement_offsets, const int32_t* provider_offsets,
    const int32_t* providers, const uint8_t* incumbent,
    uint8_t* selected, spacepdhcg_gtoc12_fleet_report* report,
    int32_t exchange_rounds, spacepdhcg_gtoc12_fleet_exchange_report* exchange);
/* Retained immutable pool. Creation copies all input arrays and ranks columns
 * once on the current CUDA device. Solve uploads only the incumbent mask; all
 * search buffers remain allocated until destroy. Create a new workspace when
 * columns, weights or prefix_bits change. Calls on a workspace must be serial
 * and solve must use the creation device. Destroy(NULL) is a no-op.
 * Conflict rows must be symmetric, unique and exclude their own column.
 */
int spacepdhcg_gtoc12_fleet_workspace_create_host(
    int32_t columns, int32_t prefix_bits,
    const spacepdhcg_gtoc12_fleet_column* data,
    const int32_t* conflict_offsets, const int32_t* conflicts,
    const int32_t* requirement_offsets, const int32_t* provider_offsets,
    const int32_t* providers, void** workspace);
int spacepdhcg_gtoc12_fleet_workspace_solve_host(
    void* workspace, int32_t max_ships, uint64_t node_cap,
    const uint8_t* incumbent, uint8_t* selected,
    spacepdhcg_gtoc12_fleet_report* report, int32_t exchange_rounds,
    spacepdhcg_gtoc12_fleet_exchange_report* exchange);
void spacepdhcg_gtoc12_fleet_workspace_destroy_host(void* workspace);
typedef struct { int64_t asteroid; double epoch; } spacepdhcg_gtoc12_fleet_event;
typedef struct { double mass, weight; } spacepdhcg_gtoc12_fleet_mass;
/* Build the pool on CUDA from serialized route records. metadata supplies ships,
 * identifier and reserved=certified (0/1); its value/mass fields are ignored.
 * Each offsets array has columns+1 entries. Mass records preserve source order.
 * CUDA scores, filters, sorts and constructs conflict/provider CSR. permutation
 * receives usable_count original row indices, in canonical search order.
 * All arrays are host resident; outputs must hold columns indices and one count.
 * Missing foreign providers are checked against ALL certified input columns,
 * matching the existing single-pass eligibility policy (not recursive closure).
 */
int spacepdhcg_gtoc12_fleet_workspace_create_routes_host(
    int32_t columns, int32_t prefix_bits,
    const spacepdhcg_gtoc12_fleet_column* metadata,
    const int32_t* deploy_offsets, const spacepdhcg_gtoc12_fleet_event* deploys,
    const int32_t* collect_offsets, const int64_t* collects,
    const int32_t* foreign_offsets, const spacepdhcg_gtoc12_fleet_event* foreign,
    const int32_t* mass_offsets, const spacepdhcg_gtoc12_fleet_mass* masses,
    void** workspace, int32_t* permutation, int32_t* usable_count);
#ifdef __cplusplus
}
#endif
