#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    int32_t abi_version, ratio_inflation, compensated_sum, reserved;
    double thrust, day_seconds, exhaust_velocity, authority_ratio;
    double inflation, floor, slope, miner_mass, initial_mass;
    double arrival_horizon, mining_horizon, mining_rate, year_days;
    double propellant_weight, time_weight, lookahead_weight;
} spacepdhcg_gtoc12_expansion_policy;
typedef struct {
    int32_t deploy_begin, deploy_count;
    double mass, epoch, launch, hop_propellant, lookahead;
} spacepdhcg_gtoc12_expansion_parent;
typedef struct {
    int64_t body;
    double epoch, weight, price;
} spacepdhcg_gtoc12_expansion_deploy;
typedef struct {
    int32_t parent, allowed;
    int64_t target;
    double departure, tof, delta_v, lookahead, weight, price, cluster_bonus;
} spacepdhcg_gtoc12_expansion_option;
typedef struct {
    int32_t parent, valid;
    int64_t target;
    double departure, arrival, delta_v, inflation, propellant, mass, score;
    double lookahead, hop_propellant;
} spacepdhcg_gtoc12_expansion_result;
/* Status 0 success, 1 invalid input, 2 CUDA/allocation failure, 3 busy.
 * Create/destroy and all calls require the selected CUDA device. Workspaces
 * retain capacity; rank uploads one depth, evaluates and sorts on CUDA, then
 * reads only its valid count. Read copies a requested slice of ranked results.
 * Ties order by score descending, epoch ascending, deployed ID sequence, then
 * original option order. Input partitions are validated; failed ranks invalidate
 * the previous result, so read cannot expose stale candidates. No dynamics or
 * original authority/mass tolerances are changed by these surrogate operators.
 */
int spacepdhcg_gtoc12_expansion_create(int32_t device, int32_t parents,
    int32_t deploys, int32_t options, void** workspace);
int spacepdhcg_gtoc12_expansion_destroy(void** workspace);
int spacepdhcg_gtoc12_expansion_rank(void* workspace,
    const spacepdhcg_gtoc12_expansion_policy* policy,
    int32_t parent_count, const spacepdhcg_gtoc12_expansion_parent* parents,
    int32_t deploy_count, const spacepdhcg_gtoc12_expansion_deploy* deploys,
    int32_t option_count, const spacepdhcg_gtoc12_expansion_option* options,
    int32_t* valid_count);
int spacepdhcg_gtoc12_expansion_read(void* workspace, int32_t offset,
    int32_t count, spacepdhcg_gtoc12_expansion_result* results);
#ifdef __cplusplus
}
#endif
