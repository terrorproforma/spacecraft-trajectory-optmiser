// SPDX-License-Identifier: Apache-2.0
#pragma once
// Updated before each retained graph replay; never capture these as constants.
struct QocoIpmParameters {
    double k, kinv, absolute, relative, inaccurate_absolute, inaccurate_relative;
    double ir_tolerance;
    int maximum, ir_maximum;
};
static __global__ void qoco_ipm_parameters_set(QocoIpmParameters* target, QocoIpmParameters value) {
    *target = value;
}
static __global__ void qoco_ipm_scale(double* values, int count, const QocoIpmParameters* p) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x)
        values[i] = __dmul_rn(values[i], p->kinv);
}
