#pragma once

#include <cuda_runtime.h>
#include <cstdint>

struct QocoAuditCsc {
    int rows, columns, nonzeros;
    const int* offsets;
    const int* indices;
    const double* values;
};

struct QocoAuditInput {
    QocoAuditCsc quadratic, equality, conic, dual_map;
    const double* objective;
    const double* equality_rhs;
    const double* conic_rhs;
    int nonnegative, soc_count;
    const int* soc_sizes;
};

struct QocoAuditResult {
    double primal, dual, absolute_primal, absolute_dual, dual_cone, complementarity;
};
struct QocoReplayStatus { int status, iterations; };

struct QocoAuditTransfers {
    std::uint64_t h2d_count{}, h2d_bytes{}, d2h_count{}, d2h_bytes{};
};

// Audit-owned allocations only, excluding opaque QOCO/cuDSS workspaces.
struct QocoAuditMemory {
    std::uint64_t allocations{}, bytes{}, peak_bytes{};
};

struct QocoGpuAudit;

cudaError_t qoco_gpu_audit_create(const QocoAuditInput&, bool host_solution,
                                cudaStream_t, QocoGpuAudit**);
cudaError_t qoco_gpu_audit_update(QocoGpuAudit*, const QocoAuditInput&, cudaStream_t);
cudaError_t qoco_gpu_audit_update_device(QocoGpuAudit*, const double* packed_values, cudaStream_t);
cudaError_t qoco_gpu_audit_upload_solution(QocoGpuAudit*, const double*, const double*,
                                         const double*, cudaStream_t,
                                         const double**, const double**, const double**);
// Inputs are unscaled device vectors. The mapped dual is written on the same
// stream. Only the six audit scalars are downloaded; no CPU numerical work.
cudaError_t qoco_gpu_audit_run(QocoGpuAudit*, const double*, const double*, const double*,
                             double* mapped_dual, cudaStream_t, QocoAuditResult*);
// Queue the same independent audit and return retained device scalars. No host
// download or wait. Consume on the same stream before updating/reusing the audit.
cudaError_t qoco_gpu_audit_run_device(QocoGpuAudit*, const double*, const double*, const double*,
                                    double* mapped_dual, cudaStream_t, const QocoAuditResult**);
// Queue an explicitly requested host report; caller owns destination lifetime
// and completes the stream. Transfer accounting remains owned by the audit.
cudaError_t qoco_gpu_audit_download_async(QocoGpuAudit*, cudaStream_t, QocoAuditResult*);
// Validate completion ABI v1 on device and retain only the two fields needed by
// transitional host dispatch. An unknown ABI produces status -1.
cudaError_t qoco_gpu_audit_replay_status(QocoGpuAudit*, const int* completion_header,
                                      cudaStream_t, const QocoReplayStatus**);
// Bootstrap bridge for a synchronous priming solve; subsequent replay status
// comes directly from the device completion packet.
cudaError_t qoco_gpu_audit_publish_status(QocoGpuAudit*, int status, int iterations,
                                       cudaStream_t, const QocoReplayStatus**);
const QocoAuditResult* qoco_gpu_audit_device_result(const QocoGpuAudit*);
QocoAuditTransfers qoco_gpu_audit_transfers(const QocoGpuAudit*);
QocoAuditMemory qoco_gpu_audit_memory(const QocoGpuAudit*);
void qoco_gpu_audit_destroy(QocoGpuAudit*);

struct QocoTopologyInput {
    int counts[6]{};
    const int* arrays[6]{};
};
struct QocoGpuTopology;
// Create from host topology once; subsequent validation compares device arrays
// exactly and downloads a single mismatch flag, rather than the sparse indices.
cudaError_t qoco_gpu_topology_create(const QocoTopologyInput&, cudaStream_t, QocoGpuTopology**);
cudaError_t qoco_gpu_topology_validate(QocoGpuTopology*, const QocoTopologyInput&,
                                     cudaStream_t, bool* match);
QocoAuditTransfers qoco_gpu_topology_transfers(const QocoGpuTopology*);
QocoAuditMemory qoco_gpu_topology_memory(const QocoGpuTopology*);
void qoco_gpu_topology_destroy(QocoGpuTopology*);

// Canonical arrays: Q, A, F, c, scalar lower/upper, affine offset, variable lower/upper.
// A term is scale * (input[index] + other_scale * input[other]); other=-1 omits
// the second operand, input=-1 denotes the constant scale. Output rows sum terms
// in their compiled order, preserving duplicate sparse entries and SOC transforms.
struct QocoConversionTerm { int input, index, other; double scale, other_scale; };
struct QocoConversionPair { int first, second; };
struct QocoConversionPlan {
    int input_counts[9]{};
    int outputs{}, terms{}, symmetry_pairs{};
    const int* offsets{};
    const QocoConversionTerm* entries{};
    const int* bound_types{}; // scalar then variable; 0=free, 1=upper, 2=lower, 3=both, 4=equality
    const QocoConversionPair* symmetry{};
};
struct QocoConversionInputs { const double* arrays[9]{}; };
struct QocoGpuConversion;
cudaError_t qoco_gpu_conversion_create(const QocoConversionPlan&, cudaStream_t, QocoGpuConversion**);
// invalid: bit 1=changed bound classification, bit 2=nonfinite arithmetic,
// bit 4=asymmetric quadratic. A null host_output retains values only on the device.
cudaError_t qoco_gpu_conversion_run(QocoGpuConversion*, const QocoConversionInputs&,
    double* host_output, int* invalid, cudaStream_t);
const double* qoco_gpu_conversion_values(const QocoGpuConversion*);
QocoAuditTransfers qoco_gpu_conversion_transfers(const QocoGpuConversion*);
QocoAuditMemory qoco_gpu_conversion_memory(const QocoGpuConversion*);
void qoco_gpu_conversion_destroy(QocoGpuConversion*);
