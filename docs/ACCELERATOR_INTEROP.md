# Accelerator pointer and DLPack interoperability contract

## 1. Purpose

This document freezes the host/framework-to-SpacePDHCG zero-copy boundary before the persistent
CUDA workspace is implemented. The boundary is represented in C++ by
`core/accelerator_view.hpp`, in C by `accelerator_c_api.h`, and optionally converted from versioned
DLPack managed tensors by `dlpack_adapter.hpp`.

The contract answers, without reference to a particular Python framework:

- which device executes the work;
- which memory class stores each pointer;
- scalar type, length, byte offset, and stride;
- whether SpacePDHCG may mutate the storage;
- which CUDA stream consumes the storage;
- who owns the allocation and when its deleter is called;
- which sparse topology fingerprint the buffers belong to;
- which buffers are immutable for workspace lifetime and which may change every SCvx iteration.

It does **not** claim that the current repository has executed a CUDA pointer. Validation of actual
device accessibility, stream ordering, and lifetime remains issue #2.

## 2. Versioning

The exchange ABI version is

```text
SPACEPDHCG_ACCELERATOR_EXCHANGE_ABI_VERSION = 1
```

An incompatible struct or semantic change increments this integer. Adding an optional field through
a separately versioned extension does not silently reinterpret version 1.

DLPack is optional. The build is pinned for reproducibility to the metadata in
`third_party/dlpack.lock.json`. The adapter accepts only the current versioned managed-tensor ABI
with matching major version. A minor-version increase is acceptable only when the encountered
device, dtype, flags, and fields are already understood.

## 3. Execution device versus storage device

These are distinct concepts:

- `AcceleratorStream.device` is always an ordinary CUDA execution device, such as CUDA device 0 or
  device 3. Its native handle is interpreted as a `cudaStream_t` for that device.
- Every `AcceleratorBufferView.device` describes the storage memory class.

Version 1 recognises:

| SpacePDHCG code | DLPack code | Meaning |
|---|---:|---|
| `cpu` | `kDLCPU = 1` | ordinary host memory |
| `cuda` | `kDLCUDA = 2` | CUDA device allocation |
| `cuda_host` | `kDLCUDAHost = 3` | CUDA-pinned host allocation |
| `cuda_managed` | `kDLCUDAManaged = 13` | CUDA managed allocation |

A persistent solve exchange accepts one of two storage modes:

1. ordinary CUDA storage whose device id exactly matches the consumer stream's CUDA device id;
2. CUDA-managed storage, represented by DLPack device id zero, consumed on the declared ordinary
   CUDA stream device.

Pinned host views are staging views, not persistent solver state. CPU storage is not a GPU-resident
exchange. One exchange may not mix CUDA and managed storage or several CUDA device ids. Multi-GPU
ownership is represented by one exchange per local shard.

## 4. Supported scalar types

Version 1 recognises:

- signed `int32` for current CSC offsets and row indices;
- signed `int64` for future large-index adapters, but not current fixed CQP topology;
- IEEE `float32` as a recognised framework type, not a valid primary Paper 1 numerical buffer;
- IEEE `float64` for all CQP coefficients, bounds, affine offsets, primal iterates, and dual
  iterates.

Vector lanes must equal one. Endianness must be native. Sub-byte, complex, boolean, bfloat, and
opaque-handle dtypes are rejected.

## 5. Tensor shape and layout

Each workspace buffer is a rank-one logical tensor:

```text
shape   = [number of scalar elements]
stride  = [1]
offset  = byte_offset divisible by sizeof(dtype)
```

Noncontiguous views are rejected. SpacePDHCG does not copy, flatten, or materialise a strided
framework tensor behind the caller's back.

A zero-length view must have:

```text
data = NULL
elements = 0
byte_offset = 0
positive stride
```

A nonempty view must have a non-null pointer. The implementation may not dereference a pointer until
stream and lifetime preconditions have been met.

## 6. Fixed topology versus mutable numerical state

### 6.1 Immutable workspace-lifetime topology

The following are exactly read-only after workspace creation:

- `Q.col_offsets`, `Q.row_indices`;
- `A.col_offsets`, `A.row_indices`;
- `F.col_offsets`, `F.row_indices` when an affine-cone matrix exists;
- cone kinds, starts, dimensions, and power parameters;
- CQP dimensions;
- topology fingerprint.

The lengths must exactly equal those in the owning `FixedStructure`. The fingerprint must equal
`FixedStructure::fingerprint()`.

### 6.2 Mutable iteration values

The following are exactly read-write and may change in place at every SCvx iteration:

- `Q.values`;
- `A.values`;
- `F.values`;
- linear objective;
- scalar lower/upper bounds;
- affine-cone offset;
- variable lower/upper bounds.

The sparse index arrays may never be reallocated or changed through a numerical update.

### 6.3 Mutable solver state

The primal and dual iterate views are read-write and retained between solves. Their exact lengths are:

```text
primal = number of CQP variables
dual   = scalar rows + affine-cone rows
```

Additional upstream PDHCG state, such as averaged iterates, restart statistics, scaling vectors, or
Krylov/proximal state, is workspace-owned and is not exposed through version 1.

## 7. Access and aliasing

Topology views must be marked read-only. Numerical and iterate views must be marked read-write. A
DLPack tensor carrying `DLPACK_FLAG_BITMASK_READ_ONLY` cannot satisfy a writable slot. Writable
topology is rejected as a contract error rather than tolerated.

Unless a future API explicitly permits it, writable slots may not overlap in storage. In particular:

- primal and dual may not alias;
- objective and bounds may not alias;
- sparse matrix value arrays may not alias one another;
- topology arrays may not overlap writable arrays.

The CUDA implementation must add an optional debug validation pass that checks address intervals
when pointer provenance permits. Release mode may trust the caller after the exchange has been
validated once.

## 8. Stream semantics

`consumer_stream.native_handle` is the CUDA stream on which SpacePDHCG first consumes and then
mutates the supplied views. Handle zero means the legacy/default CUDA stream for the selected CUDA
execution device; nonzero values are interpreted as the native `cudaStream_t` bit pattern transported
through `uintptr_t`.

The producer must make its latest writes visible to the consumer stream before the workspace call.
For Python Array API/DLPack producers, the preferred sequence is:

1. obtain the target SpacePDHCG CUDA consumer stream;
2. call the producer's `__dlpack__(stream=<consumer stream>)` protocol;
3. consume the resulting `DLManagedTensorVersioned` without an additional host synchronization;
4. retain the managed tensor until SpacePDHCG releases the borrow.

If a producer cannot accept the consumer stream, it must record an event on its producer stream and
make the consumer stream wait on that event. `cudaDeviceSynchronize()` is forbidden in the normal
hot path.

On return from an asynchronous update or solve call, storage is not necessarily host-readable.
The future API must expose an event/future or an explicit stream-synchronization boundary. Compact
host diagnostics may be copied only after their completion event.

Managed memory is not implicitly coherent for performance purposes. The implementation may prefetch
managed pages to the consumer stream's GPU, but such prefetch time/bytes must be recorded. It may not
pretend managed memory has the same residency semantics as an ordinary device allocation.

## 9. DLPack ownership

`DLPackBorrow` owns exactly one `DLManagedTensorVersioned*` and calls the producer-provided deleter
exactly once unless ownership is explicitly released.

Rules:

- the DLPack capsule/managed tensor must be consumed only once;
- the producer's allocation must outlive every asynchronous consumer operation;
- SpacePDHCG may retain a borrow for the complete persistent workspace lifetime;
- destroying a workspace releases all retained borrows only after outstanding CUDA work is complete;
- a major-version mismatch calls the deleter without interpreting later fields;
- the adapter never guesses or performs stream synchronization;
- copied tensors signalled by DLPack remain governed by the producer's deleter;
- byte offsets are preserved, not folded into a modified ownership pointer.

The adapter is enabled only when `<dlpack/dlpack.h>` is available. The default native build has no
mandatory DLPack dependency.

The persistent CUDA C ABI additionally exposes
`persistent_pdhcg_dlpack_c_api.h`. Its managed-tensor wrappers accept both the
legacy `DLManagedTensor` layout and `DLManagedTensorVersioned` version 1
without introducing a Python or DLPack-header runtime dependency. The
`create_from_dlpack`, `update_from_dlpack_async`, and
`warm_start_from_dlpack_async` entry points consume ownership on entry,
validate the tensor metadata before forwarding accelerator views, and retain
the producer deleters until the corresponding asynchronous work is complete.
Python capsules must be renamed from `dltensor`/`dltensor_versioned` to their
`used_*` names exactly once; `spacepdhcg.backends.dlpack_capsule` performs this
capsule step before invoking the C ABI.

## 10. Validation order

A workspace creation call must validate in this order:

1. exchange ABI version;
2. CQP topology fingerprint;
3. ordinary CUDA execution device and stream descriptor;
4. one allowed persistent storage class;
5. CUDA storage id matching the stream GPU, or managed storage id zero;
6. rank/shape/stride/offset/dtype of every view;
7. exact element counts;
8. exact read-only versus read-write access;
9. optional address-overlap diagnostics;
10. actual CUDA pointer attributes on the selected device;
11. stream/event readiness.

A numerical update repeats replaceable-view checks but never accepts changed topology.

## 11. Planned persistent CUDA calls

The future exported implementation must preserve these semantics even if exact function names
change before ABI version 2:

```c
workspace_create(structure, exchange, options, &workspace)
workspace_update_async(workspace, numeric_views, consumer_stream, &event)
workspace_warm_start_async(workspace, iterate_views, consumer_stream, &event)
workspace_solve_async(workspace, tolerance, iteration_limit, consumer_stream, &event)
workspace_diagnostics_async(workspace, compact_host_result, consumer_stream, &event)
workspace_checkpoint_async(workspace, checkpoint_views, consumer_stream, &event)
workspace_destroy(workspace)
```

Creation owns or borrows topology and solver state. Update changes values only. Solve retains
iterates/scaling unless policy requests a reset or refresh. Destruction must not race outstanding
work.

## 12. Python wheel boundary

The wheel contains the stable C ABI shared library under `spacepdhcg/native`. Python loads it via
`ctypes` only for control-plane calls. The repeated numerical hot loop remains C++/CUDA.

Framework-specific DLPack capsule consumption will be a thin Python extension or C ABI shim that:

1. requests tensors on the workspace consumer stream;
2. converts each versioned managed tensor to an `AcceleratorBufferView`;
3. transfers the borrow into the persistent workspace;
4. returns an opaque workspace/event handle;
5. does not convert tensors to NumPy or copy through host memory.

## 13. Required GPU validation

Issue #2 cannot close until real CUDA tests demonstrate:

- pointer attributes match declared devices and memory classes;
- ordinary CUDA storage matches the stream GPU;
- managed storage is prefetched/consumed on the declared stream without invalid device assumptions;
- no topology allocation/copy after creation;
- coefficient updates occur on the supplied stream;
- no hidden device-wide synchronization;
- deleters are called exactly once and only after completion;
- use-after-free and cross-stream race tests fail safely;
- DLPack views from at least CuPy, PyTorch, and one JAX-compatible path produce identical CQP
  solutions;
- zero-copy and explicit-copy reference paths agree numerically;
- compute-sanitizer reports no invalid access or race in the declared test matrix.
