#!/usr/bin/env python3
"""Add an opt-in cuDSS determinism diagnostic to the prepared GPU solver."""

# ruff: noqa: E501 -- pinned CUDA replacement text
import argparse
import hashlib
import json
from pathlib import Path

from prepare_qoco_device_control import once


def prepare(root):
    path = root / "algebra/cuda/cudss_backend.cu"
    original = path.read_text()
    configured = once(
        original,
        "  int value = 0;\n  CUDSS_CHECK(g_cuda_funcs.cudssConfigSet(linsys_data->config,",
        '  const char* superpanels_option=getenv("SPACEPDHCG_TEST_QOCO_CUDSS_SUPERPANELS");\n'
        "  int value = superpanels_option && superpanels_option[0]=='1' ? 1 : 0;\n"
        "  CUDSS_CHECK(g_cuda_funcs.cudssConfigSet(linsys_data->config,",
    )
    marker = """  CUDSS_CHECK(g_cuda_funcs.cudssConfigSet(linsys_data->config,
                                          CUDSS_CONFIG_USE_SUPERPANELS,
                                          (void*)&value, sizeof(int)));"""
    diagnostic = r"""
  // Keep the default numerical path unchanged. This option diagnoses vendor
  // factor/solve variation; QOCO's own device refinement remains enabled.
  const char* deterministic_option=getenv("SPACEPDHCG_TEST_QOCO_CUDSS_DETERMINISTIC");
  const bool deterministic=deterministic_option && deterministic_option[0]=='1';
  const bool trace_deterministic=getenv("SPACEPDHCG_TEST_QOCO_CUDSS_DETERMINISTIC_TRACE")!=nullptr;
  if (deterministic || trace_deterministic) {
#if CUDSS_VERSION >= 800
    const auto get_config=reinterpret_cast<decltype(&cudssConfigGet)>(
        dlsym(g_cudss_handle,"cudssConfigGet"));
    if (!get_config) { fprintf(stderr,"cuDSS configuration query unavailable\n"); exit(1); }
    int vendor_ir=-1,active=-1,superpanels=-1; size_t written=0;
    CUDSS_CHECK(get_config(linsys_data->config,CUDSS_CONFIG_IR_N_STEPS,
        &vendor_ir,sizeof(vendor_ir),&written));
    if (written!=sizeof(vendor_ir) || (deterministic && vendor_ir!=0)) {
        fprintf(stderr,"Deterministic mode requires disabled cuDSS iterative refinement\n"); exit(1);
    }
    if (deterministic) {
        const int enabled=1;
        CUDSS_CHECK(g_cuda_funcs.cudssConfigSet(linsys_data->config,
            CUDSS_CONFIG_DETERMINISTIC_MODE,&enabled,sizeof(enabled)));
    }
    CUDSS_CHECK(get_config(linsys_data->config,CUDSS_CONFIG_DETERMINISTIC_MODE,
        &active,sizeof(active),&written));
    if (written!=sizeof(active) || (deterministic && active!=1)) {
        fprintf(stderr,"cuDSS deterministic mode was not enabled\n"); exit(1);
    }
    CUDSS_CHECK(get_config(linsys_data->config,CUDSS_CONFIG_USE_SUPERPANELS,
        &superpanels,sizeof(superpanels),&written));
    if (written!=sizeof(superpanels)) { fprintf(stderr,"cuDSS superpanel query failed\n"); exit(1); }
    if (trace_deterministic) fprintf(stderr,"CUDSS_DETERMINISTIC requested=%d active=%d vendor_ir=%d superpanels=%d\n",
        int(deterministic),active,vendor_ir,superpanels);
#else
    if (deterministic) { fprintf(stderr,"cuDSS >=0.8 required for deterministic mode\n"); exit(1); }
#endif
  }
"""
    updated = once(configured, marker, marker + diagnostic)
    path.write_text(updated)
    return {
        "path": str(path),
        "before_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "after_sha256": hashlib.sha256(updated.encode()).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.destination), indent=2))
