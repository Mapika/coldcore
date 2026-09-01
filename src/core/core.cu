// core.cu — libcoldcore state-core utilities.
//
// Implements coldcore/core.h: device probing, mode-tagged allocation
// (HBM / ATS-LPDDR / managed), and the automatic placement policy that
// carried the covering-code records on GH200 (stream big sequential
// arrays from LPDDR over C2C, keep random-access arrays in HBM).

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>

#include "coldcore/core.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "coldcore: %s failed: %s\n", #x, cudaGetErrorString(e_)); \
  return -1; } } while (0)

extern "C" {

int cc_core_api_version(void) { return COLDCORE_CORE_API_VERSION; }

const char *cc_version_string(void) { return "coldcore 0.1.0-dev"; }

int cc_device_probe(cc_device_info *info) {
  if (!info) return -2;
  memset(info, 0, sizeof(*info));
  int ndev = 0;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) return -1;
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  size_t freeb = 0, total = 0;
  CK(cudaMemGetInfo(&freeb, &total));
  info->hbm_total = (int64_t)total;
  info->hbm_free = (int64_t)freeb;
  int v = 0;
  CK(cudaDeviceGetAttribute(&v, cudaDevAttrPageableMemoryAccess, 0));
  info->pageable_access = v;
  CK(cudaDeviceGetAttribute(&v, cudaDevAttrHostNativeAtomicSupported, 0));
  info->host_native_atomics = v;
  info->cc_major = p.major;
  info->cc_minor = p.minor;
  snprintf(info->name, sizeof(info->name), "%s", p.name);
  return 0;
}

void *cc_alloc(int64_t bytes, int mode) {
  if (bytes <= 0) return nullptr;
  if (mode == CC_MEM_ATS) {
    int coherent = 0;
    if (cudaDeviceGetAttribute(&coherent, cudaDevAttrPageableMemoryAccess, 0)
            != cudaSuccess || !coherent)
      mode = CC_MEM_MANAGED;   // graceful degrade off GH200-class devices
  }
  void *p = nullptr;
  cudaError_t e = cudaSuccess;
  switch (mode) {
    case CC_MEM_DEV:
      e = cudaMalloc(&p, (size_t)bytes);
      break;
    case CC_MEM_ATS:
      p = malloc((size_t)bytes);
      break;
    case CC_MEM_MANAGED:
      e = cudaMallocManaged(&p, (size_t)bytes);
      if (e == cudaSuccess) {
        cudaMemAdvise(p, (size_t)bytes, cudaMemAdviseSetPreferredLocation,
                      cudaCpuDeviceId);
        cudaMemAdvise(p, (size_t)bytes, cudaMemAdviseSetAccessedBy, 0);
      }
      break;
    default:
      return nullptr;
  }
  if (e != cudaSuccess) {
    fprintf(stderr, "coldcore: cc_alloc(%lld, mode=%d): %s\n",
            (long long)bytes, mode, cudaGetErrorString(e));
    return nullptr;
  }
  return p;
}

int cc_free_mem(void *p, int mode) {
  if (!p) return 0;
  switch (mode) {
    case CC_MEM_DEV:
    case CC_MEM_MANAGED:
      CK(cudaFree(p));
      return 0;
    case CC_MEM_ATS:
      free(p);
      return 0;
    default:
      return -2;
  }
}

int cc_place(int64_t bytes_seq, int64_t bytes_rand, int64_t hbm_budget,
             int *mode_seq, int *mode_rand) {
  if (!mode_seq || !mode_rand) return -2;
  if (hbm_budget <= 0) {
    cc_device_info di;
    if (cc_device_probe(&di) != 0) return -1;
    hbm_budget = di.hbm_total * 5 / 6;   // leave headroom for tables/scratch
  }
  // Everything in HBM when it fits; otherwise stream the sequential
  // arrays from LPDDR and keep the random-access arrays in HBM while
  // they fit on their own.
  if (bytes_seq + bytes_rand <= hbm_budget) {
    *mode_seq = CC_MEM_DEV;
    *mode_rand = CC_MEM_DEV;
  } else {
    *mode_seq = CC_MEM_ATS;
    *mode_rand = (bytes_rand <= hbm_budget / 2) ? CC_MEM_DEV : CC_MEM_ATS;
  }
  return 0;
}

}  // extern "C"
