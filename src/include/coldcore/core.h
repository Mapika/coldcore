/* coldcore/core.h — state-core C API (libcoldcore).
 *
 * Small, stable, boring: device probing and mode-tagged memory so that
 * plugins and host code share one placement policy. Everything returns
 * int rc (0 = ok) unless documented otherwise; no exceptions cross this
 * boundary.
 */
#ifndef COLDCORE_CORE_H
#define COLDCORE_CORE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define COLDCORE_CORE_API_VERSION 1

/* Memory placement modes (match the proven upstream engine's convention). */
#define CC_MEM_DEV     0  /* cudaMalloc: HBM                                */
#define CC_MEM_ATS     1  /* plain malloc: LPDDR, GPU access via ATS/C2C    */
#define CC_MEM_MANAGED 2  /* cudaMallocManaged, preferred location CPU      */

typedef struct cc_device_info {
  int64_t hbm_total;            /* bytes */
  int64_t hbm_free;             /* bytes at probe time */
  int     pageable_access;      /* cudaDevAttrPageableMemoryAccess          */
  int     host_native_atomics;  /* cudaDevAttrHostNativeAtomicSupported     */
  int     cc_major, cc_minor;   /* compute capability */
  char    name[256];
} cc_device_info;

int         cc_core_api_version(void);
const char *cc_version_string(void);

/* Probe device 0. rc != 0 => no usable CUDA device (info zeroed). */
int cc_device_probe(cc_device_info *info);

/* Allocate `bytes` in the given mode. MEM_ATS falls back to MEM_MANAGED
 * when the device is not ATS-coherent. Returns NULL on failure.
 * cc_free_mem must be called with the same mode that allocated. */
void *cc_alloc(int64_t bytes, int mode);
int   cc_free_mem(void *p, int mode);

/* Automatic placement policy (the GH200 policy that carried the records):
 * sequential-streamed arrays go to LPDDR when the total exceeds the HBM
 * budget; random-access arrays stay in HBM while they fit.
 *   bytes_seq  — total bytes of transform-streamed arrays (layers)
 *   bytes_rand — total bytes of random-access arrays (cnt)
 *   hbm_budget — bytes of HBM the caller may use (<=0: probe, use 5/6)
 * Outputs one CC_MEM_* mode per class. rc 0 = ok. */
int cc_place(int64_t bytes_seq, int64_t bytes_rand, int64_t hbm_budget,
             int *mode_seq, int *mode_rand);

#ifdef __cplusplus
}
#endif
#endif /* COLDCORE_CORE_H */
