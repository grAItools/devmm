/*
 * abi_oracle.c — compiled reference for the ctypes mirrors in
 * src/devmm/_dlpack/_abi.py (design §7.1).
 *
 * Prints the sizeof/offsetof of every DLPack struct as JSON so the test
 * suite can diff a real compiler's layout against ctypes, and so the
 * per-platform snapshots can be regenerated verbatim:
 *
 *   cc -std=c11 abi_oracle.c -o abi_oracle
 *   ./abi_oracle > snapshots/<platform>.json
 */
#include <stddef.h>
#include <stdio.h>

#include "dlpack.h"

int main(void) {
  printf("{\n");
  printf("  \"dlpack_version\": \"%d.%d\",\n", DLPACK_MAJOR_VERSION,
         DLPACK_MINOR_VERSION);
  printf("  \"structs\": {\n");
  printf("    \"DLPackVersion\": {\"size\": %zu, \"fields\": "
         "{\"major\": %zu, \"minor\": %zu}},\n",
         sizeof(DLPackVersion), offsetof(DLPackVersion, major),
         offsetof(DLPackVersion, minor));
  printf("    \"DLDevice\": {\"size\": %zu, \"fields\": "
         "{\"device_type\": %zu, \"device_id\": %zu}},\n",
         sizeof(DLDevice), offsetof(DLDevice, device_type),
         offsetof(DLDevice, device_id));
  printf("    \"DLDataType\": {\"size\": %zu, \"fields\": "
         "{\"code\": %zu, \"bits\": %zu, \"lanes\": %zu}},\n",
         sizeof(DLDataType), offsetof(DLDataType, code),
         offsetof(DLDataType, bits), offsetof(DLDataType, lanes));
  printf("    \"DLTensor\": {\"size\": %zu, \"fields\": "
         "{\"data\": %zu, \"device\": %zu, \"ndim\": %zu, \"dtype\": %zu, "
         "\"shape\": %zu, \"strides\": %zu, \"byte_offset\": %zu}},\n",
         sizeof(DLTensor), offsetof(DLTensor, data), offsetof(DLTensor, device),
         offsetof(DLTensor, ndim), offsetof(DLTensor, dtype),
         offsetof(DLTensor, shape), offsetof(DLTensor, strides),
         offsetof(DLTensor, byte_offset));
  printf("    \"DLManagedTensor\": {\"size\": %zu, \"fields\": "
         "{\"dl_tensor\": %zu, \"manager_ctx\": %zu, \"deleter\": %zu}},\n",
         sizeof(DLManagedTensor), offsetof(DLManagedTensor, dl_tensor),
         offsetof(DLManagedTensor, manager_ctx),
         offsetof(DLManagedTensor, deleter));
  printf("    \"DLManagedTensorVersioned\": {\"size\": %zu, \"fields\": "
         "{\"version\": %zu, \"manager_ctx\": %zu, \"deleter\": %zu, "
         "\"flags\": %zu, \"dl_tensor\": %zu}}\n",
         sizeof(struct DLManagedTensorVersioned),
         offsetof(struct DLManagedTensorVersioned, version),
         offsetof(struct DLManagedTensorVersioned, manager_ctx),
         offsetof(struct DLManagedTensorVersioned, deleter),
         offsetof(struct DLManagedTensorVersioned, flags),
         offsetof(struct DLManagedTensorVersioned, dl_tensor));
  printf("  }\n");
  printf("}\n");
  return 0;
}
