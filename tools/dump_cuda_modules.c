#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

typedef int CUresult;
typedef void* CUmodule;
typedef void* CUfunction;
typedef int CUjit_option;
typedef void* CUlibrary;
typedef void* CUstream;
typedef void* cudaStream_t;

static int dump_counter = 0;
static CUresult (*real_cuLaunchKernel_pfn)(CUfunction, unsigned int, unsigned int,
                                           unsigned int, unsigned int,
                                           unsigned int, unsigned int,
                                           unsigned int, CUstream, void**,
                                           void**) = NULL;
static CUresult (*real_cuModuleLoadData_pfn)(CUmodule*, const void*) = NULL;
static CUresult (*real_cuModuleLoadDataEx_pfn)(CUmodule*, const void*, unsigned int,
                                               CUjit_option*, void**) = NULL;

static size_t guess_blob_size(const void* image) {
  const unsigned char* p = (const unsigned char*)image;
  if (!p) return 0;
  if (p[0] == 0x7f && p[1] == 'E' && p[2] == 'L' && p[3] == 'F') {
    const uint64_t shoff = *(const uint64_t*)(p + 0x28);
    const uint16_t shentsize = *(const uint16_t*)(p + 0x3a);
    const uint16_t shnum = *(const uint16_t*)(p + 0x3c);
    size_t size = shoff + (size_t)shentsize * shnum;
    if (size > 0 && size < (512ul << 20)) return size;
  }
  return 0;
}

static void dump_image(const char* api, const void* image, size_t explicit_size) {
  const char* dir = getenv("CUDA_MODULE_DUMP_DIR");
  if (!dir || !image) return;
  mkdir(dir, 0777);

  size_t size = explicit_size ? explicit_size : guess_blob_size(image);
  if (size == 0 || size > (512ul << 20)) {
    const unsigned char* p = (const unsigned char*)image;
    fprintf(stderr, "[dump_cuda_modules] skip %s image=%p magic=%02x %02x %02x %02x size=%zu\n",
            api, image, p[0], p[1], p[2], p[3], size);
    return;
  }

  char path[4096];
  int id = __sync_fetch_and_add(&dump_counter, 1);
  const unsigned char* p = (const unsigned char*)image;
  const char* ext = (p[0] == 0x7f && p[1] == 'E' && p[2] == 'L' && p[3] == 'F')
                        ? "cubin"
                        : "fatbin";
  snprintf(path, sizeof(path), "%s/module_%04d_%s.%s", dir, id, api, ext);
  FILE* f = fopen(path, "wb");
  if (!f) return;
  fwrite(image, 1, size, f);
  fclose(f);
  fprintf(stderr, "[dump_cuda_modules] wrote %s size=%zu\n", path, size);
}

CUresult cuModuleLoadData(CUmodule* module, const void* image) {
  static CUresult (*real_fn)(CUmodule*, const void*) = NULL;
  if (!real_fn) real_fn = dlsym(RTLD_NEXT, "cuModuleLoadData");
  dump_image("cuModuleLoadData", image, 0);
  return real_fn(module, image);
}

static CUresult wrap_cuModuleLoadData(CUmodule* module, const void* image) {
  dump_image("cuModuleLoadData_pfn", image, 0);
  return real_cuModuleLoadData_pfn(module, image);
}

CUresult cuModuleLoadDataEx(CUmodule* module, const void* image,
                            unsigned int numOptions, CUjit_option* options,
                            void** optionValues) {
  static CUresult (*real_fn)(CUmodule*, const void*, unsigned int, CUjit_option*,
                             void**) = NULL;
  if (!real_fn) real_fn = dlsym(RTLD_NEXT, "cuModuleLoadDataEx");
  dump_image("cuModuleLoadDataEx", image, 0);
  return real_fn(module, image, numOptions, options, optionValues);
}

static CUresult wrap_cuModuleLoadDataEx(CUmodule* module, const void* image,
                                        unsigned int numOptions,
                                        CUjit_option* options,
                                        void** optionValues) {
  dump_image("cuModuleLoadDataEx_pfn", image, 0);
  return real_cuModuleLoadDataEx_pfn(module, image, numOptions, options,
                                     optionValues);
}

CUresult cuLibraryLoadData(CUlibrary* library, const void* code,
                           void** jitOptions, void** jitOptionsValues,
                           unsigned int numJitOptions, void** libraryOptions,
                           void** libraryOptionValues,
                           unsigned int numLibraryOptions) {
  static CUresult (*real_fn)(CUlibrary*, const void*, void**, void**, unsigned int,
                             void**, void**, unsigned int) = NULL;
  if (!real_fn) real_fn = dlsym(RTLD_NEXT, "cuLibraryLoadData");
  dump_image("cuLibraryLoadData", code, 0);
  return real_fn(library, code, jitOptions, jitOptionsValues, numJitOptions,
                 libraryOptions, libraryOptionValues, numLibraryOptions);
}

CUresult cuModuleLoad(CUmodule* module, const char* fname) {
  static CUresult (*real_fn)(CUmodule*, const char*) = NULL;
  if (!real_fn) real_fn = dlsym(RTLD_NEXT, "cuModuleLoad");
  fprintf(stderr, "[dump_cuda_modules] cuModuleLoad %s\n", fname ? fname : "(null)");
  return real_fn(module, fname);
}

CUresult cuLibraryLoadFromFile(CUlibrary* library, const char* fileName,
                               void** jitOptions, void** jitOptionsValues,
                               unsigned int numJitOptions,
                               void** libraryOptions,
                               void** libraryOptionValues,
                               unsigned int numLibraryOptions) {
  static CUresult (*real_fn)(CUlibrary*, const char*, void**, void**, unsigned int,
                             void**, void**, unsigned int) = NULL;
  if (!real_fn) real_fn = dlsym(RTLD_NEXT, "cuLibraryLoadFromFile");
  fprintf(stderr, "[dump_cuda_modules] cuLibraryLoadFromFile %s\n",
          fileName ? fileName : "(null)");
  return real_fn(library, fileName, jitOptions, jitOptionsValues, numJitOptions,
                 libraryOptions, libraryOptionValues, numLibraryOptions);
}

CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                        unsigned int gridDimZ, unsigned int blockDimX,
                        unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes, CUstream hStream,
                        void** kernelParams, void** extra) {
  static CUresult (*real_fn)(CUfunction, unsigned int, unsigned int, unsigned int,
                             unsigned int, unsigned int, unsigned int,
                             unsigned int, CUstream, void**, void**) = NULL;
  static CUresult (*name_fn)(const char**, CUfunction) = NULL;
  if (!real_fn) real_fn = dlsym(RTLD_NEXT, "cuLaunchKernel");
  if (!name_fn) name_fn = dlsym(RTLD_NEXT, "cuFuncGetName");
  const char* name = NULL;
  if (name_fn) name_fn(&name, f);
  fprintf(stderr,
          "[dump_cuda_modules] launch name=%s grid=%ux%ux%u block=%ux%ux%u smem=%u\n",
          name ? name : "(unknown)", gridDimX, gridDimY, gridDimZ, blockDimX,
          blockDimY, blockDimZ, sharedMemBytes);
  return real_fn(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                 blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
}

static CUresult wrap_cuLaunchKernel(CUfunction f, unsigned int gridDimX,
                                    unsigned int gridDimY, unsigned int gridDimZ,
                                    unsigned int blockDimX,
                                    unsigned int blockDimY,
                                    unsigned int blockDimZ,
                                    unsigned int sharedMemBytes,
                                    CUstream hStream, void** kernelParams,
                                    void** extra) {
  static CUresult (*name_fn)(const char**, CUfunction) = NULL;
  if (!name_fn) name_fn = dlsym(RTLD_NEXT, "cuFuncGetName");
  const char* name = NULL;
  if (name_fn) name_fn(&name, f);
  fprintf(stderr,
          "[dump_cuda_modules] launch_pfn name=%s grid=%ux%ux%u block=%ux%ux%u smem=%u\n",
          name ? name : "(unknown)", gridDimX, gridDimY, gridDimZ, blockDimX,
          blockDimY, blockDimZ, sharedMemBytes);
  return real_cuLaunchKernel_pfn(f, gridDimX, gridDimY, gridDimZ, blockDimX,
                                 blockDimY, blockDimZ, sharedMemBytes, hStream,
                                 kernelParams, extra);
}

typedef struct dim3_ {
  unsigned int x;
  unsigned int y;
  unsigned int z;
} dim3_;

int cudaLaunchKernel(const void* func, dim3_ gridDim, dim3_ blockDim,
                     void** args, size_t sharedMem, cudaStream_t stream) {
  static int (*real_fn)(const void*, dim3_, dim3_, void**, size_t,
                        cudaStream_t) = NULL;
  if (!real_fn) real_fn = dlsym(RTLD_NEXT, "cudaLaunchKernel");
  fprintf(stderr,
          "[dump_cuda_modules] cudaLaunchKernel func=%p grid=%ux%ux%u block=%ux%ux%u smem=%zu\n",
          func, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y,
          blockDim.z, sharedMem);
  return real_fn(func, gridDim, blockDim, args, sharedMem, stream);
}

CUresult cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion,
                          uint64_t flags) {
  static CUresult (*real_fn)(const char*, void**, int, uint64_t) = NULL;
  if (!real_fn) real_fn = dlsym(RTLD_NEXT, "cuGetProcAddress");
  CUresult ret = real_fn(symbol, pfn, cudaVersion, flags);
  if (ret == 0 && symbol && pfn && *pfn) {
    if (strcmp(symbol, "cuLaunchKernel") == 0) {
      real_cuLaunchKernel_pfn = *pfn;
      *pfn = (void*)wrap_cuLaunchKernel;
      fprintf(stderr, "[dump_cuda_modules] hooked cuGetProcAddress cuLaunchKernel\n");
    } else if (strcmp(symbol, "cuModuleLoadData") == 0) {
      real_cuModuleLoadData_pfn = *pfn;
      *pfn = (void*)wrap_cuModuleLoadData;
      fprintf(stderr, "[dump_cuda_modules] hooked cuGetProcAddress cuModuleLoadData\n");
    } else if (strcmp(symbol, "cuModuleLoadDataEx") == 0) {
      real_cuModuleLoadDataEx_pfn = *pfn;
      *pfn = (void*)wrap_cuModuleLoadDataEx;
      fprintf(stderr, "[dump_cuda_modules] hooked cuGetProcAddress cuModuleLoadDataEx\n");
    }
  }
  return ret;
}
