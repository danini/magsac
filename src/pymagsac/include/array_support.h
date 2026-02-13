#ifndef ARRAY_SUPPORT_HEADER_H
#define ARRAY_SUPPORT_HEADER_H

#include <array>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h> // for madvise on Linux
#endif

namespace nb = nanobind;
// using namespace nb::literals;

template <typename T, size_t N>
using NDArray = nb::ndarray<nb::numpy, T, nb::ndim<N>, nb::c_contig>;

template <typename T>
T *AllocateArray(size_t total, bool zero_initialize = false) {
  T *data = zero_initialize ? new T[total]() : new T[total];

#ifdef __linux__
  const size_t hugepage_threshold = 1u << 22u; // 4MB threshold
  const size_t page_size = 4096u;

  if (total * sizeof(T) >= hugepage_threshold) {
    uintptr_t data_addr = reinterpret_cast<uintptr_t>(data);
    size_t offset = page_size - (data_addr % page_size);
    size_t length = total * sizeof(T) - offset;

    madvise(reinterpret_cast<void *>(data_addr + offset), length,
            MADV_HUGEPAGE);
  }
#endif

  return data;
}

// wrap an existing array as a numpy ndarray
template <typename T, size_t N>
NDArray<T, N> WrapNDarray(T *data, const std::array<int, N> shape,
                          bool zero_initialize = false) {
  size_t shape_[N];
  for (size_t i = 0; i < N; ++i) {
    shape_[i] = shape[i];
  }

  nb::capsule owner(data, [](void *p) noexcept { delete[] (T *)p; });

  return NDArray<T, N>(data, N, shape_, owner);
}

template <typename T, size_t N>
NDArray<T, N> MakeNDArray(const std::array<int, N> shape,
                          bool zero_initialize = false) {

  // Calculate the total number of elements in the ndarray
  size_t total = 1;
  for (size_t i = 0; i < N; i++) {
    total *= shape[i];
  }

  size_t shape_[N];
  for (size_t i = 0; i < N; i++) {
    shape_[i] = shape[i];
  }

  T *data = AllocateArray<T>(total, zero_initialize);
  nb::capsule owner(data, [](void *p) noexcept { delete[] (T *)p; });

  return NDArray<T, N>(data, N, shape_, owner);
}

#endif // ARRAY_SUPPORT_HEADER_H
