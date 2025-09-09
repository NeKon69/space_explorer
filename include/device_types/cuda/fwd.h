#pragma once

namespace raw::device_types::cuda {
enum class side { host, device };
template<typename T, side Side = side::device>
class buffer;

class cuda_stream;

class resource;

template<typename T>
class surface;

template<typename T>
class resource_description;

enum class cudaMemcpyOrder { cudaMemcpy1to2, cudaMemcpy2to1 };
} // namespace raw::cuda

