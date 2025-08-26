//
// Created by progamers on 7/20/25.
//

#ifndef SPACE_EXPLORER_STREAM_H
#define SPACE_EXPLORER_STREAM_H
#include <cuda_runtime.h>

#include "cuda_types/fwd.h"
namespace raw::cuda_types {
class cuda_stream {
private:
	cudaStream_t _stream = nullptr;
	// Yea yea, it's used to not delete same stream twice. I could've made "cuda_stream" object only
	// moveable, but whatever
	bool created;

public:
	cuda_stream();
	cuda_stream(const cuda_stream& rhs)			   = delete;
	cuda_stream& operator=(const cuda_stream& rhs) = delete;
	cuda_stream(cuda_stream&& rhs) noexcept;
	cuda_stream& operator=(cuda_stream&& rhs) noexcept;
	void		 sync();
	void		 destroy();
	void		 create();
	cudaStream_t stream();
	~cuda_stream();
};
} // namespace raw::cuda_types

#endif // SPACE_EXPLORER_STREAM_H
