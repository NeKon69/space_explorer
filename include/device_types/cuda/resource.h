//
// Created by progamers on 8/5/25.
//

#ifndef SPACE_EXPLORER_CUDA_RESOURCE_H
#define SPACE_EXPLORER_CUDA_RESOURCE_H
#include <cuda_egl_interop.h>
#include <cuda_gl_interop.h>

#include <memory>

#include "device_types/cuda/error.h"
#include "device_types/cuda/fwd.h"
#include "device_types/cuda/stream.h"

namespace raw::device_types::cuda {
/**
 * @class resource
 * @brief Base class for CUDA data from opengl, takes in the constructor function to register the
 * resource, unmaps the stored resource in the destructor and unregisters it
 */
class resource {
private:
	cudaGraphicsResource_t		 m_resource = nullptr;
	bool						 mapped		= false;
	std::shared_ptr<cuda_stream> stream;

private:
	void unmap_noexcept() noexcept;
	void cleanup() noexcept;

public:
	class mapped_resource {
	private:
		std::shared_ptr<cuda_stream> stream;
		cudaGraphicsResource_t		 resource;

	public:
		mapped_resource(std::shared_ptr<cuda_stream> stream, cudaGraphicsResource_t resource);
		~mapped_resource();
		cudaGraphicsResource_t &operator*();
	};
	resource() = default;

	template<typename F, typename... Args>
		requires std::invocable<F, cudaGraphicsResource_t *, Args...>
	explicit resource(const F &&func, std::shared_ptr<cuda_stream> stream, Args &&...args)
		: stream(stream) {
		create(func, std::forward<Args &&>(args)...);
	}

	template<typename F, typename... Args>
	void create(const F &&func, Args &&...args) {
		cleanup();
		CUDA_SAFE_CALL(func(&m_resource, std::forward<Args &&>(args)...));
	}

	mapped_resource get_resource();

	void			unmap();

	void map();

	void set_stream(std::shared_ptr<cuda_stream> stream_);

	virtual ~resource();

	resource &operator=(const resource &) = delete;
	resource(const resource &)			  = delete;

	resource &operator=(resource &&rhs) noexcept;
	resource(resource &&rhs) noexcept;

protected:
};

} // namespace raw::device_types::cuda

#endif // SPACE_EXPLORER_CUDA_RESOURCE_H