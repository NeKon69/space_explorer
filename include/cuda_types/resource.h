//
// Created by progamers on 8/5/25.
//

#ifndef SPACE_EXPLORER_RESOURCE_H
#define SPACE_EXPLORER_RESOURCE_H
#include <cuda_gl_interop.h>
#include <helper/helper_macros.h>
#include <raw_memory.h>

#include "cuda_types/fwd.h"
#include "cuda_types/stream.h"

namespace raw::cuda {
	/**
	 * @class resource
	 * @brief Base class for CUDA data from opengl, takes in the constructor function to register the
	 * resource, unmaps the stored resource in the destructor, however, doesn't unregister the resource
	 * itself, it's done in the destructors of inherited classes
	 */
	class resource {
	private:
		cudaGraphicsResource_t m_resource = nullptr;
		bool mapped = false;
		raw::shared_ptr<raw::cuda_stream> stream;

	protected:
		// I heard somewhere that this down here is better than directly accessing the protected member
	cudaGraphicsResource_t& get_resource();

public:
	resource() = default;
	template<typename F, typename... Args>
	// TODO: i dont remember how "is_callable" is called and where it is, need to add it here
		requires std::is_function_v<F>
	explicit resource(const F &&func, Args &&... args) {
		create(func, std::forward<Args&&>(args)...);
	}
	template<typename F, typename... Args>
	void create(const F&& func, Args&&... args) {
		CUDA_SAFE_CALL(func(&m_resource, std::forward<Args&&>(args)...));
		CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &m_resource, stream->stream()));
		mapped = true;
	}
	void unmap();
	void map();
	~resource();
	resource& operator=(const resource&) = delete;
	resource(const resource&)			 = delete;
	resource& operator=(resource&&)		 = default;
	resource(resource&&)				 = default;
	};
} // namespace raw::cuda

#endif // SPACE_EXPLORER_RESOURCE_H
