//
// Created by progamers on 8/5/25.
//
#include "cuda_types/resource.h"

#include <cuda/std/__ranges/data.h>

namespace raw::cuda_types {
void resource::cleanup() {
	unmap();
	if (m_resource) {
		CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(m_resource));
	}
}

resource& resource::operator=(resource&& rhs) {
	stream = std::move(rhs.stream);

	cleanup();
	m_resource = rhs.m_resource;
	mapped	   = rhs.mapped;

	rhs.m_resource = nullptr;
	rhs.mapped	   = false;
	return *this;
}
resource::resource(resource&& rhs)
	: m_resource(rhs.m_resource), mapped(rhs.mapped), stream(std::move(rhs.stream)) {
	rhs.m_resource = nullptr;
	rhs.mapped	   = false;
	rhs.stream	   = nullptr;
}

cudaGraphicsResource_t& resource::get_resource() {
	return m_resource;
}

void resource::unmap() {
	// After a bit of thinking with my brain i understood that this can't be streamed
	// Cause, well since streams are things that supposed to make calls asynchronously and this
	// map/unmap function would have different stream than the one that would for example, use that
	// resource, and we would end up in the position when resource is still being in the queue of
	// stream while we try to use it What i could do however is add function called `use` that would
	// just sync the stream, however still not sure if it's the best i've got, since then every time
	// i want to use some resource i would need to call use(), IDK yet OR what i should 've done to
	// make this thing easier, is for each part (sphere generation/n_body....) give each own
	// stream and that'll be it, sounds much cooler and expandable
	// HOWEVER that shit would require some class that manages those streams
	// The point is - fuck this shit
	if (mapped && m_resource) {
		CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &m_resource, stream->stream()));
		mapped = false;
	}
}

void resource::map() {
	if (!mapped && m_resource) {
		CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &m_resource, stream->stream()));
		mapped = true;
	}
}

resource::~resource() {
	try {
		cleanup();
	} catch (const std::runtime_error& e) {
		std::cerr << "[CRITICAL] An error occured in resource destructor: " << e.what()
				  << std::endl;
	}
}
} // namespace raw::cuda_types
