//
// Created by progamers on 8/5/25.
//
#include "cuda_types/resource.h"

#include <iostream>

namespace raw::cuda_types {
void resource::unmap_noexcept() noexcept {
	if (mapped && m_resource) {
		try {
			CUDA_SAFE_CALL(
				cudaGraphicsUnmapResources(1, &m_resource, stream ? stream->stream() : nullptr));
			mapped = false;
		} catch (const cuda_exception& e) {
			std::cerr << std::format("[CRITICAL] Failed to unmap graphics resource. \n{}",
									 e.what());
		}
	}
}

void resource::cleanup() noexcept {
	unmap_noexcept();
	if (m_resource) {
		try {
			CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(m_resource));
		} catch (const cuda_exception& e) {
			std::cerr << std::format("[CRITICAL] Failed to unregister resource. \n{}", e.what());
		}
	}
}

resource& resource::operator=(resource&& rhs) noexcept {
	if (this == &rhs) {
		return *this;
	}
	cleanup();
	m_resource = rhs.m_resource;
	mapped	   = rhs.mapped;
	stream	   = std::move(rhs.stream);

	rhs.m_resource = nullptr;
	rhs.mapped	   = false;
	return *this;
}
resource::resource(resource&& rhs) noexcept
	: m_resource(rhs.m_resource), mapped(rhs.mapped), stream(std::move(rhs.stream)) {
	rhs.m_resource = nullptr;
	rhs.mapped	   = false;
}

cudaGraphicsResource_t& resource::get_resource() {
	return m_resource;
}

void resource::unmap() {
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

void resource::set_stream(std::shared_ptr<cuda_stream> stream_) {
	stream = std::move(stream_);
}

resource::~resource() {
	cleanup();
}
} // namespace raw::cuda_types
