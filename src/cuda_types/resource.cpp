//
// Created by progamers on 8/5/25.
//
#include "cuda_types/resource.h"

namespace raw::cuda_types {
    raw::shared_ptr<raw::cuda_stream> resource::stream = raw::make_shared<raw::cuda_stream>();

    cudaGraphicsResource_t &resource::get_resource() {
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
        if (mapped) {
            CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &m_resource, 0));
        }
    }

    void resource::map() {
        if (!mapped) {
            CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &m_resource, 0));
        }
    }

    resource::~resource() {
        unmap();
        cudaGraphicsUnregisterResource(m_resource);
    }
} // namespace raw::cuda