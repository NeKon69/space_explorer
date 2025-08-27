//
// Created by progamers on 7/21/25.
//
#include "n_body/physics/launch_leapfrog.h"
#include "n_body/physics/leapfrog_kernels.h"
#include "n_body/physics/space_object.h"

namespace raw::n_body::physics {
    template void launch_leapfrog<double>(space_object<double> *, glm::mat4 *objects_model, double,
                                          uint16_t, double, cudaStream_t stream);

    template void launch_leapfrog<float>(space_object<float> *, glm::mat4 *objects_model, float,
                                         unsigned short, double, cudaStream_t stream);

    template<typename T>
    void launch_leapfrog(space_object<T> *objects, glm::mat4 *objects_model, T time, uint16_t count,
                         double g, cudaStream_t stream) {
        auto threads_per_block = 256;
        auto blocks = (count + threads_per_block - 1) / 256;
        if (count < 512) {
            threads_per_block = count % 32 == 0 ? count : (count / 32 + 1) * 32;
        }
        compute_leapfrog<T><<<blocks, threads_per_block, 0, stream>>>(objects, objects_model, count,
                                                                      time, static_cast<T>(g));
    }
} // namespace raw::n_body::physics