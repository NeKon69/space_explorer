#ifndef SPACE_EXPLORER_N_BODY_CUDA_FWD_H
#define SPACE_EXPLORER_N_BODY_CUDA_FWD_H

namespace raw::n_body::cuda {
template<typename T>
class n_body_resource_manager;
template<typename T>
class n_body_simulator;

namespace physics {
template<typename T = double>
struct space_object_data;
}
} // namespace raw::n_body::cuda

#endif // SPACE_EXPLORER_N_BODY_CUDA_FWD_H
