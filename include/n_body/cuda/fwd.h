#ifndef SPACE_EXPLORER_N_BODY_CUDA_FWD_H
#define SPACE_EXPLORER_N_BODY_CUDA_FWD_H

namespace raw::n_body::cuda {
template<typename T>
class interaction_system;

template<typename T = double>
struct space_object_data;
namespace physics {
template<typename T>
class space_object;
}
} // namespace raw::n_body::cuda

#endif // SPACE_EXPLORER_N_BODY_CUDA_FWD_H
