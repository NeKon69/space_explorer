#ifndef SPACE_EXPLORER_N_BODY_CUDA_FWD_H
#define SPACE_EXPLORER_N_BODY_CUDA_FWD_H

namespace raw::n_body::cuda {
template<typename T>
class n_body_resource_manager;
template<typename T>
class n_body_simulator;
enum class pending_action_type { ADD, REMOVE };
template<typename T>
struct pending_action;

namespace physics {
template<typename T = double>
struct space_object_data;

}
template<typename T>
struct is_in_set {
	uint32_t*				 ids_begin;
	uint32_t*				 ids_end;
	__host__ __device__ bool operator()(const physics::space_object_data<T>& object) const {
		for (uint32_t* it = ids_begin; it != ids_end; ++it) {
			if (*it == object.id) {
				return true;
			}
		}
		return false;
	}
};
} // namespace raw::n_body::cuda

#endif // SPACE_EXPLORER_N_BODY_CUDA_FWD_H
