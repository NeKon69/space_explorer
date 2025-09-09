//
// Created by progamers on 9/9/25.
//
#include <thrust/remove.h>

#include "n_body/cuda/n_body_resource_manager.h"
namespace raw::n_body::cuda {
template<typename T>
void n_body_resource_manager<T>::update_data() {
	if (!pending_actions.empty()) {
		std::vector<uint32_t> ids_to_remove;
		for (const auto& action : pending_actions) {
			if (action.type == pending_action_type::REMOVE) {
				ids_to_remove.push_back(action.id_to_remove);
			}
		}

		if (!ids_to_remove.empty()) {
			device_types::cuda::buffer<uint32_t> d_ids_to_remove(ids_to_remove.size(),
																 local_stream);
			d_ids_to_remove.memcpy(ids_to_remove.data(), ids_to_remove.size(), 0,
								   cudaMemcpyHostToDevice);

			auto new_end_iter = thrust::remove_if(
				thrust::cuda::par.on(local_stream->stream()), physics_data.get(),
				physics_data.get() + objects.size(),
				is_in_set<T> {d_ids_to_remove.get(),
							  d_ids_to_remove.get() + d_ids_to_remove.get_size()});

			size_t new_size = new_end_iter - physics_data.get();
			objects.erase(std::remove_if(objects.begin(), objects.end(),
										 [&](const auto& obj) {
											 return std::find(ids_to_remove.begin(),
															  ids_to_remove.end(),
															  obj.id) != ids_to_remove.end();
										 }),
						  objects.end());

			assert(new_size == objects.size());
		}

		std::vector<space_object_data<T>> objects_to_add;
		for (const auto& action : pending_actions) {
			if (action.type == pending_action_type::ADD) {
				objects_to_add.push_back(action.object_to_add);
			}
		}

		if (!objects_to_add.empty()) {
			physics_data.memcpy(objects_to_add.data(), objects_to_add.size(),
								objects.size() * sizeof(space_object_data<T>),
								cudaMemcpyHostToDevice);
			objects.insert(objects.end(), objects_to_add.begin(), objects_to_add.end());
		}

		pending_actions.clear();
	}
}

template void n_body_resource_manager<float>::update_data();
} // namespace raw::n_body::cuda
