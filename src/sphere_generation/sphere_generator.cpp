//
// Created by progamers on 8/28/25.
//
#include "sphere_generation/sphere_generator.h"

#include "sphere_generation/icosahedron_data_manager.h"
#include "sphere_generation/kernel_launcher.h"

namespace raw::sphere_generation {
void sphere_generator::generate(UI steps, cuda_types::cuda_stream& stream,
								icosahedron_data_manager& source) {
	sync();
	auto		 data_for_thread = source.get_data();
	cudaStream_t local_stream	 = stream.stream();
	worker_thread				 = std::jthread(
		   [data = std::move(data_for_thread), stream = local_stream, steps, &source] mutable {
			   auto context = source.create_context();
			   std::apply(sphere_generation::launch_tessellation,
									  std::tuple_cat(std::move(data), std::make_tuple(std::ref(stream), steps)));
		   });
}
void sphere_generator::sync() {
	if (worker_thread.joinable())
		worker_thread.join();
}

} // namespace raw::sphere_generation