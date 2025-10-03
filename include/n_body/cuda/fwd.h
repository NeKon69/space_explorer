#pragma once

#include "n_body/fwd.h"
namespace raw::n_body::cuda {
template<typename T>
class n_body_resource_manager;
template<typename T>
class n_body_simulator;
enum class pending_action_type { ADD, REMOVE };
template<typename T>
struct pending_action;
template<typename T>
struct object_data_view;
} // namespace raw::n_body::cuda
