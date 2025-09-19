//
// Created by progamers on 9/18/25.
//

#pragma once

#include <cstdint>
#include <queue>
#include <unordered_map>
#include <vector>

#include "components/component.h"
#include "entity_management/entity_id.h"
#include "n_body/physics_component.h"
#include "rendering/visual_component.h"
#include "texture_generation/generation_recipe_component.h"
namespace raw::entity_management {
template<typename T>
class entity_manager {
private:
	std::vector<uint64_t>										generations;
	std::queue<uint64_t>										free_indices;
	std::unordered_map<entity_id, n_body::physics_component<T>> physics_components;
	std::unordered_map<entity_id, rendering::visual_component>	visual_components;
	std::unordered_map<entity_id, texture_generation::generation_recipe_component>
		generation_recipe_components;

public:
	entity_id create_entity();
	void	  destroy_entity(entity_id id);
	bool	  is_valid(entity_id id) const;

	template<typename ComponentType>
	void add_component(entity_id id, ComponentType component) noexcept
		requires(components::IsComponent<ComponentType>)
	{
		if constexpr (std::is_same_v<ComponentType, n_body::physics_component<T>>) {
			physics_components[id] = component;
		} else if constexpr (std::is_same_v<ComponentType, rendering::visual_component>) {
			visual_components[id] = component;
		} else if constexpr (std::is_same_v<ComponentType,
											texture_generation::generation_recipe_component>) {
			generation_recipe_components[id] = component;
		}
	}

	template<typename ComponentType>
	void remove_component(entity_id id) noexcept
		requires(components::IsComponent<ComponentType>)
	{
		if constexpr (std::is_same_v<ComponentType, n_body::physics_component<T>>) {
			physics_components.erase(id);
		} else if constexpr (std::is_same_v<ComponentType, rendering::visual_component>) {
			visual_components.erase(id);
		} else if constexpr (std::is_same_v<ComponentType,
											texture_generation::generation_recipe_component>) {
			generation_recipe_components.erase(id);
		}
	}

	template<typename ComponentType>
	ComponentType get_component(entity_id id) noexcept
		requires(components::IsComponent<ComponentType>)
	{
		if (!is_valid(id)) {
			throw std::invalid_argument("Invalid entity ID");
		}
		if constexpr (std::is_same_v<ComponentType, n_body::physics_component<T>>) {
			return physics_components[id];
		} else if constexpr (std::is_same_v<ComponentType, rendering::visual_component>) {
			return visual_components[id];
		} else if constexpr (std::is_same_v<ComponentType,
											texture_generation::generation_recipe_component>) {
			return generation_recipe_components[id];
		} else {
			throw std::invalid_argument("Invalid Component Type");
		}
	}

	template<typename ComponentType>
	[[nodiscard]] size_t get_component_count() const noexcept {
		if constexpr (std::is_same_v<ComponentType, n_body::physics_component<T>>) {
			return physics_components.size();
		} else if constexpr (std::is_same_v<ComponentType, rendering::visual_component>) {
			return visual_components.size();
		} else if constexpr (std::is_same_v<ComponentType,
											texture_generation::generation_recipe_component>) {
			return generation_recipe_components.size();
		} else {
			throw std::invalid_argument("Invalid Component Type");
		}
	}

	template<typename ComponentType>
	const std::unordered_map<entity_id, ComponentType>& get_components() const noexcept {
		if constexpr (std::is_same_v<ComponentType, n_body::physics_component<T>>) {
			return physics_components;
		} else if constexpr (std::is_same_v<ComponentType, rendering::visual_component>) {
			return visual_components;
		} else if constexpr (std::is_same_v<ComponentType,
											texture_generation::generation_recipe_component>) {
			return generation_recipe_components;
		} else {
			throw std::invalid_argument("Invalid Component Type");
		}
	}
};
} // namespace raw::entity_management