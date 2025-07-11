//
// Created by progamers on 7/4/25.
//

#ifndef SPACE_EXPLORER_EVENT_HANDLER_H
#define SPACE_EXPLORER_EVENT_HANDLER_H

#include <SDL3/SDL.h>
#include <raw_memory.h>

#include <functional>
#include <glm/gtc/type_ptr.hpp>
#include <unordered_map>

#include "button.h"
#include "clock.h"
#include "shader.h"

namespace raw {
template<typename T>
void update_uniform(raw::shared_ptr<raw::shader> &shader, std::string name, T value) {
	shader->use();
	// since I am a lazy man, I will just spam "if constexpr" statements here
	if constexpr (std::is_same_v<T, glm::vec3>) {
		shader->set_vec3(name, value);
	} else if constexpr (std::is_same_v<T, glm::vec4>) {
		shader->set_vec4(name, value);
	} else if constexpr (std::is_same_v<T, glm::vec2>) {
		shader->set_vec2(name, value);
	} else if constexpr (std::is_same_v<T, int>) {
		shader->set_int(name, value);
	} else if constexpr (std::is_same_v<T, float>) {
		shader->set_float(name, value);
	} else if constexpr (std::is_same_v<T, glm::mat4>) {
		shader->set_mat4(name, glm::value_ptr(value));
	} else if constexpr (std::is_same_v<T, bool>) {
		shader->set_bool(name, value);
	}
}

class scene;

template<typename T, typename... Shaders>
void update_uniform_for_shaders(const std::string &name, const T &value, Shaders &...shaders) {
	(update_uniform(shaders, name, value), ...);
}

class event_handler {
public:
	// for some reason unordered map doesn't want to store events, well, let's do it the ugly way
	std::unordered_map<Uint32, std::function<void(SDL_Event)>> events;
	std::unordered_map<SDL_Scancode, raw::button>			   buttons;
	raw::clock												   clock_callback;

	friend class scene;

public:
	event_handler();
	void setup(raw::scene *scene);
	void handle(const SDL_Event &event);

private:
	void _setup_keys(raw::scene *scene);
	void _update();
};
} // namespace raw

#endif // SPACE_EXPLORER_EVENT_HANDLER_H
