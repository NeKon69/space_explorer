//
// Created by progamers on 6/30/25.
//
#include "button.h"

#include <utility>

namespace raw {
button::button(std::function<void()> pressed, std::function<void()> released)
	: pressed_callback(std::move(pressed)), released_callback(std::move(released)) {}

button::button(std::function<void()> pressed) : pressed_callback(std::move(pressed)) {}

void button::press() {
	is_pressed = true;
    is_released = false;
}

void button::release() {
	is_released = true;
    is_pressed = false;
}

void button::update() {
    if (is_pressed) {
        if (pressed_callback) {
            pressed_callback();
        }
    }

    if (is_released) {
        if (released_callback) {
            released_callback();
        }
        is_released = false;
    }
}

} // namespace raw