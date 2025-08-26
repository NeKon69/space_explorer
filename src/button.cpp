//
// Created by progamers on 6/30/25.
//
#include "z_unused/button.h"

#include <iostream>
#include <utility>

namespace raw {
    button::button(raw::func_type type, std::function<void()> function) {
        switch (type) {
                using enum raw::func_type;
            case PRESSED:
                pressed_callback = std::move(function);
                break;
            case RELEASED:
                released_callback = std::move(function);
                break;
            case HELD:
                held_callback = std::move(function);
                break;
        }
    }


    button::button(std::function<void()> pressed, std::function<void()> held)
        : pressed_callback(std::move(pressed)), held_callback(std::move(held)) {
    }

    button::button(std::function<void()> pressed, std::function<void()> released,
                   std::function<void()> held)
        : pressed_callback(std::move(pressed)),
          released_callback(std::move(released)),
          held_callback(std::move(held)) {
    }

    button::button(std::function<void()> pressed) : pressed_callback(std::move(pressed)) {
    }

    void button::press() {
        is_held = !first_press;
        is_pressed = first_press;
        first_press = false;
    }

    void button::release() {
        is_released = true;
        is_pressed = false;
        is_held = false;
        first_press = true;
    }

    void button::update() {
        if ((is_held || is_pressed) && held_callback) {
            held_callback();
        }

        if (is_pressed && pressed_callback) {
            pressed_callback();
            is_pressed = false;
        }

        if (is_released && released_callback) {
            released_callback();
            is_released = false;
        }
    }
} // namespace raw