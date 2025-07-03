//
// Created by progamers on 6/30/25.
//

#ifndef SPACE_EXPLORER_BUTTON_H
#define SPACE_EXPLORER_BUTTON_H

#include <functional>

namespace raw {
    enum class func_type {
        HELD,
        PRESSED,
        RELEASED
    };

class button {
private:
	std::function<void()> pressed_callback	= nullptr;
	std::function<void()> released_callback = nullptr;
	std::function<void()> held_callback		= nullptr;
	bool				  is_pressed		= false;
	bool				  is_released		= false;
    bool				  is_held			= false;

    bool first_press = true;

public:
    button() = default;
    button(func_type type, std::function<void()> function);
    button(std::function<void()> pressed, std::function<void()> held);
	button(std::function<void()> pressed, std::function<void()> released, std::function<void()> held);
	explicit button(std::function<void()> pressed);

	// setters
	void set_pressed_callback(const std::function<void()>& pressed) {
		pressed_callback = pressed;
	}

	void set_released_callback(const std::function<void()>& released) {
		released_callback = released;
	}

    void set_held_callback(const std::function<void()>& held) {
        held_callback = held;
    }

	// getter
	[[nodiscard]] inline bool pressed() const {
		return is_held || is_pressed;
	}

	void press();
	void release();
	void update();
};
} // namespace raw

#endif // SPACE_EXPLORER_BUTTON_H
