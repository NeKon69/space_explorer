//
// Created by progamers on 6/30/25.
//

#ifndef SPACE_EXPLORER_BUTTON_H
#define SPACE_EXPLORER_BUTTON_H

#include <functional>

namespace raw {
class button {
private:
	std::function<void()> pressed_callback	= nullptr;
	std::function<void()> released_callback = nullptr;
	bool				  is_pressed		= false;
	bool				  is_released		= false;

public:
	button() = default;
	button(std::function<void()> pressed, std::function<void()> released);
	explicit button(std::function<void()> pressed);

	// setters
	void set_pressed_callback(const std::function<void()>& pressed) {
		pressed_callback = pressed;
	}

	void set_released_callback(const std::function<void()>& released) {
		released_callback = released;
	}

	// getter
	[[nodiscard]] inline bool pressed() const {
		return is_pressed;
	}

	void press();
	void release();
    void update();
};
} // namespace raw

#endif // SPACE_EXPLORER_BUTTON_H
