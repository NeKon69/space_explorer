//
// Created by progamers on 6/19/25.
//

#ifndef SPACE_EXPLORER_CLOCK_H
#define SPACE_EXPLORER_CLOCK_H
#include <cmath>
#include "helper/fwd.h"
#include "helper_macros.h"

namespace raw {
	struct time {
		long double val;
		time_rate	curr = time_rate::NANO;
	explicit constexpr time(long double value) : val(value) {};

	inline void handle_conversion(time_rate target) {
		// obtain how much to multiply in 1000^x where x is difference between types (nano, micro,
		// etc...)
		// I use this structure for cuda kernels, so can't use std::to_underlying
		val = val / std::pow(1000, static_cast<int>(target) - static_cast<int>(curr));
		curr = target;
	}

	inline void to_micro() {
		handle_conversion(time_rate::MICRO);
	}

	inline void to_nano() {
		handle_conversion(time_rate::NANO);
	}

	inline void to_milli() {
		handle_conversion(time_rate::MILLI);
	}

	inline void to_sec() {
		handle_conversion(time_rate::ORD);
	}

	[[nodiscard]] inline long double operator()() const {
		return val;
	}

	[[nodiscard]] inline long double value() const {
		return val;
	}

	friend inline time operator-(const time& lhs, const time& rhs) {
		auto cp = rhs;
		cp.handle_conversion(lhs.curr);
		return time(lhs.val - cp.val);
	}

	[[nodiscard]] inline std::partial_ordering operator<=>(const time& rhs) const {
		return val <=> rhs.val;
	}

	[[nodiscard]] inline bool operator!=(const time& rhs) const {
		return this->val != rhs.val;
	}
	friend std::ostream& operator<<(std::ostream& os, const time& par);
};

class clock {
private:
	time clock_start = time(((std_clock::now()).time_since_epoch()).count());
	time clock_stop;
	// Check how good clock resolution is
	static_assert(std::ratio_less_equal_v<std::chrono::high_resolution_clock::period, std::micro>,
				  "Clock resolution is too low. Expecting at least a microsecond precision");

public:
	// calls `start` by default
	clock();

	[[nodiscard]] time get_elapsed_time() const;

	[[nodiscard]] bool is_running() const;

	void start();

	void stop();

	time restart();

	// stops the clock, returns elapsed time
	time reset();
};
} // namespace raw

#endif // SPACE_EXPLORER_CLOCK_H
