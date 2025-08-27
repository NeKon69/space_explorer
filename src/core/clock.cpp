//
// Created by progamers on 6/19/25.
//
#include "core/clock.h"

namespace raw::core {
inline long double ticks_now() {
	return ((std_clock::now()).time_since_epoch()).count();
}

clock::clock() : clock_stop(0) {
	start();
}

void clock::start() {
	clock_start = time(ticks_now());
}

void clock::stop() {
	clock_stop	= time(ticks_now());
	clock_start = clock_stop;
}

time clock::get_elapsed_time() const {
	if (clock_start != clock_stop) {
		time now(ticks_now());
		return time(now - clock_start);
	}
	return time(0);
}

bool clock::is_running() const {
	return clock_stop < clock_start;
}

time clock::restart() {
	auto elapsed_time = time(0);
	if (clock_start != clock_stop) {
		elapsed_time = time(ticks_now()) - clock_start;
	}
	start();
	return elapsed_time;
}

time clock::reset() {
	auto elapsed_time = time(0);
	if (clock_start != clock_stop) {
		elapsed_time = time(ticks_now()) - clock_start;
	}
	stop();
	return elapsed_time;
}

std::ostream &operator<<(std::ostream &os, const time &par) {
	os << par.val;
	switch (par.curr) {
		using enum time_rate;
	case NANO:
		os << "ns";
		break;
	case MICRO:
		os << "us";
		break;
	case MILLI:
		os << "ms";
		break;
	case ORD:
		os << "s";
		break;
	}
	return os;
}
} // namespace raw::core