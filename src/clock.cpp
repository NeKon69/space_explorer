//
// Created by progamers on 6/19/25.
//
#include "clock.h"

namespace raw {

inline double ticks_now() {
	return std::chrono::duration_cast<std::chrono::microseconds>(
			   (std_clock::now()).time_since_epoch())
			   .count() /
		   1000.0;
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
		elapsed_time = time(time(ticks_now()) - clock_start);
	}
	start();
	return elapsed_time;
}
time clock::reset() {
	auto elapsed_time = time(0);
	if (clock_start != clock_stop) {
		elapsed_time = time(time(ticks_now()) - clock_start);
	}
	stop();
	return elapsed_time;
}
std::ostream& operator<<(std::ostream& os, const time& par) {
	os << par.val;
	switch (par.curr) {
		using enum raw::time_rate;
	case NANO:
		os << "ns\n";
		break;
	case MICRO:
		os << "us\n";
		break;
	case MILLI:
		os << "ms\n";
		break;
	case ORD:
		os << "s\n";
		break;
	}
	return os;
}
} // namespace raw