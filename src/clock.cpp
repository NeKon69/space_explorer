//
// Created by progamers on 6/19/25.
//
#include "clock.h"

#define ticks_now() std::chrono::duration_cast<std::chrono::microseconds>((std_clock::now()).time_since_epoch()).count() / 1000.0

namespace raw {
clock::clock() : clock_stop(0) {
	start();
}
void clock::start() {
	clock_start = ticks_now();
}
void clock::stop() {
	clock_stop = ticks_now();
    clock_start = clock_stop;
}
time clock::get_elapsed_time() const {
    if(clock_start != clock_stop) {
        time now(ticks_now());
        return now - clock_start;
    }
    return 0;
}
bool clock::is_running() const {
	return clock_stop < clock_start;
}
time clock::restart() {
    time elapsed_time = 0;
    if(clock_start != clock_stop) {
        elapsed_time = time(ticks_now()) - clock_start;
    }
	start();
	return elapsed_time;
}
time clock::reset() {
    time elapsed_time = 0;
    if(clock_start != clock_stop) {
        elapsed_time = time(ticks_now()) - clock_start;
    }
    stop();
    return elapsed_time;
}
std::ostream& operator <<(std::ostream& os, time par) {
    os << par.val;
    switch(par.curr) {
        case time_rate::NANO:
            os << "ns\n";
            break;
        case time_rate::MICRO:
            os << "us\n";
            break;
        case time_rate::MILLI:
            os << "ms\n";
            break;
        case time_rate::ORD:
            os << "s\n";
            break;
    }
    return os;
}
} // namespace raw