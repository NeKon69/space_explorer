//
// Created by progamers on 7/24/25.
//

#ifndef SPACE_EXPLORER_RUNNING_H
#define SPACE_EXPLORER_RUNNING_H
#include "command.h"
namespace raw::command {

class change_simulation_running : public raw::command::command {
public:
	void execute(raw::scene& scene) override;
};

} // namespace raw::command
#endif // SPACE_EXPLORER_RUNNING_H
