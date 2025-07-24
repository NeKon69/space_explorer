//
// Created by progamers on 7/24/25.
//

#ifndef SPACE_EXPLORER_CLOSE_H
#define SPACE_EXPLORER_CLOSE_H

#include "command.h"
namespace raw::command {

class window_close : public raw::command::command {
public:
	void execute(raw::scene& scene) override;
};

} // namespace raw::command

#endif // SPACE_EXPLORER_CLOSE_H
