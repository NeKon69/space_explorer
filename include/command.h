//
// Created by progamers on 7/24/25.
//

#ifndef SPACE_EXPLORER_COMMAND_H
#define SPACE_EXPLORER_COMMAND_H

namespace raw {
class scene;

namespace command {
class command {
public:
	virtual ~command()						= default;
	virtual void execute(raw::scene& scene) = 0;
};
} // namespace command

} // namespace raw
#endif // SPACE_EXPLORER_COMMAND_H
