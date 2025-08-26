//
// Created by progamers on 7/24/25.
//

#ifndef SPACE_EXPLORER_COMMAND_H
#define SPACE_EXPLORER_COMMAND_H

namespace raw {
class playing_state;

namespace command {
class command {
public:
	virtual ~command()								= default;
	virtual void execute(raw::playing_state& scene) = 0;
};
} // namespace command

} // namespace raw
#endif // SPACE_EXPLORER_COMMAND_H
