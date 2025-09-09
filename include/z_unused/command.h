//
// Created by progamers on 7/24/25.
//

#pragma once

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
