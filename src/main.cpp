//
// Created by progamers on 6/2/25.
//
#include "core/game.h"
#include <cuda_runtime.h>
/**
 * @brief Program entry point that constructs and runs the game.
 *
 * The function creates a raw::core::game instance initialized with the player
 * name "Mike Hawk", invokes its run() method, and returns with status 0.
 *
 * @param argc Number of command-line arguments (unused).
 * @param argv Array of command-line argument strings (unused).
 * @return int Exit status code; 0 indicates successful termination.
 */
int main(int argc, char *argv[]) {
	raw::core::game game("Mike Hawk");
	game.run();

	return 0;
}
