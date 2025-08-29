//
// Created by progamers on 6/2/25.
//
#include "core/game.h"
#include <cuda_runtime.h>
int main(int argc, char *argv[]) {
	cudaSetDevice(0);
	raw::core::game game("Mike Hawk");
	game.run();

	return 0;
}
