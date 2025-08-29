## Top Priority (Refactoring, need to be done now)

1. ~~Divide kernel for sphere tesselation into 2 kernels, one for vertices and one for another TBN data~~
2. ~~Split "icosahedron_generator" into two classes, one for managing resources second for launching kernels~~
3. Fix n-body by dividing single kernel into 2 separate ones, "kick-drift" and "kick" to properly work on more than 256
   objects
4. ~~Update basic vertices/indices generators to support normal mapping~~
5. Check if UV coordinate system actually works and if it has "edges"

## Top priority (features, need to be done now)

1. Finish texture streaming/generations
2. Introduce CUDA/OpenCL streams to each part of project (n-body, sphere tesselation, texture generation) to prevent
   chaos of randomly creating/deleting threads locally.
3. Introduce manager for GPU streams
4. ~~Negate vertices duplicating in the "sphere tessellation" process by using several kernels, this need to be done to
   prevent bullshit UV/TBN fuckups.~~

## Secondary Priority (Refactoring, need to be done later)

1. Make base classes for GPU-backend from which other classes will be inherited based on chosen backend (OpenCL or CUDA)
2. Refactor "playing_state" class to be more modular, introduce "event_manager", "input_handler", and other stuff