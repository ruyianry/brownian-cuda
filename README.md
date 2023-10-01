# Brownian Motion CUDA Simulator

A simulator of Brownian motion of fixed-sized particles within a 2-dimensional circular arena.

## Running the program

### Input generation

The input should be generated using `generate_input.py` in the `input` directory. The input is generated using the following command:

`python3 generate_input.py <NUM_PARTICLES> <NUM_ITERATIONS> <OUTPUT_FILE_NAME>`

This will generate `<NUM_PARTICLES>` number of particles with randomly assigned x and y coordinates and velocities along both axes.

### Compilation

`Makefile` is provided for compilation. To compile, run `make` in the root directory.

### Running

`./simulator.out < input/<INPUT_TEXT_NAME>.txt`

The output will be saved under `./output`.


## Some notes

### Assumptions

We made a few assumptions about the simulation.

1. Particles collide with only one other particle at a time (each step).
2. If particles collide with the wall and another particle at the same time, the particle will become stationary with respect to the wall.

### Optimization

#### Flow control and branches
As threads in a warp execute differently at different control flows and threads may have to wait for the batch to finish, branches are minimized in the execution.
We achieved this by first calculating the type of collision for each particle pair and then executing the collision in a batch. This way, we can avoid branches in collision detection.
The reduction in branches is at the cost of more instructions, but considering the tradeoff between the two, we believe this is a good optimization.

CUDA stream is used for asynchronous execution of the batch. This is to avoid the threads in a warp to wait for each other.
This can be seen in the `timeWallCollision`.

#### Possible direction of improvement
1. Task granularity can be further investigated. We can try to increase the number of particles per block and reduce the number of blocks. This may reduce the overhead of launching a kernel.
2. In this implementation, `__managed__` variables are used to declare the global variables. This is to avoid the overhead of copying the variables to the device. 
However, this may cause the program to run slower by copying overhead. 
If possible, use `__device__` for all variables and use `cudaMemcpy` to copy the variables to the host only when needed.
