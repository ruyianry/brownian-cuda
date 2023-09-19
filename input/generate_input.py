#!/usr/bin/env python3

import random

# Input parameters
num_particles = 1000  # Change this to the desired number of particles
side_length = 20000
particle_radius = 1
num_of_step = 500
output_filename = "input.txt"

# Open the file for writing
with open(output_filename, "w") as file:
    # Write the header lines
    file.write(f"{num_particles}\n")
    file.write(f"{side_length}\n")
    file.write(f"{particle_radius}\n")
    file.write(f"{seed_for_step}\n")
    file.write("perf\n")

    # Generate and write particle data
    for i in range(num_particles):
        x = random.uniform(particle_radius, side_length - particle_radius)
        y = random.uniform(particle_radius, side_length - particle_radius)
        vx = random.uniform(100, 5000)
        vy = random.uniform(100, 5000)

        file.write(f"{i} {x} {y} {vx} {vy}\n")

print(f"File '{output_filename}' generated successfully.")