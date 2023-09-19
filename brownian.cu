#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <atomic>
#include <stdio.h>

#include <particle.h>


#define NUM_STREAMS 2
#define NOT_FOUND -1

using namespace std;

mt19937 rng;   // random number generator
random_device rd;
__managed__ int n, l, r, s // number of particles, length of side of square, radius of particles, steps


class Particle {
public:
    operator string() const { // overloading the cast operator to convert the object to a string
        char buffer[200];
        snprintf(buffer, 200, "%d %.8lf %.8lf %.8lf %.8lf", i, x, y, vX, vY);
        return buffer;
    }

    // attributes are explicitly declared as public
    int i;
    double x;
    double y;
    double vX;
    double vY;
    int pColl; // particle collisions. 0 = no collision, 1 = collision with particle
    int wColl; // wall collisions. 0 = no collision, 1 = collision with left wall, 2 = collision with right wall, 3 = collision with top wall, 4 = collision with bottom wall

    Particle() {};

    Particle(int i, double x, double y, double vX, double vY, int l) {
        this->i = i;
        this->x = x;
        this->y = y;
        this->vX = vX;
        this->vY = vY;
        this->pColl = 0;
        this->wColl = 0;
    }

    __device__ __host__ int getIndex() {
        return this->i;
    }

    string getFullRepresentation() {
        return [=] {
            std::ostringstream stream;
            stream << std::fixed << std::setprecision(8); // precision of 8 decimal places
            stream << i << " " << x << " " << y << " " << vX << " " << vY << " " << pColl << " " << wColl;
            return stream.str();
        };
    }
};

class CollisionEvent {
public:
    const static int WALL = 0;
    const static int PARTICLE = 1;
    const static int NONE = 2;

    Particle *first_p; // first colliding particle
    Particle *second_p; // second colliding particle
    double time; // time of collision
    int type; // type of collision, refer to the 3 constants above

    __host__ __device__ bool operator==(CollisionEvent other) {
        int firstIndex = (*this->first).getIndex();
        int otherFirstIndex = (*other.first).getIndex();
        if (this->second != nullptr) {
            int secondIndex = (*this->second).getIndex();
            int otherSecondIndex = (*other.second).getIndex();
            return (firstIndex == otherSecondIndex && secondIndex == otherFirstIndex) ||
                   (firstIndex == otherFirstIndex && secondIndex == otherSecondIndex);
        }
        return (firstIndex == otherFirstIndex);
    }


    __host__ __device__ bool operator<(CollisionEvent other) {
        if (this->time == other.getTime()) {
            return this->getSmallestIndex() > other.getSmallestIndex();
        } else {
            return this->time > other.getTime();
        }
    }

    __host__ __device__ CollisionEvent() {}

    __host__ __device__ virtual ~CollisionEvent() {}

    // wall collision event
    __host__ __device__ CollisionEvent(Particle *first, double time) {
        this->first = first;
        this->second = nullptr; // second particle is null for wall collisions
        this->time = time;
        this->type = this->WALL;
    }

    // particle collision event
    __host__ __device__ CollisionEvent(Particle *first, Particle *second, double time) {
        this->first = first;
        this->second = second;
        this->time = time;
        this->type = this->PARTICLE;
    }

    // no collision event
    __host__ __device__ CollisionEvent(Particle *first) {
        this->first = first;
        this->second = nullptr;
        this->time = 1.0;
        this->type = this->NONE;
    }

    __host__ __device__ double getTime() {
        return this->time;
    }

    __host__ __device__ int getType() {
        return this->type;
    }

    __host__ __device__ double getSmallestIndex() {
        if (this->second != nullptr) {
            return (*first).getIndex() < (*second).getIndex() ? (*first).getIndex() : (*second).getIndex();
        }
        return (*first).getIndex();
    }
};


// cuda streams
cudaStream_t streams[NUM_STREAMS];

// particle objects
__managed__ Particle *particles; // array of particles

// collision times arrays
// store in a __managed__ array so that it can be accessed by both host and device instead of copying it back and forth
__managed__ double **particleCollisionTimes; // 2D array of particle collision times of n * n size
__managed__ double *wallCollisionTimes; // array of wall collision times of n size

// collision events
__managed__ CollisionEvent *temp; // temporary array to store collision events
__managed__ int *found; // array to store the number of collisions found for each particle

__managed__ CollisionEvent **particleCollisions; // array of particle collision events
__managed__ int particleCollisionsCount; // number of particle collision events

__managed__ CollisionEvent **wallCollisions; // array of wall collision events
__managed__ int wallCollisionsCount; // number of wall collision events

__managed__ CollisionEvent **noCollisions; // array of no collision events
__managed__ int noCollisionsCount; // number of no collision events

// function headers
__host__ void moveParticles(Particle *particles);

__global__ void findEarliestCollision();

__global__ void timeParticleCollision();

__global__ void timeWallCollision();

__global__ void executeParticleCollision();

__global__ void executeWallCollision();

__global__ void executeNoCollision();

__host__ int main(void) { // main is on host
    string command;
    cin >> n >> l >> r >> s >> command;

    // random init
    rng.seed(rd(42));
    uniform_real_distribution<double> pos(r, l - r);
    uniform_real_distribution<double> velocity((double) l / (8 * r), (double) l / 4);

    // assign a device space for the particles array
    cudaError_t cudaStatus = cudaMallocManaged(&particles, sizeof(Particle) * n);
    int scan_count;
    for (scan_count = 0; scan_count < n; ++scan_count) {
        int index;
        double x;
        double y;
        double vX; // velocity in x direction
        double vY; // velocity in y direction
        int count;
        count = scanf("%d %lf %lf %lf %lf", &index, &x, &y, &vX, &vY);
        if (count == EOF || count <= 0) {
            break;
        }
        particles[scan_count] = Particle(index, x, y, vX, vY, l);
    }
    for (int j = scan_count; j < n; j++) {
        double x = pos(rng);
        double y = pos(rng);
        double vX = velocity(rng);
        double vY = velocity(rng);
        particles[scan_count] = Particle(j, x, y, vX, vY, l);
    }


    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }
    cudaMallocManaged(&particleCollisionTimes, sizeof(double *) * n);
    for (int i = 0; i < n; ++i) {
        cudaMallocManaged(&particleCollisionTimes[i], sizeof(double) * n);
    }
    cudaMallocManaged(&wallCollisionTimes, sizeof(double) * n);
    cudaMallocManaged(&found, sizeof(int) * n);
    cudaMallocManaged(&particleCollisions, sizeof(CollisionEvent *) * n);
    cudaMallocManaged((void **) &particleCollisionsCount, sizeof(int));
    cudaMallocManaged(&wallCollisions, sizeof(CollisionEvent *) * n);
    cudaMallocManaged((void **) &wallCollisionsCount, sizeof(int));
    cudaMallocManaged(&noCollisions, sizeof(CollisionEvent *) * n);
    cudaMallocManaged((void **) &noCollisionsCount, sizeof(int));
    cudaMallocManaged(&temp, sizeof(CollisionEvent) * n);

    // recording start time
    auto start = chrono::high_resolution_clock::now();

    for (int i = 0; i < s; ++i) {
        if (!command.compare("print")) {
            for (int j = 0; j < n; ++j) {
                cout << i << " " << (string) particles[j] << endl;
            }
        }
        moveParticles(particles); // simulation
    }

    // recording end time
    auto finish = chrono::high_resolution_clock::now();

    for (int j = 0; j < n; ++j) {
        cout << s << " " << particles[j].getFullRepresentation() << endl;
    }
    double timeTaken = (double) chrono::duration_cast<chrono::nanoseconds>(finish - start).count() / 1000000000;
    printf("Time taken: %.5f s for %d steps\n", timeTaken, s);
    return 0;
}


__host__ void moveParticles(Particle *particles) {
    for (int i = 0; i < n; ++i) {
        found[i] = NOT_FOUND;
    }
    particleCollisionsCount = 0;
    wallCollisionsCount = 0;
    noCollisionsCount = 0;

    // calculate collision times
    timeWallCollision<<<(n - 1) / 64 + 1, 64, 0, streams[0]>>>();

    dim3 threadsPerBlock(16, 16, 1);
    dim3 blocksPerGrid((n - 1) / 16 + 1, (n - 1) / 16 + 1);
    timeParticleCollision<<<blocksPerGrid, threadsPerBlock, 0, streams[1]>>>();

    cudaDeviceSynchronize();

    // find valid collisions
    int foundCount = 0;
    while (foundCount != n) {
        findEarliestCollision<<<(n - 1) / 64 + 1, 64>>>();
        cudaDeviceSynchronize();
        for (int i = 0; i < n; ++i) {
            if (found[i] != NOT_FOUND) continue;
            CollisionEvent *e = &temp[i];
            // particle-particle collision
            if ((*e).getType() == CollisionEvent::PARTICLE) {
                int otherIndex = (*(*e).second).getIndex();
                if (temp[otherIndex].getType() == CollisionEvent::PARTICLE) {
                    if ((*e) == temp[otherIndex] && i < otherIndex) {
                        found[i] = 0;
                        found[otherIndex] = 0;
                        foundCount += 2;
                        particleCollisions[particleCollisionsCount++] = e;
                    }
                }
            }
                // particle-wall collision
            else if ((*e).getType() == CollisionEvent::WALL) {
                found[i] = 0;
                ++foundCount;
                wallCollisions[wallCollisionsCount++] = e;
            }
                // no collision
            else {
                found[i] = 0;
                ++foundCount;
                noCollisions[noCollisionsCount++] = e;
            }
        }
    }

    // apply valid collisions
    executeParticleCollision<<<(particleCollisionsCount - 1) / 64 + 1, 64, 0, streams[0]>>>();
    executeWallCollision<<<(wallCollisionsCount - 1) / 64 + 1, 64, 0, streams[1]>>>();
    executeNoCollision<<<(n - 1) / 64 + 1, 64, 0, streams[1]>>>();
    cudaDeviceSynchronize();
}

__global__ void findEarliestCollision() {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n || found[index] != NOT_FOUND) return;
    // first assume no collision
    temp[index] = CollisionEvent(&particles[index]);

    // check for particle-wall collision
    if (wallCollisionTimes[index] < temp[index].getTime() && wallCollisionTimes[index] < 1) {
        temp[index] = CollisionEvent(&particles[index], wallCollisionTimes[index]);
    }

    // check for particle-particle collision
    for (int j = 0; j < n; ++j) {
        if (index == j || found[j] != NOT_FOUND) continue;
        double time = particleCollisionTimes[index][j];

        if (time > -1 && time < temp[index].getTime() && time < 1) {
            temp[index] = CollisionEvent(&particles[index], &particles[j], time);
        }
    }

}


// "input": 2 Particles
// "output": Returns time taken before collision occurs if they collide, negative value otherwise.
__global__ void timeParticleCollision() {
    int firstIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int secondIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if (firstIndex >= n || secondIndex >= n || firstIndex >= secondIndex) return;
    else {
        Particle first = particles[firstIndex];
        Particle second = particles[secondIndex];
        //a, b and c are as in the quadratic formula representation.
        //t, the time taken for the 2 circles to touch, is the unknown variable we are solving for
        //by taking difference in circle centres, setting an unknown t for collision time, and then taking distance moved in time t,
        //we can solve for t such that the circle centers are <= 2r and therefore collide. 4r^2 is to solve for radius distance.
        double c = pow((first.x - second.x), 2) + pow((first.y - second.y), 2) - 4 * r * r;
        double b = 2 * ((first.x - second.x) * (first.vX - second.vX) + (first.y - second.y) * (first.vY - second.vY));
        double a = pow((first.vX - second.vX), 2) + pow((first.vY - second.vY), 2);
        //check for solution
        double solfirst;
        if (b * b - 4 * a * c < 0) {
            solfirst = 100000.0;
        } else {
            //else if there is a solution, the one with smaller value should be the main collision. Second value is after the 2 circles phase through each other
            solfirst = (-sqrt(b * b - 4 * a * c) - b) / (2 * a);
            if (solfirst < 0) {
                solfirst = (sqrt(b * b - 4 * a * c) - b) / (2 * a);
                if (solfirst > 0) solfirst = 0;
                else solfirst = 100000.0;
            }
        }
        particleCollisionTimes[first.i][second.i] = solfirst;
        particleCollisionTimes[second.i][first.i] = solfirst;
    }
}

// "input": 1 Particle
// "output": Returns time taken before collision occurs if it collides with wall, negative value otherwise.
__global__ void timeWallCollision() {
    int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (particleIndex >= n) {
        return;
    } else {
        Particle particle = particles[particleIndex];
        //check for x wall, y wall collisions
        double xCollide =
                particle.vX < 0 ? (particle.x - r) / (0 - particle.vX) : ((double) l - particle.x - r) / particle.vX;
        double yCollide =
                particle.vY < 0 ? (particle.y - r) / (0 - particle.vY) : ((double) l - particle.y - r) / particle.vY;
        wallCollisionTimes[particle.i] = fmin(xCollide, yCollide);
    }
}

__global__ void executeParticleCollision() {
    int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (particleIndex < particleCollisionsCount) {
        CollisionEvent *e = particleCollisions[particleIndex];
        Particle *first = e->first;
        Particle *second = e->second;
        double time = e->time;
        if (first->getIndex() >= second->getIndex())
            return;
        //move them to proper position first
        first->x += time * first->vX;
        first->y += time * first->vY;
        second->x += time * second->vX;
        second->y += time * second->vY;

        //perform collision here
        //find normal vector
        double normalX = first->x - second->x;
        double normalY = first->y - second->y;
        double normalMag = sqrt(pow(normalX, 2) + pow(normalY, 2));
        normalX = normalX / normalMag;
        normalY = normalY / normalMag;
        double tangentX = -normalY;
        double tangentY = normalX;
        //compute velocity vectors wrt to normal and tangent
        double vFirstNormal = normalX * first->vX + normalY * first->vY;
        double vFirstTangent = tangentX * first->vX + tangentY * first->vY;
        double vSecondNormal = normalX * second->vX + normalY * second->vY;
        double vSecondTangent = tangentX * second->vX + tangentY * second->vY;
        //collision simply swaps velocities
        double temp = vFirstNormal;
        vFirstNormal = vSecondNormal;
        vSecondNormal = temp;
        first->vX = vFirstNormal * normalX + vFirstTangent * tangentX;
        first->vY = vFirstNormal * normalY + vFirstTangent * tangentY;
        second->vX = vSecondNormal * normalX + vSecondTangent * tangentX;
        second->vY = vSecondNormal * normalY + vSecondTangent * tangentY;
        //eliminate negative 0s
        if (first->vX == -0.0) first->vX = 0.0;
        if (first->vY == -0.0) first->vY = 0.0;
        if (second->vX == -0.0) second->vX = 0.0;
        if (second->vY == -0.0) second->vY = 0.0;
        //Continue to move them here
        //Check for wall collisions and stop the particle at wall if so
        double timeToMove;
        double xCollide = first->vX < 0 ? (first->x - r) / (0 - first->vX) : ((double) l - r - first->x) / first->vX;
        double yCollide = first->vY < 0 ? (first->y - r) / (0 - first->vY) : ((double) l - r - first->y) / first->vY;
        if (xCollide >= 1 - time && yCollide >= 1 - time) {
            timeToMove = 1 - time;
        } else {
            timeToMove = fmin(xCollide, yCollide);
        }
        first->x += timeToMove * first->vX;
        first->y += timeToMove * first->vY;

        xCollide = second->vX < 0 ? (second->x - r) / (0 - second->vX) : ((double) l - r - second->x) / second->vX;
        yCollide = second->vY < 0 ? (second->y - r) / (0 - second->vY) : ((double) l - r - second->y) / second->vY;
        if (xCollide >= 1 - time && yCollide >= 1 - time) {
            timeToMove = 1 - time;
        } else {
            timeToMove = fmin(xCollide, yCollide);
        }
        second->x += timeToMove * second->vX;
        second->y += timeToMove * second->vY;
        first->pColl++;
        second->pColl++;
    }
}

__global__ void executeWallCollision() {
    int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (particleIndex < wallCollisionsCount) {
        CollisionEvent *e = wallCollisions[particleIndex];
        Particle *first = e->first;
        //check for x wall collisions
        //check for y wall collisions
        double xCollide = first->vX < 0 ? (first->x - r) / (0 - first->vX) : ((double) l - first->x - r) / first->vX;
        double yCollide = first->vY < 0 ? (first->y - r) / (0 - first->vY) : ((double) l - first->y - r) / first->vY;
        double earlierTime = fmin(xCollide, yCollide);
        double laterTime = fmax(xCollide, yCollide);
        first->x += earlierTime * first->vX;
        first->y += earlierTime * first->vY;
        //Reverse direction depending on which collision happens first
        if (xCollide <= yCollide) {
            first->vX = -first->vX;
        }
        if (yCollide <= xCollide) {
            first->vY = -first->vY;
        }
        //artificially set timing to allow particle to continue after hitting corner
        if (xCollide == yCollide) {
            laterTime = 1;
        }
        first->x += (fmin(1.0, laterTime) - earlierTime) * first->vX;
        first->y += (fmin(1.0, laterTime) - earlierTime) * first->vY;
        first->wColl++;
    }
}

__global__ void executeNoCollision() {
    int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (particleIndex < noCollisionsCount) {
        CollisionEvent *e = noCollisions[particleIndex];
        Particle *first = e->first;
        first->x += first->vX;
        first->y += first->vY;
    }
}
