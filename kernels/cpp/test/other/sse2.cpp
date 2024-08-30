#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <pmmintrin.h> // Include SSE3 intrinsics

struct Point {
    double x, y;
};

class Trajectory {
public:
    std::vector<Point> points;

    Trajectory(int num_points) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-100.0, 100.0);

        for (int n = 0; n < num_points; ++n) {
            points.push_back({dis(gen), dis(gen)});
        }
    }
};

// SSE vectorized distance calculation
__m128d distance_sse(const Point& p1, const Point& p2) {
    __m128d p1vec = _mm_set_pd(p1.x, p1.y);
    __m128d p2vec = _mm_set_pd(p2.x, p2.y);
    __m128d diff = _mm_sub_pd(p2vec, p1vec);
    __m128d sq = _mm_mul_pd(diff, diff);
    __m128d sum = _mm_hadd_pd(sq, sq);
    return sum;
}

double distance(const Trajectory& t1, const Trajectory& t2) {
    double total = 0.0;
    for (size_t i = 0; i < t1.points.size(); i += 2) { // Process two points at a time
        __m128d dist = distance_sse(t1.points[i], t2.points[i]);
        total += _mm_cvtsd_f64(dist); // Add the lower double of the result
        if (i + 1 < t1.points.size()) { // Check if there's a second point to process
            dist = distance_sse(t1.points[i + 1], t2.points[i + 1]);
            total += _mm_cvtsd_f64(_mm_unpackhi_pd(dist, dist)); // Add the upper double of the result
        }
    }
    return total;
}

int main() {
    const int num_trajectories = 10000;
    const int num_points = 48;

    std::vector<Trajectory> trajectories;
    for (int i = 0; i < num_trajectories; ++i) {
        trajectories.push_back(Trajectory(num_points));
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_trajectories; ++i) {
        for (int j = i + 1; j < num_trajectories; ++j) {
            std::cout << "Distance between trajectory " << i << " and " << j << ": " << distance(trajectories[i], trajectories[j]) << "\n";
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time taken: " << diff.count() << " s\n";

    return 0;
}
