#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <emmintrin.h>  // SSE4.2: _mm_set_pd, _mm_sub_pd, _mm_mul_pd, _mm_sqrt_pd, _mm_add_pd

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

double distance_sse(const Point& p1, const Point& p2) {
    __m128d a = _mm_set_pd(p1.x, p1.y);
    __m128d b = _mm_set_pd(p2.x, p2.y);
    __m128d result = _mm_sub_pd(a, b);
    result = _mm_mul_pd(result, result);
    result = _mm_sqrt_pd(result);
    result = _mm_add_pd(result, _mm_shuffle_pd(result, result, 1));
    return ((double*)&result)[0];
}

double distance(const Trajectory& t1, const Trajectory& t2) {
    double total = 0.0;
    for (size_t i = 0; i < t1.points.size(); ++i) {
        total += distance_sse(t1.points[i], t2.points[i]);
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
            std::cout << "Distance between trajectory " << i << " and " << j << ": " << distance(trajectories[i], trajectories[j]) << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-start;
    std::cout << "Time taken: " << diff.count() << " s\n";

    return 0;
}
//29s 495s