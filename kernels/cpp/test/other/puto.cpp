#include <iostream>
#include <vector>
#include <random>
#include <chrono>

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

double distance(const Point& p1, const Point& p2) {
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    return sqrt(dx * dx + dy * dy);
}

double distance(const Trajectory& t1, const Trajectory& t2) {
    double total = 0.0;
    for (size_t i = 0; i < t1.points.size(); ++i) {
        total += distance(t1.points[i], t2.points[i]);
    }
    return total;
}

int main() {
    const int num_trajectories =10000;
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
    std::chrono::duration<double> diff = end-start;
    std::cout << "Time taken: " << diff.count() << " s\n";

    return 0;
}

//575.81s

/*
正常执行：575s
编译器优化：504s 改用"\n"506s
SSE指令集优化:495s 501s

*/ 