#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

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

struct Task {
    int i, j;
};

std::queue<Task> tasks;
std::mutex mtx;
std::condition_variable cv;

void worker(const std::vector<Trajectory>& trajectories) {
    while (true) {
        Task task;
        {
            std::unique_lock<std::mutex> lock(mtx);
            if (tasks.empty()) {
                return;
            }
            task = tasks.front();
            tasks.pop();
        }
        double dist = distance(trajectories[task.i], trajectories[task.j]);
        std::cout << "Distance between trajectory " << task.i << " and " << task.j << ": " << dist << std::endl;
    }
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
            tasks.push({i, j});
        }
    }

    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        threads[i] = std::thread(worker, std::ref(trajectories));
    }
    for (auto& thread : threads) {
        thread.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-start;
    std::cout << "Time taken: " << diff.count() << " s\n";

    return 0;
}
//38s