#include "http_handler.h"
#include "bt_compute.h"
#include "features.h"
#include "json11.hpp"

using json11::Json;
using std::string;
using namespace httplib;
std::string dump_headers(const Headers &headers) {
    std::string s;
    char buf[BUFSIZ];

    for (auto it = headers.begin(); it != headers.end(); ++it) {
        const auto &x = *it;
        snprintf(buf, sizeof(buf), "%s: %s\n", x.first.c_str(), x.second.c_str());
        s += buf;
    }

    return s;
}

std::string log(const Request &req, const Response &res) {
    std::string s;
    char buf[BUFSIZ];

    s += "================================\n";

    snprintf(buf, sizeof(buf), "%s %s %s", req.method.c_str(),
            req.version.c_str(), req.path.c_str());
    s += buf;

    std::string query;
    for (auto it = req.params.begin(); it != req.params.end(); ++it) {
        const auto &x = *it;
        snprintf(buf, sizeof(buf), "%c%s=%s",
                (it == req.params.begin()) ? '?' : '&', x.first.c_str(),
                x.second.c_str());
        query += buf;
    }
    snprintf(buf, sizeof(buf), "%s\n", query.c_str());
    s += buf;

    s += dump_headers(req.headers);

    s += "--------------------------------\n";

    snprintf(buf, sizeof(buf), "%d %s\n", res.status, res.version.c_str());
    s += buf;
    s += dump_headers(res.headers);
    s += "\n";

    if (!res.body.empty()) { s += res.body; }

    s += "\n";

    return s;
}

void handleSingleCompute(const httplib::Request &req, httplib::Response &res)
{
    string err;

    Json requestJson = Json::parse(req.body, err);
    if (!err.empty()) {
        res.set_content("Invalid JSON", "text/plain");
        res.status = 400; // Bad Request
        return;
    }

    std::string file_path = requestJson["file"].string_value();
    int hour = requestJson["hour"].int_value();
    float latitude = requestJson["latitude"].number_value();
    float longitude = requestJson["longitude"].number_value();
    float level = requestJson["level"].number_value();

    if (file_path.empty() || hour == 0 || latitude == 0.0f || longitude == 0.0f || level == 0.0f) {
        res.set_content("Missing or invalid parameters", "text/plain");
        res.status = 400; 
        return;
    }

    std::unique_ptr<BackTraj> bt = std::make_unique<BackTraj>();
    bt->InitNc(1, file_path.c_str());


    Point cur_loc(latitude, longitude, level);
    std::vector<Point> trajectory;
    bt->ComputeSingle(file_path.c_str(), hour, cur_loc, 3600.0f,trajectory);

    // Json responseJson = Json::object {
    //         {"trajectory", Json::array(trajectory.begin(), trajectory.end())}
    //     };
    // res.set_content(responseJson.dump(), "application/json");
    std::vector<Json> json_trajectory;
    std::transform(trajectory.begin(), trajectory.end(), std::back_inserter(json_trajectory), [](const Point& p) {
        return p.to_json();
    });

    Json responseJson = Json::object {
        {"trajectory", json_trajectory}
    };
    
    res.set_content(responseJson.dump(), "application/json");

    log(req,res);
    
}

void handleComputePer6h(const httplib::Request &req, httplib::Response &res)
{
    string err;

    Json requestJson = Json::parse(req.body, err);
    if (!err.empty()) {
        res.set_content("Invalid JSON", "text/plain");
        res.status = 400; // Bad Request
        return;
    }

    auto files = requestJson["file"].array_items();
    for (const auto& file : files) {
        std::cout << "File: " << file.string_value() << std::endl;
    }

    std::string yesterday_nc = files[0].string_value();
    std::string today_nc = files[1].string_value();

    int hour = requestJson["hour"].int_value();
    float latitude = requestJson["latitude"].number_value();
    float longitude = requestJson["longitude"].number_value();
    float level = requestJson["level"].number_value();

    if (today_nc.empty() || yesterday_nc.empty() || hour == 0 || latitude == 0.0f || longitude == 0.0f || level == 0.0f) {
        res.set_content("Missing or invalid parameters", "text/plain");
        res.status = 400; 
        return;
    }

    //std::unique_ptr<BackTraj> bt = std::make_unique<BackTraj>();
    BackTraj* bt = BackTraj::getInstance();

    bt->InitNc(2, yesterday_nc.c_str(),today_nc.c_str());

    Point cur_loc(latitude, longitude, level);
    std::vector<std::vector<Point>> trajectorys;
    bt->ComuteSinglePer6h(yesterday_nc,today_nc, hour, cur_loc, 3600.0f,trajectorys);

    vector<vector<Json>> json_trajectory_list;

    for (const auto& trajectory : trajectorys) {
            std::vector<json11::Json> json_trajectory;
            std::transform(trajectory.begin(), trajectory.end(), std::back_inserter(json_trajectory), [](const Point& p) {
                return p.to_json();
            });
            json_trajectory_list.push_back(json_trajectory);
        }

    json11::Json responseJson = json11::Json::object {
        {"trajectories", json11::Json::array({json_trajectory_list.begin(), json_trajectory_list.end()})}
    };

    std::string response_string = responseJson.dump();
    res.set_content(responseJson.dump(), "application/json");
    std::cout << log(req,res) << std::endl;;
}

void handleComputeFeatures(const httplib::Request &req, httplib::Response &res)
{
    string err;

    Json requestJson = Json::parse(req.body, err);
    if (!err.empty()) {
        res.set_content("Invalid JSON", "text/plain");
        res.status = 400; // Bad Request
        return;
    }
    vector<std::vector<Point>> inputMatrix;

    auto trajectories = requestJson["trajectories"].array_items();


    for (const auto& trajectory : trajectories) {
        // 遍历轨迹中的点
        vector<Point> single_traj;
        for (const auto& point : trajectory.array_items()) {
            // 获取点的坐标
            float x = point[0].number_value();
            float y = point[1].number_value();
            float z = point[2].number_value();
            Point loc(x, y, z);
            single_traj.emplace_back(loc);
            // 输出点的坐标
            //std::cout << "Point: (" << x << ", " << y << ", " << z << ")" << std::endl;
        }
        inputMatrix.emplace_back(std::move(single_traj));
    }

    std::unique_ptr<FeatureComputer> feature_ptr = std::make_unique<FeatureComputer>();

    vector<vector<float>> distanceMatrix,normalizedMatrix;
    vector<vector<float>> output_marix;

    feature_ptr->DtwCompute(inputMatrix,distanceMatrix);
    feature_ptr->NormalizeFeatures(distanceMatrix, normalizedMatrix);


    vector<vector<json11::Json>> json_feature_list;

    for (const auto& feature : normalizedMatrix) {
        std::vector<json11::Json> json_feature;
        for (const auto& value : feature) {
            json_feature.push_back(value);
        }
        json_feature_list.push_back(json_feature);
    }

    // 创建响应的Json对象
    json11::Json responseJson = json11::Json::object {
        {"features", json_feature_list}
    };

    std::string response_string = responseJson.dump();
    res.set_content(responseJson.dump(), "application/json");
    std::cout << log(req,res) << std::endl;
}
