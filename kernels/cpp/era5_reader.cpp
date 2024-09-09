#include "era5_reader.h"
#include "omp.h"
#include <algorithm>
#include <limits>
#include <cmath>
#include <chrono>

using namespace netCDF;
void NetCDFReader::InitIndex()
{
    vector<float> levels(dim1_);
    vector<float> latitudes(dim2_);
    vector<float> longitudes(dim3_);

    int ret = Nc1dReader("level",levels);
    if(ret < 0){
        std::cerr << "read levels failed." << std::endl;
    }

    ret = Nc1dReader("latitude",latitudes);
    if(ret < 0){
        std::cerr << "read latitudes failed." << std::endl;
    }

    ret = Nc1dReader("longitude",longitudes);
    if(ret < 0){
        std::cerr << "read longitudes failed." << std::endl;
    }

    index_map["level"] = std::move(levels);
    index_map["latitude"] = std::move(latitudes);
    index_map["longitude"] = std::move(longitudes);


    vector<vector<vector<vector<float>>>> value_4d;

    ret = Nc4dReader("u",value_4d);
    if(ret < 0){
        std::cerr << "read u failed." << std::endl;
    }
    u_4d_ = std::move(value_4d);
    value_4d.clear();

    ret = Nc4dReader("v",value_4d);
    if(ret < 0){
        std::cerr << "read v failed." << std::endl;
    }
    v_4d_ = std::move(value_4d);
    value_4d.clear();

    ret = Nc4dReader("w",value_4d);
    if(ret < 0){
        std::cerr << "read w failed." << std::endl;
    }
    w_4d_ = std::move(value_4d);
    value_4d.clear();

}
std::tuple<int,int,int,int,bool> NetCDFReader::getIndex(int time_index, float level, float latitude, float longitude)
{
    int levIndex = this->seq_select_lev(index_map["level"],level);
    int latIndex = this->bin_select_lat(index_map["latitude"],latitude);
    int lonIndex = this->bin_select_lon(index_map["longitude"],longitude);

    if(levIndex < 0 || latIndex < 0 || lonIndex < 0 || time_index < 0){
        return std::make_tuple(time_index,levIndex,latIndex,lonIndex,false);
    }

    return std::make_tuple(time_index,levIndex,latIndex,lonIndex,true);
}
float NetCDFReader::getVar(std::tuple<int, int, int, int> &index, string varName)
{
    int timeIndex = std::get<0>(index);
    int levIndex = std::get<1>(index);
    int latIndex = std::get<2>(index);
    int lonIndex = std::get<3>(index);

    float value = -1.0f;

    try
    {
        if(varName == "u"){
            value = u_4d_[timeIndex][levIndex][latIndex][lonIndex];
        }else if(varName == "v"){
            value = v_4d_[timeIndex][levIndex][latIndex][lonIndex];
        }else if(varName == "w"){
            value = w_4d_[timeIndex][levIndex][latIndex][lonIndex];
        }else{
            ;
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    return value;
}
int NetCDFReader::Nc1dReader(const string &value_name, vector<float> &values_1d)
{
    
    NcFile dataFile(file_path_, NcFile::FileMode::read,NcFile::FileFormat::nc4);
    auto data = dataFile.getVar(value_name);
    if (data.isNull()){
        std::cerr << "the " << value_name << "is null." <<std::endl;
        return -1;
    }
                        
    try
    {
        data.getVar(values_1d.data());
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    return 0;
}

int NetCDFReader::Nc2dReader(const string& value_name,vector<vector<float>>& values_2d)
{
    NcFile dataFile(file_path_, NcFile::FileMode::read, NcFile::FileFormat::classic);
    auto data = dataFile.getVar(value_name);
    
    if (data.isNull()){
        std::cerr << "The variable " << value_name << " is null." << std::endl;
        return -1;
    }

    try {

        std::vector<float> flatData(dim0_ * dim1_ );
        data.getVar(values_2d[0].data());

    } catch(const std::exception& e) {
        std::cerr << "Exception: " << e.what() << '\n';
        return -1;
    }
    return 0;
}

int NetCDFReader::Nc3dReader(const string &value_name,vector<vector<vector<float>>>& value_3d)
{
    NcFile dataFile(file_path_, NcFile::FileMode::read, NcFile::FileFormat::classic);
    auto data = dataFile.getVar(value_name);
    
    if (data.isNull()){
        std::cerr << "The variable " << value_name << " is null." << std::endl;
        return -1;
    }

    try {

        //std::vector<float> flatData(dim0_ * dim1_ * dim2_);
        data.getVar(value_3d[0][0].data());

    } catch(const std::exception& e) {
        std::cerr << "Exception: " << e.what() << '\n';
        return -1;
    }
    return 0;
}

int NetCDFReader::Nc4dReader(const string &value_name,vector<vector<vector<vector<float>>>>& value_4d)
{
    using namespace std;
    NcFile dataFile(file_path_, NcFile::FileMode::read, NcFile::FileFormat::classic);
    auto data = dataFile.getVar(value_name);
    
    if (data.isNull()){
        std::cerr << "The variable " << value_name << " is null." << std::endl;
        return -1;
    }

    multimap<string, NcDim> dimension_map;
    dimension_map = dataFile.getDims();
        NcVarAtt attribute_offset = data.getAtt("add_offset");
    NcVarAtt attribute_scale = data.getAtt("scale_factor");
    double offset, scale;
    attribute_offset.getValues(&offset);
    attribute_scale.getValues(&scale);

    cout<< "offset : " << offset << endl;
    cout<< "scale : " << scale << endl;

    try {
        using namespace std::chrono;
        auto start = high_resolution_clock::now();
        {

            vector<size_t> startp(4, 0);
            vector<size_t> countp = {1,dim1_,dim2_,dim3_};

            for (size_t time = 0; time < dim0_; ++time) {
                short value_3d_tmp[dim1_][dim2_][dim3_];
                startp[0] = time;

                vector<vector<vector<float>>> value_3d(dim1_,vector<vector<float>>(dim2_,vector<float>(dim3_,0)));

                data.getVar(startp, countp, value_3d_tmp);
                // cout << "Data at time " << time << ", level 0, lat 0, lon 0: " << value_3d_tmp[0][0][0] * scale + offset<< endl;
                // cout << "value_3d[1][0][0] = " << value_3d_tmp[1][0][0] <<endl;
                for (int i = 0; i < dim1_; ++i) {
                    for (int j = 0; j < dim2_; ++j) {
                        for (int k = 0; k < dim3_; ++k) {
                            value_3d[i][j][k] = static_cast<float>(value_3d_tmp[i][j][k] * scale + offset);
                        }
                    }
                }
                value_4d.emplace_back(std::move(value_3d));
            } 
        }
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start).count();
        std::cout << "Elapsed time: " << duration << " milliseconds" << std::endl;
    } catch(const std::exception& e) {
        std::cerr << "Exception: " << e.what() << '\n';
        return -1;
    }

    return 0;
}

int NetCDFReader::bin_select_lat(const vector<float> &nums, int target, bool is_seq)
{
    int low = 0;
    int high = nums.size() - 1;
    int closest_index = -1;
    float min_diff = std::numeric_limits<float>::max();

    while (low <= high) {
        int mid = low + (high - low) / 2;
        float mid_val = nums[mid];
        float diff = std::abs(mid_val - target);

        // 更新最接近的值和索引
        if (diff < min_diff) {
            min_diff = diff;
            closest_index = mid;
        }

        // 由于数组是逆序的，我们需要调整比较的方向
        if (mid_val > target) {
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }

    return closest_index;
#if 0
    int low = 0;
    int high = nums.size() - 1;
    while (low <= high) {
        int mid = (low + high) / 2;
        if (target - 0.125 <= nums[mid] && nums[mid] <= target + 0.125) {
            return mid; 
        } else if (nums[mid] < target) {
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }
    return -1; 
#endif
#if 0
    // 由于nums是倒序的，需要反转迭代器的方向
    auto it = std::lower_bound(nums.rbegin(), nums.rend(), target - 0.125,
        [](float a, float b) { return a < b; }); 
    
    // 计算与target最近的索引
    int index = -1;
    float min_diff = std::numeric_limits<float>::max();
    
    if (it != nums.rend()) {
        // 检查当前找到的元素是否在允许的范围内
        float diff = std::abs(*it - target);
        if (diff < min_diff) {
            min_diff = diff;
            index = std::distance(nums.rbegin(), it);
        }
        
        // 检查前一个元素（如果存在）
        if (it != nums.rbegin()) {
            auto prev_it = std::next(it);
            diff = std::abs(*prev_it - target);
            if (diff < min_diff) {
                min_diff = diff;
                index = std::distance(nums.rbegin(), prev_it);
            }
        }
    }
    
    return index;
#endif
}

int NetCDFReader::bin_select_lon(const vector<float> &nums, int target, bool is_seq)
{
#if 1
    int low = 0;
    int high = nums.size() - 1;
    while (low <= high) {
        int mid = (low + high) / 2;
        if (target - 0.125 <= nums[mid] && nums[mid] <= target + 0.125) {
            return mid; // nums数据是顺序的
        } else if (nums[mid] < target) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    return -1;
#endif
#if 0
    // 使用std::lower_bound查找第一个不小于target - 0.25的元素
    auto it = std::lower_bound(nums.begin(), nums.end(), target - 0.25,
        [](float a, float b) { return a < b; });

    int index = -1;
    float min_diff = std::numeric_limits<float>::max();

    if (it != nums.end()) {
        // 检查当前找到的元素是否在允许的范围内
        float diff = std::abs(*it - target);
        if (diff < min_diff) {
            min_diff = diff;
            index = std::distance(nums.begin(), it);
        }

        // 检查前一个元素（如果存在）
        if (it != nums.begin()) {
            auto prev_it = it - 1;
            diff = std::abs(*prev_it - target);
            if (diff < min_diff) {
                min_diff = diff;
                index = std::distance(nums.begin(), prev_it);
            }
        }
    }

    return index;
#endif    
}

int NetCDFReader::seq_select_lev(const vector<float> &nums, int target, bool is_seq)
{
    int mLevIndex = -1;  // 初始化为-1，表示未找到
    float min_diff = std::numeric_limits<float>::infinity(); 

    for (size_t index = 0; index < nums.size(); ++index) {
        float value = nums[index];
        float diff = std::abs(value - target);  // 计算当前值与mLev的差异
        if (diff < min_diff) {
            min_diff = diff;
            mLevIndex = index;
        }
    }

    if (mLevIndex == -1) {
        mLevIndex = nums.size() - 1;  // 如果没有找到，可以选择最后一个索引或其他逻辑
    }

    return mLevIndex;
}

int NetCDFReader::seq_select_time(const vector<float> &nums, int target, bool is_seq)
{
    return 0;
}
