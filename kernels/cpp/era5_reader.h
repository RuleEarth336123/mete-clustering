#ifndef ERA5_H
#define ERA5_H

#include <iostream>
#include <string>
#include <netcdf>
#include <vector>
#include <unordered_map>
#include <tuple>

using std::vector;
using std::string;

class NetCDFReader{
    
public:
    explicit NetCDFReader(const string& file_path,int dim0,int dim1,int dim2,int dim3) 
        : file_path_(file_path),dim0_(dim0),dim1_(dim1),dim2_(dim2),dim3_(dim3){
        }
    explicit NetCDFReader(const string& file_path,int dim0,int dim1,int dim2) 
        : file_path_(file_path),dim0_(dim0),dim1_(dim1),dim2_(dim2){}
    explicit NetCDFReader(const string& file_path,int dim0,int dim1) 
        : file_path_(file_path),dim0_(dim0),dim1_(dim1){}
    explicit NetCDFReader(const string& file_path,int dim0) : file_path_(file_path),dim0_(dim0){}
    ~NetCDFReader() = default;

    void InitIndex();
    std::tuple<int,int,int,int,bool> getIndex(int time_index,float level,float latitude,float longitude);
    float getVar(std::tuple<int,int,int,int>& index,string varName = "u");
    

    int Nc1dReader(const string& value_name,vector<float>& values_1d);
    int Nc2dReader(const string& value_name,vector<vector<float>>& values_2d);

    int Nc3dReader(const string &value_name,vector<vector<vector<float>>>& value_3d);
    int Nc4dReader(const string &value_name,vector<vector<vector<vector<float>>>>& value_4d);

    
private:
    int bin_select_lat(const vector<float>& nums,float target,bool is_seq = true);
    int bin_select_lon(const vector<float>& nums,float target,bool is_seq = true);
    int seq_select_lev(const vector<float>& nums,float target,bool is_seq = true);
    int seq_select_time(const vector<float>& nums,int target,bool is_seq = true);

private:
    string file_path_;
    size_t dim0_ = 0;
    size_t dim1_ = 0;
    size_t dim2_ = 0;
    size_t dim3_ = 0;

private:    
    vector<vector<vector<vector<float>>>> u_4d_,v_4d_,w_4d_;

protected:
    std::unordered_map<string,vector<float>> index_map;

};

#endif