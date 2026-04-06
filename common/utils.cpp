
#include <map>
#include <fstream>
#include <execinfo.h>//For collecting the backtrace
#include <cxxabi.h> // For demangling C++ function names

#include "utils.hpp"

static std::map<std::string, std::chrono::_V2::system_clock::time_point> timings;
static std::vector<std::string> record_list;
static std::string filename = "times.csv";
static std::ofstream file_output_stream;
static float elapsed_measured_secs = 0.0f;
static bool recording = true;

void tic(std::string region_name, int mode){
  if (!recording) return;
  if (mode==0){
    auto t_start = std::chrono::high_resolution_clock::now();
    timings[region_name] = t_start;
  }
  else {
    auto t_end = std::chrono::high_resolution_clock::now();
    elapsed_measured_secs += (t_end-timings[region_name]).count()*1E-9;
    std::cout << region_name << " : " << (t_end-timings[region_name]).count()*1E-9 << " seconds" << std::endl;
    //and record it
    bool is_present = (std::find(record_list.begin(), record_list.end(), region_name) != record_list.end());
    if (is_present) {
      if (!file_output_stream.is_open()) file_output_stream.open(filename.c_str(),std::ios::app);
      file_output_stream << region_name << ", " << (t_end-timings[region_name]).count()*1E-9 << std::endl;
    }
  }
}

void disable_recording(){
  recording = false;
}

void enable_recording(){
  recording = true;
}

void toc(std::string region_name){
  tic(region_name,1);
}

void mark_for_recording(std::string roi){
  record_list.push_back(roi);
}

void assign_filename_for_recording(std::string fn){
  filename = fn;
}

void print_total_measured_time(){
  std::cout << "Total Measured Elapsed Time (tracked by tic-toc): " << elapsed_measured_secs << std::endl;
} 

void reset_total_measured_time(){
  elapsed_measured_secs = 0.0f;
}

