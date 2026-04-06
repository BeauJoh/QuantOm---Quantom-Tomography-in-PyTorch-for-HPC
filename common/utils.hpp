#pragma once

#define KBLK  "\033[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define RESET "\x1b[m"
#define FFLUSH(X)

#define  print_green(content, ...) do { printf( KGRN content RESET ); FFLUSH(stdout); } while (0)
#define  print_magenta(content, ...) do { printf( KMAG content RESET ); FFLUSH(stdout); } while (0)
#define  print_blue(content, ...) do { printf( KBLU content RESET ); FFLUSH(stdout); } while (0)
#define  print_cyan(content, ...) do { printf( KCYN content RESET ); FFLUSH(stdout); } while (0)
#define  print_yellow(content, ...) do { printf( KYEL content RESET ); FFLUSH(stdout); } while (0)
#define  print_red(content, ...) do { printf( KRED content RESET ); FFLUSH(stdout); } while (0)

#include <limits>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iostream>
#include <string>
#include <cassert>
#include <csignal>

#if defined(INSTRUMENTATION) && defined(VALGRIND)
  #include <valgrind/callgrind.h>
#else
  #define CALLGRIND_START_INSTRUMENTATION ;
  #define CALLGRIND_STOP_INSTRUMENTATION ;
#endif

#include "matrix.hpp"

template<typename T>
static bool are_equal(T const f1, T const f2) {
  return (std::fabs(f1 - f2) <= 0.01 * std::fmax(std::fabs(f1), std::fabs(f2)));
}

template<typename T>
static bool are_equal(std::vector<T>const& f1, std::vector<T>const& f2) {
  if (f1.size() != f2.size()) return false;
  for (size_t i = 0; i < f1.size(); i++){
    if (std::fabs((float)f1[i] - (float)f2[i]) > 1E-05){//std::numeric_limits<T>::epsilon()) {
      std::cout << "difference @ " << i << " = " << std::fabs((float)f1[i] - (float)f2[i]) << std::endl;
      std::cout << "epsilon is " << std::numeric_limits<T>::epsilon() << std::endl;
      std::cout << "f1 = " << f1[i] << " f2 = " << f2[i] << std::endl;
      raise(SIGINT);
      return false;}
  }
  return true;
}

template<typename T>
static bool are_equal(matrix<T>const& f1, matrix<T>const& f2) {
  return are_equal(f1.as_vector(), f2.as_vector());
}

template <typename T>
T max_diff(matrix<T>const& x, matrix<T>const& y){
  assert(x.shape() == y.shape());
  std::vector<T> diff;
  std::set_difference(x.as_vector().begin(), x.as_vector().end(), y.as_vector().begin(), y.as_vector().end(),
        std::inserter(diff, diff.begin()));
  if (diff.empty()){
    return 0;
  }
  T res = *std::max_element(diff.begin(),diff.end());
  return(res);
}

//timing and logging functions:
void tic(std::string,int mode=0);
void toc(std::string);
void enable_recording();
void disable_recording();
void mark_for_recording(std::string roi);
void assign_filename_for_recording(std::string fn);
void print_total_measured_time();
void reset_total_measured_time();

