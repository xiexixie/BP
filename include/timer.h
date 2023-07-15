#pragma once
#include <chrono>
#include <iostream>

// 计时器
struct Timer
{
  std::chrono::high_resolution_clock::time_point start, end;
  std::chrono::duration<float> duration;
  Timer()
  {
    start = std::chrono::high_resolution_clock::now();
  }
  ~Timer()
  {
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    float s = duration.count();
    std::cout << "runing time: " << s << "s" << std::endl;
  }
};