#pragma once
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

// 样本结构
struct Sample
{
  std::vector<double> input, output;
};
/*
神经元节点结构
神经元值 value
偏置值 bias
权值 weight
*/
struct Node
{
  double value{}, bias{}, bias_delta{};
  std::vector<double> weight, weight_delta;
};
namespace utilities
{
  // 激活函数
  double sigmoid(double x);

  std::vector<double> get_data(const std::string &filename);

}
