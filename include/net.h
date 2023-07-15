#pragma once
#include "tools.h"

#define MAX 100 // 每层最多包含的节点数

class BPNet
{
public:
#define IN_NODE in_node
#define HIDE_NODE hide_node
#define OUT_NODE out_node
  std::vector<Sample> get_train_data(const std::string &filename);
  std::vector<Sample> get_test_data(const std::string &filename);
  BPNet(size_t o_in_node, size_t o_hide_node, size_t o_out_node, size_t o_max_times, double o_thereshoid, double o_rate);
  ~BPNet();

private:
  double rate;       // 学习率，步长
  double thereshoid; // 最大误差
  size_t max_times;  // 最大迭代次数
  size_t in_node;    // 输入层节点
  size_t hide_node;  // 隐藏层节点
  size_t out_node;   // 输出层节点
  Node *input_layer[MAX], *output_layer[MAX], *hidden_layer[MAX];
};