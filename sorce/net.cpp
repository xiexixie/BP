#include "net.h"
#include <random>
// #include <iostream>

std::vector<Sample> BPNet::get_train_data(const std::string &filename)
{
  std::vector<double> buffer = get_data(filename);
  std::vector<Sample> res;
  for (int i = 0; i < buffer.size(); i += IN_NODE + OUT_NODE)
  {
    Sample temp;
    for (int j = 0; j < IN_NODE; j++)
      temp.input.push_back(buffer[i + j]);
    for (int j = 0; j < OUT_NODE; j++)
      temp.output.push_back(buffer[i + IN_NODE + j]);
    res.push_back(temp);
  }
  return res;
}

std::vector<Sample> BPNet::get_test_data(const std::string &filename)
{
  std::vector<double> buffer = get_data(filename);
  std::vector<Sample> res;
  for (int i = 0; i < buffer.size(); i += IN_NODE)
  {
    Sample temp;
    for (int j = 0; j < IN_NODE; j++)
      temp.input.push_back(buffer[i + j]);
    res.push_back(temp);
  }
  return res;
}

BPNet::BPNet(size_t o_in_node, size_t o_hide_node, size_t o_out_node, size_t o_max_times, double o_thereshoid, double o_rate)
{
  in_node = o_in_node;
  hide_node = o_hide_node;
  out_node = o_out_node;
  max_times = o_max_times;
  thereshoid = o_thereshoid;
  rate = o_rate;
  if (in_node > MAX || hide_node > MAX || out_node > MAX)
  {
    std::cout << "节点数超出范围" << std::endl;
    exit(EXIT_FAILURE);
  }
  // 生成随机值
  std::mt19937 rd;
  rd.seed(std::random_device()());
  std::uniform_real_distribution<double> dis(-1, 1);
  // std::random_device seed;  // 将用于获得随机数引擎的种子
  // std::mt19937 gen(seed()); // 以 rd() 播种的标准 mersenne_twister_engine
  // std::uniform_real_distribution<> dis(-1, 1);
  // 输入层初始化
  for (int i = 0; i < IN_NODE; i++)
  {
    input_layer[i] = new Node();
    for (int j = 0; j < HIDE_NODE; j++)
    {
      input_layer[i]->weight.push_back(dis(rd));
      input_layer[i]->weight_delta.push_back(0.f);
    }
  }
  // 隐藏层初始化
  for (int i = 0; i < HIDE_NODE; i++)
  {
    hidden_layer[i] = new Node();
    hidden_layer[i]->bias = dis(rd);
    for (int j = 0; j < OUT_NODE; j++)
    {
      hidden_layer[i]->weight.push_back(dis(rd));
      hidden_layer[i]->weight_delta.push_back(0.f);
    }
  }
  // 输出层初始化
  for (int i = 0; i < OUT_NODE; i++)
  {
    output_layer[i] = new Node();
    output_layer[i]->bias = dis(rd);
  }
}
BPNet::~BPNet()
{
  delete[] input_layer;
  delete[] hidden_layer;
  delete[] output_layer;
}
