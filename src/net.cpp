#include "net.h"
#include "timer.h"

#include <random>
// #include <iostream>

std::vector<Sample> BPNet::get_train_data(const std::string &filename)
{
  std::vector<double> buffer = utilities::get_data(filename);
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
  std::vector<double> buffer = utilities::get_data(filename);
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
  count = 0;
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
  for (int i = 0; i < IN_NODE; i++)
    delete input_layer[i];
  for (int i = 0; i < HIDE_NODE; i++)
    delete hidden_layer[i];
  for (int i = 0; i < OUT_NODE; i++)
    delete output_layer[i];
  count = 0;
}

void BPNet::delta_reset()
{
  // 输入层重置
  for (int i = 0; i < IN_NODE; i++)
  {
    input_layer[i]->weight_delta.assign(input_layer[i]->weight_delta.size(), 0.f);
  }
  // 隐藏层重置
  for (int i = 0; i < HIDE_NODE; i++)
  {
    hidden_layer[i]->bias_delta = 0.f;
    hidden_layer[i]->weight_delta.assign(hidden_layer[i]->weight_delta.size(), 0.f);
  }
  // 输出层重置
  for (int i = 0; i < OUT_NODE; i++)
  {
    hidden_layer[i]->bias_delta = 0.f;
  }
}

void BPNet::train(const std::string filename)
{
  Timer timer; // 计时器
  std::vector<Sample> train_data = get_train_data(filename);
  for (size_t time = 0; time < max_times; time++)
  {
    count++;
    delta_reset();

    double err = 0.f;
    for (auto &idx : train_data)
    {
      // 正向传播
      forward_propagation(idx);
      // 计算误差
      err = cal_err(err, idx);
      // 反向传播
      back_propagation(idx);
    }

    if (err < thereshoid)
    {
      std::cout << "train success with " << time + 1 << " times !" << std::endl;
      break;
    }

    // 更新各个节点
    updata(train_data);
  }

  if (count >= max_times)
  {
    std::cout << "train failed!\n"
              << count;
    exit(EXIT_FAILURE);
  }
}

// 前向传播
void BPNet::forward_propagation(Sample &idx)
{
  // 读取数据
  for (int i = 0; i < IN_NODE; i++)
    input_layer[i]->value = idx.input[i];
  // 求隐藏层的value
  for (int i = 0; i < HIDE_NODE; i++)
  {
    double sum = 0.f;
    for (int j = 0; j < IN_NODE; j++)
    {
      sum += input_layer[j]->value * input_layer[j]->weight[i];
    }
    sum -= hidden_layer[i]->bias;
    hidden_layer[i]->value = utilities::sigmoid(sum);
  }
  // 求输出层value
  for (int i = 0; i < OUT_NODE; i++)
  {
    double sum = 0.f;
    for (int j = 0; j < HIDE_NODE; j++)
    {
      sum += hidden_layer[j]->value * hidden_layer[j]->weight[i];
    }
    sum -= output_layer[i]->bias;
    output_layer[i]->value = utilities::sigmoid(sum);
  }
}

// 反向传播
void BPNet::back_propagation(Sample &idx)
{
  // 计算delta
  // out_node bias_delta
  for (int i = 0; i < OUT_NODE; i++)
  {
    output_layer[i]->bias_delta += -(idx.output[i] - output_layer[i]->value) * output_layer[i]->value * (1.0 - output_layer[i]->value);
  }
  // hide_node weight_delta
  for (int i = 0; i < HIDE_NODE; i++)
  {
    for (int j = 0; j < OUT_NODE; j++)
    {
      hidden_layer[i]->weight_delta[j] += (idx.output[j] - output_layer[j]->value) * output_layer[j]->value * (1.0 - output_layer[j]->value) * hidden_layer[i]->value;
    }
  }
  // hide_node bias_delta
  for (int i = 0; i < HIDE_NODE; i++)
  {
    double sum = 0.f;
    for (int j = 0; j < OUT_NODE; j++)
    {
      sum += -(idx.output[j] - output_layer[j]->value) * output_layer[j]->value * (1.0 - output_layer[j]->value) * hidden_layer[i]->weight[j];
    }
    hidden_layer[i]->bias_delta += sum * hidden_layer[i]->value * (1.0 - hidden_layer[i]->value);
  }
  // in_node weight_delta
  for (int i = 0; i < IN_NODE; i++)
  {
    for (int j = 0; j < HIDE_NODE; j++)
    {
      double sum = 0.f;
      for (int k = 0; k < OUT_NODE; k++)
      {
        sum += (idx.output[k] - output_layer[k]->value) * output_layer[k]->value * (1.0 - output_layer[k]->value) * hidden_layer[j]->weight[k];
      }
      input_layer[i]->weight_delta[j] += sum * hidden_layer[j]->value * (1.0 - hidden_layer[j]->value) * input_layer[i]->value;
    }
  }
}

double BPNet::cal_err(double err, Sample &idx)
{
  double loss = 0.f;
  for (int i = 0; i < OUT_NODE; i++)
  {
    double temp = std::abs(output_layer[i]->value - idx.output[i]);
    loss += temp * temp / 2;
  }
  err = std::max(err, loss);
  return err;
}

void BPNet::updata(std::vector<Sample> train_data)
{
  // 更新weight和bias
  auto train_data_size = double(train_data.size());
  // 更新从input_layer到hidden_layer的weight
  for (int i = 0; i < IN_NODE; i++)
  {
    for (int j = 0; j < HIDE_NODE; j++)
    {
      input_layer[i]->weight[j] += (rate * input_layer[i]->weight_delta[j] / train_data_size);
    }
  }
  // 更新hidden_layer的bias和hidden_layer到out_layer的weight
  for (int i = 0; i < HIDE_NODE; i++)
  {
    hidden_layer[i]->bias += (rate * hidden_layer[i]->bias_delta / train_data_size);
    for (int j = 0; j < OUT_NODE; j++)
    {
      hidden_layer[i]->weight[j] += (rate * hidden_layer[i]->weight_delta[j] / train_data_size);
    }
  }
  // 更新out_layer的bias
  for (int i = 0; i < OUT_NODE; i++)
  {
    output_layer[i]->bias += (rate * output_layer[i]->bias_delta / train_data_size);
  }
}

// 预测结果
void BPNet::predict(const std::string filename)
{

  std::vector<Sample> test_data = get_test_data(filename);
  std::cout << "predict result:" << std::endl;

  // predict
  for (auto &idx : test_data)
  {
    // 将数据输入Input_layer
    for (int i = 0; i < IN_NODE; i++)
    {
      input_layer[i]->value = idx.input[i];
    }
    // 计算hidde_node的value
    for (int i = 0; i < HIDE_NODE; i++)
    {
      double sum = 0.f;
      for (int j = 0; j < IN_NODE; j++)
      {
        sum += input_layer[j]->weight[i] * input_layer[j]->value;
      }
      sum -= hidden_layer[i]->bias;
      hidden_layer[i]->value = utilities::sigmoid(sum);
    }
    // 计算out_node的value
    for (int i = 0; i < OUT_NODE; i++)
    {
      double sum = 0.f;
      for (int j = 0; j < HIDE_NODE; j++)
      {
        sum += hidden_layer[j]->weight[i] * hidden_layer[j]->value;
      }
      sum -= output_layer[i]->bias;
      output_layer[i]->value = utilities::sigmoid(sum);
      idx.output.push_back((output_layer[i]->value > 0.5) ? 1 : 0);
      for (auto &temp : idx.input)
      {
        std::cout << temp << " ";
      }
      for (auto &temp : idx.output)
      {
        std::cout << temp << " ";
      }
      std::cout << std::endl;
    }
  }
}