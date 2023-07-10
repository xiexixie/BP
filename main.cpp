#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <random>
#include <chrono>

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

#define IN_NODE 2
#define HIDE_NODE 4
#define OUT_NODE 1
double rate = 0.8;        // 学习率，步长
double thereshoid = 1e-4; // 最大误差
size_t max_times = 1e6;   // 最大迭代次数
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

// 激活函数
double sigmoid(double x)
{
  return 1.0 / (1.0 + std::exp(-x));
}

std::vector<double> get_data(const std::string &filename)
{
  std::vector<double> res;
  std::ifstream ifs(filename);
  if (ifs.is_open())
  {
    while (!ifs.eof())
    {
      double buffer;
      ifs >> buffer;
      res.push_back(buffer);
    }
    ifs.close();
  }
  else
    std::cout << "Error:open file failed!" << std::endl;
  return res;
}

std::vector<Sample> get_train_data(const std::string &filename)
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

std::vector<Sample> get_test_data(const std::string &filename)
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

Node *input_layer[IN_NODE], *output_layer[OUT_NODE], *hidden_layer[HIDE_NODE];
void init()
{
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
void delta_reset()
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

int main()
{
  Timer timer; // 计时器
  init();
  int count = 0;
  std::vector<Sample> train_data = get_train_data("train_data.txt");
  // train
  for (size_t time = 0; time < max_times; time++)
  {
    count++;
    delta_reset();

    double err = 0.f;
    for (auto &idx : train_data)
    {
      // 输入层初始化
      for (int i = 0; i < IN_NODE; i++)
        input_layer[i]->value = idx.input[i];

      // 正向传播
      // 求隐藏层的value
      for (int i = 0; i < HIDE_NODE; i++)
      {
        double sum = 0.f;
        for (int j = 0; j < IN_NODE; j++)
        {
          sum += input_layer[j]->value * input_layer[j]->weight[i];
        }
        sum -= hidden_layer[i]->bias;
        hidden_layer[i]->value = sigmoid(sum);
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
        output_layer[i]->value = sigmoid(sum);
      }

      // 计算误差
      double loss = 0.f;
      for (int i = 0; i < OUT_NODE; i++)
      {
        double temp = std::abs(output_layer[i]->value - idx.output[i]);
        loss += temp * temp / 2;
      }
      err = std::max(err, loss);

      // 反向传播
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

    if (err < thereshoid)
    {
      std::cout << "train success with " << time + 1 << " times !" << std::endl;
      break;
    }

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
  if (count >= max_times)
  {
    std::cout << "train failed!\n";
    return 0;
  }
  std::vector<Sample> test_data = get_test_data("test_data.txt");
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
      hidden_layer[i]->value = sigmoid(sum);
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
      output_layer[i]->value = sigmoid(sum);
      idx.output.push_back(output_layer[i]->value);
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

  return 0;
}