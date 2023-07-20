#include "tools.h"

std::vector<double> utilities::get_data(const std::string &filename)
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
  {
    std::cout << "Error:open file failed!" << std::endl;
    exit(EXIT_FAILURE);
  }
  return res;
}

double utilities::sigmoid(double x)
{
  return 1.0 / (1.0 + std::exp(-x));
}