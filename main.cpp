#include "net.h"

int main()
{
  BPNet bp(2, 4, 1, 1e6, 1e-4, 0.8);
  bp.train("../train_data.txt");
  bp.predict("../test_data.txt");
}