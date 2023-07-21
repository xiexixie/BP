#include "net.h"

int main()
{
  BPNet bp(2, 4, 1, 200000, 1e-3, 0.6);
  bp.train("../train_data.txt");
  bp.predict("../test_data.txt");
}