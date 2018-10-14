#include <torch/torch.h>
#include <iostream>
#include <utility>

using LabeledDataSet = std::pair<at::Tensor, at::Tensor>;

LabeledDataSet get_data()
{
  auto X = torch::tensor({3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,
                          7.59,2.167, 7.042,10.791,5.313,7.997,5.654,9.27,
                          3.1}, torch::kFloat32);
  auto y = torch::tensor({1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,
                          2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,
                          1.3}, torch::kFloat32);
  return std::make_pair(X, y);
}





int main(){
  std::cout << "Chapter 02\n";
  auto train_data_set = get_data();
}

