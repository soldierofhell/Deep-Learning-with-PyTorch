#include <torch/torch.h>
#include <iostream>
#include <utility>
#include "matplotlibcpp.h"

using LabeledDataSet = std::pair<at::Tensor, at::Tensor>;

struct LinearWeights
{
  at::Tensor w = torch::randn({1}, torch::requires_grad(true));
  at::Tensor b = torch::randn({1}, torch::requires_grad(true));
};

LabeledDataSet get_data()
{
  auto X = torch::tensor({3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,
                          7.59,2.167, 7.042,10.791,5.313,7.997,5.654,9.27,
                          3.1}, 
    torch::requires_grad(false).dtype(torch::kFloat32))
    .view({17,1});
  auto y = torch::tensor({1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,
                          2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,
                          1.3}, 
    torch::requires_grad(false).dtype(torch::kFloat32));

  return std::make_pair(X, y);
}

std::vector<float> tensor_to_vector(const torch::Tensor& x)
{
  return std::vector<float>(x.data<float>(), x.data<float>() + x.numel());
}

LinearWeights get_weights()
{
  return LinearWeights{};
}

at::Tensor simple_network(const torch::Tensor& x, const LinearWeights& lw)
{
  return torch::matmul(x, lw.w)+lw.b;
}

float loss_fn(const torch::Tensor& y, const torch::Tensor& y_pred, 
              LinearWeights& lw)
{
  auto loss = (y_pred-y).pow(2).sum();
  if(lw.w.grad().defined()) lw.w.grad().zero_();
  if(lw.b.grad().defined()) lw.b.grad().zero_();
  loss.backward();
  return static_cast<float*>(loss.storage().data())[0];
}

void optimize(const float lr, LinearWeights& lw)
{
  torch::NoGradGuard g;
  lw.b -= lr * lw.b.grad();
  lw.w -= lr * lw.w.grad();
} 

int main(){
  std::cout << "Chapter 02\n";
  const float learning_rate = 1e-4;
  auto train_data_set = get_data();
  auto x = train_data_set.first;
  auto y = train_data_set.second;

  auto weights = get_weights();

  auto y_pred = simple_network(x, weights);
  for(auto i=0; i<500; i++)
  {
    y_pred = simple_network(x, weights);
    auto loss = loss_fn(y, y_pred, weights);
    if((i%50)==0) std::cout << loss << '\n';
    optimize(learning_rate, weights);
  }

  namespace plt = matplotlibcpp;
  plt::plot(tensor_to_vector(x), tensor_to_vector(y), "ro",
            tensor_to_vector(x), tensor_to_vector(y_pred), "b+");
  plt::show();


}

