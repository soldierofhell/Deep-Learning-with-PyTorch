#include <iostream>
#include <torch/torch.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/functional.h>

void simple_layer_manipulation()
{
  std::cout << "Layers : Fundamental blocks of Neural Network\n";
  auto myLayer = torch::nn::Linear(10, 5);
  auto inp = torch::randn({1, 10});
  std::cout << myLayer->forward(inp) << '\n';
  std::cout << myLayer->weight << '\n';
  std::cout << myLayer->bias << '\n';

  std::cout << "Stacking Linear layers\n";
  auto myLayer1 = torch::nn::Linear(10, 5);
  auto myLayer2 = torch::nn::Linear(5, 2);
  std::cout << myLayer2->forward(myLayer1->forward(inp));

  std::cout << "PyTorch Non-linear Activations\n";
  auto sample_data = torch::tensor({1, 2, -1, -1});
  auto myRelu = torch::nn::Functional(torch::relu);
  std::cout << myRelu->forward(sample_data) << '\n';
  // simpler
  std::cout << torch::relu(sample_data) << '\n';

}

struct MyFirstNetwork : torch::nn::Module
{
  using Linear = torch::nn::Linear;
  MyFirstNetwork(size_t input_size, size_t hidden_size, size_t output_size) :
    layer1(register_module("layer1", Linear(input_size, hidden_size))),
  layer2(register_module("layer2", Linear(hidden_size, output_size))) {};

  //how to define the architechture?
  Linear layer1;
  Linear layer2;
};

int main()
{
  std::cout << "Chapter 03\n";
  simple_layer_manipulation();
  std::cout << "Neural Network\n";
  MyFirstNetwork n(10, 20, 5);
  auto inp = torch::randn({1, 10});

  // how to train? 
  n.eval();
  n.train();


  auto input = torch::randn({3, 5}, torch::requires_grad(true));
  auto target = torch::randn({3, 5});
  auto output = torch::mse_loss(input, target);
  output.backward();
  std::cout << output << '\n';

  // https://discuss.pytorch.org/t/c-loss-functions/27471/6
  /* In the C++ API we don’t have a cross_entropy_loss per se, but as @ptrblck
   * mentioned, nn.CrossEntropyLoss()(input, target) is the same as
   * cross_entropy(input, target), which per
   * https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L1671
   * 1 is the same as nll_loss(log_softmax(input, 1), target), so your
   * code in C++ would be something like*/

  auto input2 = torch::randn({3, 5}, torch::requires_grad(true));
  auto target2 = torch::empty(3, torch::kLong).random_(5);
  auto output2 = torch::nll_loss(torch::log_softmax(input2, /*dim=*/1), target2);
  output2.backward();
  std::cout << output2 << '\n';

}
