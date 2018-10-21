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
  auto input2 = torch::randn({3, 5}, torch::requires_grad(true));
  auto target2 = torch::tensor({3},torch::kLong).random_(5);
  auto output2 = torch::binary_cross_entropy(input2, target2);
  // output2.backward();
  // std::cout << output << '\n';

// torch::log_softmax 
// torch::nll_loss

}
