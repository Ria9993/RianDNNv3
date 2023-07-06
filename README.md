# RianDNNv3
Just simple neural-network library  
### blog post  
<https://ria9993.github.io/cs/2022/10/14/intro-neural-network-learning.html>
## Exmaple
### Learning Sigmoid(x)
```cpp
//#define ONLY_FORWARD
#include"RianDNN.h"

#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
using namespace std;

double sigmoid_1(double z) {
	return (1 / (1 + std::exp(-z)));
}

int main()
{
	rian::HyperParm hyperParm;
	hyperParm.MomentumRate = 0.9f;
	hyperParm.LearningRate = 0.1E-4f;

	rian::Model model(hyperParm);
	model.AddLayer(1, rian::Activation::None);
	model.AddLayer(100, rian::Activation::LeakyReLU);
	model.AddLayer(100, rian::Activation::LeakyReLU);
	model.AddLayer(100, rian::Activation::LeakyReLU);
	model.AddLayer(100, rian::Activation::LeakyReLU);
	model.AddLayer(100, rian::Activation::LeakyReLU);
	model.AddLayer(1, rian::Activation::None);

	std::vector<float>& input = model.GetInputVector();
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> rand(0.0f, 100.0f);

	for (int i = 0; i < 10000000; i++)
	{
		static double result_list[100] = { 0, };
		{
			input[0] = rand(gen);
			model.Forward();

			std::vector<float> target(1);
			target[0] = sigmoid_1(input[0]);
			model.ComputeError(target);

			result_list[i % 100] = (model.GetResult()[0] / target[0]) * 100;
		}
		if (i % 100 == 0)
		{
			model.Optimize();
			if (i % 1000 == 0)
			{
				double sum = 0;
				for (int j = 0; j < 100; j++)
					sum += result_list[j];
				cout.precision(5);
				cout << fixed;
				cout << "Error_rate : " << setw(10) << 100 - (sum / 100) << '%' << endl;
			}
		}
	}
	return 0;
}
```
