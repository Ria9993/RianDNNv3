# RianDNNv3
Just simple neural-network library
## Exmaple
### Learning f(x) = 2x
```cpp
#include"RianDNN.h"
#include <vector>
#include <random>

int main()
{
	rian::HyperParm hyperParm;
	hyperParm.MomentumRate = 0.9f;
	hyperParm.LearningRate = 0.1E-5;

	rian::Model model(hyperParm);
	model.AddLayer(1, rian::Activation::None);
	model.AddLayer(10, rian::Activation::LeakyReLU);
	model.AddLayer(10, rian::Activation::LeakyReLU);
	model.AddLayer(1, rian::Activation::None);

	std::vector<float>& input = model.GetInputVector();
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> rand(0.0f, 10.0f);
	for (int i = 0; i < 1000; i++)
	{
		{
			input[0] = rand(gen);
			model.Forward();

			std::vector<float> target(1);
			target[0] = input[0] * 2;
			model.ComputeError(target);

			std::cout << "input : " << input[0] << ", result : ";
			for (float itr : model.GetResult())
			{
				std::cout << itr << ", ";
			}
			std::cout << std::endl;
		}
		if (i % 50 == 0)
			model.Optimize();
	}
	return 0;	
}
```
