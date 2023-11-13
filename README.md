# RianDNNv3
Just simple neural-network library  
### blog post  
<https://ria9993.github.io/cs/2022/10/14/intro-neural-network-learning.html>
## Feature Support
- **GPGPU**  
 (By default it is disabled, to enable it you need to define "GPGPU" in the RianDNN.h file)  
 (CAUTION : GPGPU doesn't work in convolution layers)  
- **Multi-Thread**  
 (By default it is enabled)
- **Layer**
  - Dense
  - Convolution 1D
  - Transposed Convolution 1D  
## Preview AutoEncoder Test
https://github.com/Ria9993/RianDNNv3/assets/44316628/2c675696-bb14-4d1d-8a11-3372584cb6ce  
## Exmaple
### Learning Sigmoid(x)
```cpp
#include"RianDNN.h"

#include <vector>
#include <random>

#include <iostream>
#include <chrono>
#include <iomanip>
using namespace std;
using namespace chrono;

double sigmoid_1(double z) {
	return (1 / (1 + std::exp(-z)));
}

int main()
{
	rian::HyperParm hyperParm;
	hyperParm.MomentumRate = 0.99f;
	hyperParm.LearningRate = 0.01f;

	rian::Model model(hyperParm);
	model.AddLayer(2, rian::Activation::None);
	model.AddLayer(128, rian::Activation::ReLU);
	model.AddLayer(128, rian::Activation::ReLU);
	model.AddLayer(128, rian::Activation::ReLU);
	model.AddLayer(2, rian::Activation::None);

	model.Build();

	std::vector<float>& input = model.GetInputVector();

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> rand(-1.0f, 1.0f);

	const int BATCH_SIZE = 100;
	for (int i = 0; true; i++)
	{
		std::vector<float> target(2);
		static double result_list[100] = { 0, };
		static double error_list[100000] = { 0, };
		{
			input[0] = rand(gen);
			input[1] = rand(gen);
			model.Forward();
			
			target[0] = sigmoid_1(input[0]);
			target[1] = sigmoid_1(input[1]);
			model.ComputeError(target);

			result_list[i % 100] = (model.GetResult()[0] / target[0]) * 100;
			//cout << (model.GetResult()[0] / target[0]) * 100 << '%' << endl;
		}
		if (i % BATCH_SIZE == 0)
		{
			if (true)
			{
				double sum = 0;
				for (int j = 0; j < 100; j++)
					sum += result_list[j];
				cout.precision(5);
				cout << fixed;
				cout << "Error_rate : " << setw(10) << (100 - (sum / 100)) << '%' << endl;

				// DEBUG
				error_list[i / BATCH_SIZE] = (100 - (sum / 100));

				cout << "input[0] : " << input[0] << ", target : " << target[0] << ", result : " << model.GetResult()[0] << ", error = " << target[0] - model.GetResult()[0] << endl;
				cout << "input[1] : " << input[1] << ", target : " << target[1] << ", result : " << model.GetResult()[1] << ", error = " << target[1] - model.GetResult()[1] << endl;
				
				cout << std::endl;
			}
			model.Optimize();
		}
	}

	return 0;
}
```
