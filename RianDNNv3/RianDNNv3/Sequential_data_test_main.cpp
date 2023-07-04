#if 0
//#define ONLY_FORWARD
#include"RianDNN.h"

#include <vector>
#include <random>

#include <iostream>
#include <chrono>
#include <iomanip>
using namespace std;
using namespace chrono;

system_clock::time_point startTime;
system_clock::time_point endTime;
__forceinline void timeStart()
{
	startTime = system_clock::now();
}
__forceinline void timeEnd()
{
	endTime = system_clock::now();
}
void timePrint()
{
	cout << static_cast<nanoseconds>(endTime - startTime).count() << "ns" << endl;
}

double sigmoid_1(double z) {
	return (1 / (1 + std::exp(-z)));
}

int main()
{
	rian::HyperParm hyperParm;
	hyperParm.MomentumRate = 0.90f;
	hyperParm.LearningRate = 0.1E-3f;

	rian::Model model(hyperParm);
	model.AddLayer(20, rian::Activation::None);
	model.AddLayer(100, rian::Activation::LeakyReLU);
	model.AddLayer(100, rian::Activation::LeakyReLU);
	model.AddLayer(100, rian::Activation::LeakyReLU);
	model.AddLayer(100, rian::Activation::LeakyReLU);
	model.AddLayer(1, rian::Activation::None);

	std::vector<float>& input = model.GetInputVector();
	//std::vector<float> target = { 1.0f, -1.0f, 3.1415f };
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> rand(0.0f, 100.0f);
	std::uniform_real_distribution<float> rand2(-10.0f, 10.0f);

	std::uniform_int_distribution<int> rand_idx(0, 9980 - 1);
	static float table[10000];
	for (int i = 0; i < 10000; i++)
		table[i] = rand2(gen);

	for (int i = 0; i < 100000000; i++)
	{
		static double result_list[10000] = { 0, };
		{
			const int sample_idx = rand_idx(gen);
			for (int z = 0; z < 20; z++)
				input[z] = table[sample_idx + z];

			model.Forward();

			std::vector<float> target(1);
			target[0] = table[sample_idx + 20];
			model.ComputeError(target);

			/* cout << "input : " << input[0] << ", result : ";
			for (float itr : model.GetResult())
			{
				cout << itr << ", ";
			}
			cout << std::endl;*/
			result_list[i % 10000] = (model.GetResult()[0] - target[0]);
		}
		if (i % 1000 == 0)
		{
			model.Optimize();

			if (i % 10000 == 0)
			{
				float sum = 0;
				for (int i = 0; i < 10000; i++)
					sum += result_list[i];

				cout.precision(5);
				cout << fixed;
				cout << setw(12) << (sum / 10000) << '.' << endl;
				static double vec[100000];
				vec[i / 10000] = sum;
			}
		}
	}

	return 0;
}
#endif