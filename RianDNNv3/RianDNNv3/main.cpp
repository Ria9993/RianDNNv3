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
	//std::vector<float> target = { 1.0f, -1.0f, 3.1415f };
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

			/* cout << "input : " << input[0] << ", result : ";
			for (float itr : model.GetResult())
			{
				cout << itr << ", ";
			}
			cout << std::endl;*/
			result_list[i % 100] = (model.GetResult()[0] / target[0]) * 100;
			//cout << (model.GetResult()[0] / target[0]) * 100 << '%' << endl;
		}
		if (i % 100 == 0)
		{
			if (i % 1000 == 0)
			{
				double sum = 0;
				for (int j = 0; j < 100; j++)
					sum += result_list[j];
				cout.precision(5);
				cout << fixed;
				cout << "¿ÀÂ÷À² : " << setw(10) <<  100 - (sum / 100) << '%' << endl;
			}
			model.Optimize();
		}
	}

	//timeStart();
	//model.Forward();
	//timeEnd();
	//timePrint();

	//for (float itr : model.GetResult())
	//{
	//	cout << itr;
	//}

	return 0;	
}