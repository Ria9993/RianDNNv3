//#define ONLY_FORWARD
#include"RianDNN.h"

#include <vector>
#include <random>

#include <iostream>
#include <chrono>
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
int main()
{
	rian::HyperParm hyperParm;
	hyperParm.MomentumRate = 0.9f;
	hyperParm.LearningRate = 0.1E-5f;

	rian::Model model(hyperParm);
	model.AddLayer(1, rian::Activation::None);
	model.AddLayer(10, rian::Activation::LeakyReLU);
	model.AddLayer(10, rian::Activation::LeakyReLU);
	model.AddLayer(1, rian::Activation::None);

	std::vector<float>& input = model.GetInputVector();
	//std::vector<float> target = { 1.0f, -1.0f, 3.1415f };
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

			cout << "input : " << input[0] << ", result : ";
			for (float itr : model.GetResult())
			{
				cout << itr << ", ";
			}
			cout << endl;
		}
		if (i % 50 == 0)
			model.Optimize();
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