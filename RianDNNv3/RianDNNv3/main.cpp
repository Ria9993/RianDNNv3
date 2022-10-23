//#define ONLY_FORWARD
#include"RianDNN.h"

#include <vector>

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

	rian::Model model(hyperParm);
	model.AddLayer(2, rian::None);
	model.AddLayer(1000, rian::ReLU);
	model.AddLayer(1000, rian::ReLU);
	model.AddLayer(2, rian::None);

	std::vector<float>& input = model.GetInputVector();
	input[0] = 5.0f;
	input[1] = 2.5f;
	
	timeStart();
	model.Forward();
	timeEnd();
	timePrint();

	for (float itr : model.GetResult())
	{
		cout << itr;
	}

	return 0;	
}