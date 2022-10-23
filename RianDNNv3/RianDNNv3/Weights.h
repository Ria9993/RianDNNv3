#pragma once

#include <vector>
#include <random>
#include <cmath>

class Weights
{
public:
	Weights(int srcSize, int destSize)
	{
		// weight
		v.resize(srcSize, std::vector<float>(destSize));
		std::random_device rd;
		std::mt19937 gen(rd());
		std::normal_distribution<float> HE_init(0, sqrtf(2 / srcSize));
		for (int i = 0; i < srcSize; i++)
		{
			for (int j = 0; j < destSize; j++)
			{
				v[i][j] = HE_init(gen);
			}
		}

#ifndef ONLY_FORWARD
		//learning data
		momentum.resize(srcSize, std::vector<float>(destSize, 0));
		forwardSum.resize(srcSize, std::vector<float>(destSize, 0));
#endif
	}
	
	// weight
	std::vector<std::vector<float>> v;

#ifndef ONLY_FORWARD
	// learning data
	std::vector<std::vector<float>> momentum;
	std::vector<std::vector<float>> forwardSum;
#endif

};