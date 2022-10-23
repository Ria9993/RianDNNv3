#include "Weights.h"
#include <random>
#include <cmath>

namespace rian
{
	Weights::Weights(int srcSize, int destSize)
	{
		// weight
		v.resize((size_t)srcSize * destSize);
		std::random_device rd;
		std::mt19937 gen(rd());
		std::normal_distribution<float> HE_init(0, sqrtf((float)2 / srcSize));
		for (int i = 0; i < srcSize * destSize; i++)
		{
			v[i] = HE_init(gen);
		}

		// learning data
#ifndef ONLY_FORWARD
		momentum.resize((size_t)srcSize * destSize, 0);
#endif
	}
}