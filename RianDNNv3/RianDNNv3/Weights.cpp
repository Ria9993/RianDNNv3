#include "Weights.h"

namespace rian
{
	Weights::Weights(int srcSize, int destSize)
	{
		// weight
		v.resize(srcSize, std::vector<float>(destSize));
		std::random_device rd;
		std::mt19937 gen(rd());
		std::normal_distribution<float> HE_init(0, sqrtf((float)2 / srcSize));
		for (int i = 0; i < srcSize; i++)
		{
			for (int j = 0; j < destSize; j++)
			{
				v[i][j] = HE_init(gen);
			}
		}

		// learning data
#ifndef ONLY_FORWARD
		momentum.resize(srcSize, std::vector<float>(destSize, 0));
		forwardSum.resize(srcSize, std::vector<float>(destSize, 0));
#endif
	}
}