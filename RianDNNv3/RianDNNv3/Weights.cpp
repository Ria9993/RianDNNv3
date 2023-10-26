#include <random>
#include <cmath>
#include "Weights.h"

namespace rian
{
	Weights::Weights(int srcSize, int destSize)
	{
		// weight
		v.resize((size_t)srcSize * destSize);
		std::random_device rd;
		std::mt19937 gen(rd());
		std::normal_distribution<float> HE_init(0, sqrtf((float)2 / (srcSize)));
		//std::uniform_real_distribution<float> HE_init(-sqrtf(6 / srcSize), -sqrtf(6 / srcSize));
		//std::uniform_real_distribution<float> HE_init(-(sqrtf(3) / sqrtf(srcSize)), (sqrtf(3) / sqrtf(srcSize)));
#pragma omp parallel for
		for (int i = 0; i < srcSize * destSize; i++)
		{
			v[i] = HE_init(gen);
		}

		// learning data
#ifndef ONLY_FORWARD
		momentum.resize((size_t)srcSize * destSize, 0);
#endif
	}

	Weights::Weights(int srcSize, int destSize, float std)
	{
		// weight
		v.resize((size_t)srcSize * destSize);
		std::random_device rd;
		std::mt19937 gen(rd());
		std::normal_distribution<float> HE_init(0, std);
		//std::uniform_real_distribution<float> HE_init(-sqrtf(6 / srcSize), -sqrtf(6 / srcSize));
		//std::uniform_real_distribution<float> HE_init(-(sqrtf(3) / sqrtf(srcSize)), (sqrtf(3) / sqrtf(srcSize)));

#pragma omp parallel for
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