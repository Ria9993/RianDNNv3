#pragma once

#include <vector>
#include <random>
#include <cmath>

namespace rian
{
	class Weights
	{
	public:
		Weights(int srcSize, int destSize);

		// weight
		std::vector<std::vector<float>> v;

		// learning data
#ifndef ONLY_FORWARD
		std::vector<std::vector<float>> momentum;
		std::vector<std::vector<float>> forwardSum;
#endif

	};
}