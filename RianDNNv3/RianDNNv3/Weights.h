#pragma once

#include <vector>

namespace rian
{
	class Weights
	{
	public:
		//Weights() = delete;
		Weights(int srcSize, int destSize);

		// using by 2d
		std::vector<float> v;

		// learning data
#ifndef ONLY_FORWARD
		// using by 2d
		std::vector<float> momentum;
#endif

	};
}