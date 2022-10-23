#pragma once

#include <vector>

namespace rian
{
	class Layer
	{
	public:
		Layer(int size, float (*Activation)(float), float biasInit);
		
		int size;

		std::vector<float> bias;
		std::vector<float> result;

		// learning data
#ifndef ONLY_FORWARD
		std::vector<float> backpropGrad;
#endif
	};
}