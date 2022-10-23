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
		float (*act)(float);

		// learning data
#ifndef ONLY_FORWARD
		std::vector<float> backpropGrad;
#endif
	};
}