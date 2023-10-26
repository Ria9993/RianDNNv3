#pragma once

#include "Activation.h"
#include <vector>

namespace rian
{
	class Layer
	{
	public:
		Layer(int size, int idx, Activation act, float biasInit);
		//~Layer() = default;
		
		int size;
		int idx;

		std::vector<float> bias;
		std::vector<float> result;
		Activation act;

		// learning data
#ifndef ONLY_FORWARD
		std::vector<float> actDiffSum; // activation function derivative
		std::vector<float> forwardSum; // forwarded value sum (for calculate weight gradient)
		std::vector<float> backprop;
		std::vector<float> biasMomentum;
#endif
	};
}