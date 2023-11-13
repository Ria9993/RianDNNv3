#pragma once

#include "Activation.h"
#include <vector>

namespace rian
{
	enum class LayerType;

	class Layer
	{
	public:
		Layer() = default;
		Layer(LayerType type, int size, int idx, Activation act, float biasInit);
		//~Layer() = default;
		
		int size;
		int idx;

		std::vector<float> bias;
		std::vector<float> result;
		Activation act;

		LayerType type;

		// learning data
#ifndef ONLY_FORWARD
		std::vector<float> actDiffSum; // activation function derivative
		std::vector<float> forwardSum; // forwarded value sum (for calculate weight gradient)
		std::vector<float> backprop;
		std::vector<float> biasMomentum;
#endif
	};

	enum class LayerType
	{
		Dense,
		Conv1d,
		TransConv1d,
		RNN,
		LSTM
	};
}