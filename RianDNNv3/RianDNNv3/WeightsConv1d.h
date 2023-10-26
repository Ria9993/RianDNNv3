#pragma once

#include "Weights.h"

namespace rian
{
	class WeightsConv1d : public Weights
	{
	public:
		WeightsConv1d(int srcSize, int destSize, int kernelSize0, int stride0);
		void Forward(Layer& src_layer, Layer& dest_layer, Model& model);
		void Backprop(Layer& layer, Layer& frontLayer, Model& model);

		int kernelSize;
		int stride;

		std::vector<float> sum_grad_v;
	};
}
