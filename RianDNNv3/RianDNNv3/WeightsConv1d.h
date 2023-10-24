#pragma once

#include "Weights.h"

namespace rian
{
	class WeightsConv1d : public Weights
	{
	public:
		WeightsConv1d(int srcSize, int destSize, int kernelSize0, int stride0)
			: Weights(kernelSize0, 1)
			, kernelSize(kernelSize0)
			, stride(stride0)
		{
			sum_grad_v.resize(kernelSize0);
		}

		void Forward(Layer& src_layer, Layer& dest_layer) override;
		void Backprop(Layer& layer, Layer& frontLayer, HyperParm& hyperParm) override;

		int kernelSize;
		int stride;

		std::vector<float> sum_grad_v;
	};
}
