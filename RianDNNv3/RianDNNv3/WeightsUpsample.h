#pragma once

#include "Weights.h"

namespace rian
{
	class WeightsUpsample : public Weights
	{
	public:
		WeightsUpsample(int srcSize, int destSize)
			: Weights(srcSize, destSize) {}

		void Forward(Layer& src_layer, Layer& dest_layer) override;
		void Backprop(Layer& layer, Layer& frontLayer, HyperParm& hyperParm) override;
	};
}
