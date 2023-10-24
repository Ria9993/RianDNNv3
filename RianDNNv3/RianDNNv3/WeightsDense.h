#pragma once

#include "Weights.h"

namespace rian
{
	class WeightsDense : public Weights
	{
	public:
		WeightsDense(int srcSize, int destSize)
			: Weights(srcSize, destSize) {}

		void Forward(Layer& src_layer, Layer& dest_layer) override;
		void Backprop(Layer& layer, Layer& frontLayer, HyperParm& hyperParm) override;
	};
}
