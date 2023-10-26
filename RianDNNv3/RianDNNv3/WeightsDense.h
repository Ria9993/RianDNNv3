#pragma once

#include "Weights.h"

namespace rian
{
	class WeightsDense : public Weights
	{
	public:
		WeightsDense(int srcSize, int destSize);

		void Forward(Layer& src_layer, Layer& dest_layer, Model& model);
		void Backprop(Layer& layer, Layer& frontLayer, Model& model);
	};
}
