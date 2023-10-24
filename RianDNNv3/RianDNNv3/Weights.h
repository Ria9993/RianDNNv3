#pragma once

#include <vector>
#include "Layer.h"
#include "HyperParm.h"

namespace rian
{
	class Weights
	{
	public:
		Weights() = delete;
		Weights(int srcSize, int destSize);
		Weights(int srcSize, int destSize, float std);

		virtual void Forward(Layer& src_layer, Layer& dest_layer) = 0;
		virtual void Backprop(Layer& layer, Layer& frontLayer, HyperParm& hyperParm) = 0;

		// using as 2d [src_idx][dest_idx]
		std::vector<float> v;

		// learning data
#ifndef ONLY_FORWARD
		// using as 2d
		std::vector<float> momentum;
#endif

	};
}