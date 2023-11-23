#pragma once

#include <vector>
#include "Layer.h"
#include "HyperParm.h"


namespace rian
{
	class Model;

	class Weights
	{
	public:
		Weights() = default;
		Weights(int srcSize, int destSize);
		Weights(int srcSize, int destSize, float std);
		Weights(int srcSize, int destSize, float uniformMin, float uniformMax);

		virtual void Forward(Layer& src_layer, Layer& dest_layer, Model& model) = 0;
		virtual void Backprop(Layer& layer, Layer& frontLayer, Model& model) = 0;

		// using as 2d [src_idx][dest_idx]
		std::vector<float> v;

		// learning data
#ifndef ONLY_FORWARD
		// using as 2d
		std::vector<float> momentum;
		std::vector<float> RMSProp;
#endif

	};
}