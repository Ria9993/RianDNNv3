#pragma once

#include "HyperParm.h"
#include "Layer.h"
#include "Weights.h"
#include "Activation.h"

#include <vector>

namespace rian
{
	class Model
	{
	public:
		Model(const HyperParm& hyperParm)
		{
		}
		void AddLayer(int size, float (*Activation)(float))
		{
			layers.emplace_back(size, Activation);
		}

	private:
		std::vector<Layer<float>> layers;
		std::vector<Weights<float>> weight;
	};
}