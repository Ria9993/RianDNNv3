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
		Model(HyperParm& hyperParm)
		{
			this->hyperParm = hyperParm;
		}

		void AddLayer(int size, float (*Activation)(float));

	private:
		HyperParm hyperParm;

		std::vector<Layer> layers;
		std::vector<Weights> weight;
	};
}