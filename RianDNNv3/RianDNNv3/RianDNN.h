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
		Model(HyperParm hyperParm)
		{
			this->hyperParm = hyperParm;
			this->forwardCount = 0;
		}

		void AddLayer(int size, float (*Activation)(float));
		std::vector<float>& GetInputVector();
		void Forward();
		const std::vector<float>& GetResult();

	private:
		HyperParm hyperParm;

		std::vector<Layer> layers;
		std::vector<Weights> weight;

		// learning data
#ifndef ONLY_FORWARD
		int forwardCount;
#endif
	};
}