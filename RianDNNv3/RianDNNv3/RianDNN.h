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
			this->errorComputeCount = 0;
		}

		void AddLayer(int size, Activation act);
		std::vector<float>& GetInputVector();
		void Forward();
		void ComputeError(const std::vector<float>& target);
		void Optimize();
		const std::vector<float>& GetResult();

		HyperParm hyperParm;

		std::vector<Layer> layers;
		std::vector<Weights> weight;

		// learning data
#ifndef ONLY_FORWARD
		int forwardCount;
		int errorComputeCount;
#endif
	};
}