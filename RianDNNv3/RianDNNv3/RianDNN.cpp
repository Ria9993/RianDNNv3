#include "RianDNN.h"

namespace rian
{
	void Model::AddLayer(int size, Activation act)
	{
		layers.emplace_back(size, act, hyperParm.BiasInitValue);

		size_t layersSize = layers.size();
		if (layersSize > 1)
		{
			Weights newWeights(layers[layersSize - 2].size, layers[layersSize - 1].size);
			weight.push_back(newWeights);
		}
	}

	std::vector<float>& Model::GetInputVector()
	{
		return layers.begin()->result;
	}

	const std::vector<float>& Model::GetResult()
	{
		return layers.rbegin()->result;
	}

	void Model::ComputeError(const std::vector<float>& target)
	{
		Layer& outLayer = layers[layers.size() - 1];
		for (int i = 0; i < outLayer.size; i++)
		{
			outLayer.backprop[i] += outLayer.result[i] - target[i];

			// Error = (Output - Target) ^ 2
			// utLayer.backprop[i] += (outLayer.result[i] - target[i]) * (outLayer.result[i] - target[i]);
		}

		// counting for case of many forward but only errorCompute once
		errorComputeCount += 1;
	}
}