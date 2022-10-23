#include "RianDNN.h"

namespace rian
{
	void Model::AddLayer(int size, float (*Activation)(float))
	{
		layers.emplace_back(size, Activation, hyperParm.BiasInitValue);

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
}