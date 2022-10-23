#include "RianDNN.h"

namespace rian
{
	void Model::AddLayer(int size, float(*Activation)(float))
	{
		layers.emplace_back(size, Activation, hyperParm.BiasInitValue);

		size_t layersSize = layers.size();
		if (layersSize > 1)
		{
			weight.emplace_back(layers[layersSize - 2].size, layers[layersSize - 1].size);
		}
	}
}