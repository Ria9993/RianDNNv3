#include "RianDNN.h"

namespace rian
{
	Model::~Model()
	{
#ifdef GPGPU
		for (size_t i = 0; i < gpu_weight.size(); i++)
			delete gpu_weight[i];
#endif
		for (size_t i = 0; i < weight.size(); i++)
			delete weight[i];
	}

	void Model::AddLayer(int size, Activation act)
	{
		AddLayerDense(size, act);
	}

	void Model::AddLayerDense(int size, Activation act)
	{
		layers.emplace_back(size, act, hyperParm.BiasInitValue);

		size_t layersSize = layers.size();
		if (layersSize > 1)
		{
			weight.push_back(new WeightsDense(layers[layersSize - 2].size, layers[layersSize - 1].size));
		}
	}

	void Model::AddLayerConv1d(int kernelSize, int stride, Activation act)
	{
		//assert(layers[layers.size() - 1].size % stride == 0
		//	&& (layers[layers.size() - 1].size - kernelSize) % stride == 0);

		int layerSize = ((layers[layers.size() - 1].size - kernelSize) / stride) + 1;
		//int layerSize = (layers[layers.size() - 1].size / stride);
		layers.emplace_back(layerSize, act, hyperParm.BiasInitValue);

		size_t layersSize = layers.size();
		if (layersSize > 1)
		{
			weight.push_back(new WeightsConv1d(layers[layers.size() - 2].size, layers[layers.size() - 1].size, kernelSize, stride));
		}
	}

	void Model::Build()
	{
#ifdef GPGPU
		for (int layer_idx = 0; layer_idx < layers.size() - 1; layer_idx++)
		{
			Layer& src_layer = layers[layer_idx];
			Weights& now_weight = weight[layer_idx];
			Layer& dest_layer = layers[(size_t)layer_idx + 1];

			gpu_weight.push_back(
				new array_view<float, 2>(src_layer.size, dest_layer.size, now_weight.v.data()));

			//gpu_weight_momentum.push_back(
			//	new array_view<float, 2>(src_layer.size, dest_layer.size, now_weight.momentum.data()));
		}
#endif
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
		float grad_sum = 0;
		
		Layer& outLayer = layers[layers.size() - 1];
		for (int i = 0; i < outLayer.size; i++)
		{			
			float grad = 2 * (outLayer.result[i] - target[i]);
			grad_sum += abs(grad);
		}

		// Compute derivative about mean error
		for (int i = 0; i < outLayer.size; i++)
		{
			float grad = 2 * (outLayer.result[i] - target[i]);
			//outLayer.backprop[i] += grad * (abs(grad) / grad_sum);
			outLayer.backprop[i] += grad;
		}

		// counting for case of many forward but only errorCompute once
		errorComputeCount += 1;
	}
}