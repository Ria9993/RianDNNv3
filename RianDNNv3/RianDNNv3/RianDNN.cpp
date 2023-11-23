#define _CRT_SECURE_NO_WARNINGS
#include "RianDNN.h"

namespace rian
{
	Model::Model()
	{
		this->forwardCount = 0;
		this->errorComputeCount = 0;
	}

	Model::Model(HyperParm hyperParm)
	{
		this->hyperParm = hyperParm;
		this->forwardCount = 0;
		this->errorComputeCount = 0;
	}

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
		layers.emplace_back(LayerType::Dense, size, layers.size(), act, hyperParm.BiasInitValue);
		layers[layers.size() - 1].type = LayerType::Dense;

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

		//int layerSize = ((layers[layers.size() - 1].size - kernelSize) / stride) + 1;
		int layerSize = (layers[layers.size() - 1].size / stride);
		layers.emplace_back(LayerType::Conv1d, layerSize, layers.size(), act, hyperParm.BiasInitValue);
		layers[layers.size() - 1].type = LayerType::Conv1d;

		size_t layersSize = layers.size();
		if (layersSize > 1)
		{
			weight.push_back(new WeightsConv1d(layers[layers.size() - 2].size, layers[layers.size() - 1].size, kernelSize, stride));
		}
	}

	void Model::AddLayerTransConv1d(int kernelSize, int stride, Activation act)
	{
		int layerSize = (layers[layers.size() - 1].size * stride);
		layers.emplace_back(LayerType::TransConv1d, layerSize, layers.size(), act, hyperParm.BiasInitValue);
		layers[layers.size() - 1].type = LayerType::TransConv1d;

		size_t layersSize = layers.size();
		if (layersSize > 1)
		{
			weight.push_back(new WeightsTransConv1d(layers[layers.size() - 2].size, layers[layers.size() - 1].size, kernelSize, stride));
		}
	}

	void Model::Build()
	{
#ifdef GPGPU
		for (int layer_idx = 0; layer_idx < layers.size() - 1; layer_idx++)
		{
			Layer& src_layer = layers[layer_idx];
			Weights& now_weight = *weight[layer_idx];
			Layer& dest_layer = layers[(size_t)layer_idx + 1];

			gpu_weight.push_back(
				new array_view<float, 2>(src_layer.size, dest_layer.size, now_weight.v.data()));

			//gpu_weight_momentum.push_back(
			//	new array_view<float, 2>(src_layer.size, dest_layer.size, now_weight.momentum.data()));
		}

		std::vector<accelerator> accs = accelerator::get_all();
		accelerator acc_chosen = accs[1];

		accelerator::set_default(acc_chosen.device_path);
		  std::wcout << "Accelerator: " << acc_chosen.description << std::endl;
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

	// Save weights and momentum to file
	void Model::Save(const char* filename)
	{
		FILE* fp = fopen(filename, "wb");
		if (fp == NULL)
		{
			printf("Error: can't open file %s\n", filename);
			return;
		}

		// Save hyperparm
		fwrite(&hyperParm, sizeof(HyperParm), 1, fp);

		size_t layerNum = layers.size();
		fwrite(&layerNum, sizeof(size_t), 1, fp);

		for (int layer_idx = 0; layer_idx < layerNum; layer_idx++)
		{
			// Save layer type
			LayerType layerType = layers[layer_idx].type;
			fwrite(&layerType, sizeof(LayerType), 1, fp);
			// Save layer activation
			Activation layerAct = layers[layer_idx].act;
			fwrite(&layerAct, sizeof(Activation), 1, fp);
			// Save layer size
			size_t layerSize = layers[layer_idx].size;
			fwrite(&layerSize, sizeof(size_t), 1, fp);
			// Save layer bias and momentum
			fwrite(layers[layer_idx].bias.data(), sizeof(float), layerSize, fp);
			fwrite(layers[layer_idx].biasMomentum.data(), sizeof(float), layerSize, fp);
			fwrite(layers[layer_idx].biasRMSProp.data(), sizeof(float), layerSize, fp);

			if (layer_idx == 0)
				continue;

			// Save Weight and elements as type
			if (layerType == LayerType::Dense) {
				WeightsDense* now_weight = (WeightsDense*)weight[layer_idx - 1];
				size_t weightSize = now_weight->v.size();
				fwrite(&weightSize, sizeof(size_t), 1, fp);
				fwrite(now_weight->v.data(), sizeof(float), now_weight->v.size(), fp);
				fwrite(now_weight->momentum.data(), sizeof(float), now_weight->momentum.size(), fp);
				fwrite(now_weight->RMSProp.data(), sizeof(float), now_weight->RMSProp.size(), fp);
			}
			else if (layerType == LayerType::Conv1d) {
				WeightsConv1d* now_weight = (WeightsConv1d*)weight[layer_idx - 1];
				size_t weightSize = now_weight->v.size();
				fwrite(&weightSize, sizeof(size_t), 1, fp);
				fwrite(now_weight->v.data(), sizeof(float), now_weight->v.size(), fp);
				fwrite(now_weight->momentum.data(), sizeof(float), now_weight->momentum.size(), fp);
				fwrite(now_weight->RMSProp.data(), sizeof(float), now_weight->RMSProp.size(), fp);
				fwrite(&now_weight->kernelSize, sizeof(int), 1, fp);
				fwrite(&now_weight->stride, sizeof(int), 1, fp);
			}
			else if (layerType == LayerType::TransConv1d) {
				WeightsTransConv1d* now_weight = (WeightsTransConv1d*)weight[layer_idx - 1];
				size_t weightSize = now_weight->v.size();
				fwrite(&weightSize, sizeof(size_t), 1, fp);
				fwrite(now_weight->v.data(), sizeof(float), now_weight->v.size(), fp);
				fwrite(now_weight->momentum.data(), sizeof(float), now_weight->momentum.size(), fp);
				fwrite(now_weight->RMSProp.data(), sizeof(float), now_weight->RMSProp.size(), fp);
				fwrite(&now_weight->kernelSize, sizeof(int), 1, fp);
				fwrite(&now_weight->stride, sizeof(int), 1, fp);
			}
			else {
				printf("Error: Unknown layer type\n");
				return;
			}
		}

		fclose(fp);
	}

	void Model::Load(const char* filename)
	{
		FILE* fp = fopen(filename, "rb");
		if (fp == NULL)
		{
			printf("Error: can't open file %s\n", filename);
			return;
		}
		// Load hyperparm
		fread(&hyperParm, sizeof(HyperParm), 1, fp);
		// Load layer size
		size_t layerNum;
		fread(&layerNum, sizeof(size_t), 1, fp);
		layers.resize(layerNum);
		weight.resize(layerNum - 1);
		for (int layer_idx = 0; layer_idx < layerNum; layer_idx++)
		{
			layers[layer_idx].idx = layer_idx;

			// Load layer type
			LayerType layerType;
			fread(&layerType, sizeof(LayerType), 1, fp);
			layers[layer_idx].type = layerType;
			// Load layer activation
			Activation layerAct;
			fread(&layerAct, sizeof(Activation), 1, fp);
			layers[layer_idx].act = layerAct;
			// Load layer size
			size_t layerSize;
			fread(&layerSize, sizeof(size_t), 1, fp);
			layers[layer_idx].size = layerSize;
			// Load layer bias and momentum
			layers[layer_idx].bias.resize(layerSize);
			fread(layers[layer_idx].bias.data(), sizeof(float), layerSize, fp);
			layers[layer_idx].biasMomentum.resize(layerSize);
			fread(layers[layer_idx].biasMomentum.data(), sizeof(float), layerSize, fp);
			layers[layer_idx].biasRMSProp.resize(layerSize);
			fread(layers[layer_idx].biasRMSProp.data(), sizeof(float), layerSize, fp);

			// resize member vectors
			layers[layer_idx].result.resize(layerSize);
			layers[layer_idx].backprop.resize(layerSize);
			layers[layer_idx].actDiffSum.resize(layerSize);
			layers[layer_idx].forwardSum.resize(layerSize);

			if (layer_idx == 0)
				continue;

			// Load Weight and elements as type
			if (layerType == LayerType::Dense) {
				WeightsDense* now_weight = new WeightsDense();
				weight[layer_idx - 1] = now_weight;
				size_t weightSize;
				fread(&weightSize, sizeof(size_t), 1, fp);
				now_weight->v.resize(weightSize);
				fread(now_weight->v.data(), sizeof(float), weightSize, fp);
				now_weight->momentum.resize(weightSize);
				fread(now_weight->momentum.data(), sizeof(float), weightSize, fp);
				now_weight->RMSProp.resize(weightSize);
				fread(now_weight->RMSProp.data(), sizeof(float), weightSize, fp);
			}
			else if (layerType == LayerType::Conv1d) {
				WeightsConv1d* now_weight = new WeightsConv1d();
				weight[layer_idx - 1] = now_weight;
				size_t weightSize;
				fread(&weightSize, sizeof(size_t), 1, fp);
				now_weight->v.resize(weightSize);
				fread(now_weight->v.data(), sizeof(float), weightSize, fp);
				now_weight->momentum.resize(weightSize);
				fread(now_weight->momentum.data(), sizeof(float), weightSize, fp);
				now_weight->RMSProp.resize(weightSize);
				fread(now_weight->RMSProp.data(), sizeof(float), weightSize, fp);
				fread(&now_weight->kernelSize, sizeof(int), 1, fp);
				fread(&now_weight->stride, sizeof(int), 1, fp);
				now_weight->sum_grad_v.resize(weightSize);
			}
			else if (layerType == LayerType::TransConv1d) {
				WeightsTransConv1d* now_weight = new WeightsTransConv1d();
				weight[layer_idx - 1] = now_weight;
				size_t weightSize;
				fread(&weightSize, sizeof(size_t), 1, fp);
				now_weight->v.resize(weightSize);
				fread(now_weight->v.data(), sizeof(float), weightSize, fp);
				now_weight->momentum.resize(weightSize);
				fread(now_weight->momentum.data(), sizeof(float), weightSize, fp);
				now_weight->RMSProp.resize(weightSize);
				fread(now_weight->RMSProp.data(), sizeof(float), weightSize, fp);
				fread(&now_weight->kernelSize, sizeof(int), 1, fp);
				fread(&now_weight->stride, sizeof(int), 1, fp);
				now_weight->sum_grad_v.resize(weightSize);
			}
			else {
				printf("Error: Unknown layer type\n");
				return;	
			}
		}
		fclose(fp);
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