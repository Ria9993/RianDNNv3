#include "RianDNN.h"
#include <cassert> 

/* C++ OpenMP
	support Multi-thread */
#pragma warning(disable:6993)
#include <omp.h>
#include <iostream>

using std::min;
using std::max;

namespace rian
{
	void Model::Optimize()
	{
		Layer& outLayer = layers[layers.size() - 1];
		for (int i = 0; i < outLayer.size; i++)
		{
			outLayer.backprop[i] /= errorComputeCount;
			outLayer.backprop[i] *= (outLayer.actDiffSum[i] / forwardCount);
		}

		// backprop & update
		for (int layer_idx = (int)layers.size() - 2; layer_idx >= 0; layer_idx--)
		{
			Layer& layer = layers[layer_idx];
			Layer& frontLayer = layers[(size_t)layer_idx + 1];

			// frontLayer bias update
			for (int i = 0; i < frontLayer.size; i++)
			{
				frontLayer.biasMomentum[i] = 
					(frontLayer.biasMomentum[i] * hyperParm.MomentumRate)
					- (hyperParm.LearningRate * frontLayer.backprop[i]);
				frontLayer.bias[i] += frontLayer.biasMomentum[i];
			}

			// Gradient clipping as
			const float clip_threshold = 2.f;
			for (int i = 0; i < frontLayer.size; i++)
			{
				//frontLayer.backprop[i] = min(frontLayer.backprop[i], clip_threshold);
				//frontLayer.backprop[i] = max(frontLayer.backprop[i], -clip_threshold);
			}

			// pull frontLayer's derivative and weight update
#pragma omp parallel for
			for (int i = 0; i < layer.size; i++)\
			{
				for (int front_idx = 0; front_idx < frontLayer.size; front_idx++)
				{
					const size_t idx_2d = (size_t)i * frontLayer.size + front_idx;
					const float grad =
						(frontLayer.backprop[front_idx])
						* (layer.forwardSum[i] / forwardCount);

					// stacking derivative
					layer.backprop[i] += weight[layer_idx].v[idx_2d] * frontLayer.backprop[front_idx];

					weight[layer_idx].momentum[idx_2d] =
						(weight[layer_idx].momentum[idx_2d] * hyperParm.MomentumRate)
						- (hyperParm.LearningRate * grad);
					weight[layer_idx].v[idx_2d] += weight[layer_idx].momentum[idx_2d];

				}				
				//layer.backprop[i] = (layer.backprop[i] / forwardCount) * (layer.actDiffSum[i] / forwardCount);
				layer.backprop[i] *= (layer.actDiffSum[i] / forwardCount);
			}
		}

#ifdef GPGPU
		for (int layer_idx = 0; layer_idx < layers.size() - 1; layer_idx++)
		{
			Layer& src_layer = layers[layer_idx];
			Weights& now_weight = weight[layer_idx];
			Layer& dest_layer = layers[(size_t)layer_idx + 1];

			array_view<float, 2>updated_weight(src_layer.size, dest_layer.size, now_weight.v.data());
			updated_weight.copy_to(*gpu_weight[layer_idx]);
		}
#endif

		
		// clear gradient
		for (int i = 0; i < layers.size(); i++)
		{
			for (int j = 0; j < layers[i].size; j++)
			{
				layers[i].actDiffSum[j] = 0;
				layers[i].backprop[j] = 0;
				layers[i].forwardSum[j] = 0;
			}

		}

		// clear parameter
		forwardCount = 0;
		errorComputeCount = 0;

		//hyperParm.LearningRate *= 0.999f;
		//std::cout.precision(10);
		//std::cout << "lr = " << hyperParm.LearningRate << std::endl;
	}
}