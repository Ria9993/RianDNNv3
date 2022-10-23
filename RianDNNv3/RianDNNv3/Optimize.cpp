#include "RianDNN.h"
#include <cassert> 

/* C++ OpenMP
	support Multi-thread */
#pragma warning(disable:6993)
#include <omp.h>

namespace rian
{
	void Model::Optimize()
	{
		Layer& outLayer = layers[layers.size() - 1];
		for (int i = 0; i < outLayer.size; i++)
		{
			outLayer.backprop[i] /= errorComputeCount;
			outLayer.backprop[i] *= outLayer.actDiffSum[i] / forwardCount;
		}

		// backprop & update
		for (int layer_idx = (int)layers.size() - 2; layer_idx >= 0; layer_idx--)
		{
			Layer& layer = layers[layer_idx];
			Layer& frontLayer = layers[(size_t)layer_idx + 1];

			// frontLayer bias update
			for (int i = 0; i < layer.size; i++)
			{
				frontLayer.biasMomentum[i] = 
					(frontLayer.biasMomentum[i] * hyperParm.MomentumRate)
					- (hyperParm.LearningRate * frontLayer.backprop[i]);
				frontLayer.bias[i] += frontLayer.biasMomentum[i];
			}

			// pull frontLayer's derivative and weight update
			for (int i = 0; i < layer.size; i++)
			{
				layer.forwardSum[i] /= forwardCount;
				for (int front_idx = 0; front_idx < frontLayer.size; front_idx++)
				{
					const size_t idx_2d = (size_t)i * frontLayer.size + front_idx;
					const float grad = frontLayer.backprop[front_idx] * layer.forwardSum[i];

					weight[layer_idx].momentum[idx_2d] =
						(weight[layer_idx].momentum[idx_2d] * hyperParm.MomentumRate)
						- (hyperParm.LearningRate * grad);
					weight[layer_idx].v[idx_2d] += weight[layer_idx].momentum[idx_2d];

					// stacking derivative
					layer.backprop[i] += grad;
				}
				// compute activation derivative
				layer.backprop[i] *= layer.actDiffSum[i] / forwardCount;
			}
		}

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
	}
}