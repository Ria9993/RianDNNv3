#include "WeightsDense.h"

namespace rian
{
	void WeightsDense::Forward(Layer& src_layer, Layer& dest_layer)
	{
#pragma omp parallel for
		for (int out_i = 0; out_i < dest_layer.size; out_i++)
		{
			dest_layer.result[out_i] = 0;
			for (int src_i = 0; src_i < src_layer.size; src_i++)
			{
				const size_t idx_2d = (size_t)src_i * dest_layer.size + out_i;

				dest_layer.result[out_i] += src_layer.result[src_i] * v[idx_2d];
			}
		}
	}

	void WeightsDense::Backprop(Layer& layer, Layer& frontLayer, HyperParm& hyperParm)
	{
#pragma omp parallel for
		for (int i = 0; i < layer.size; i++)
		{
			for (int front_idx = 0; front_idx < frontLayer.size; front_idx++)
			{
				const size_t idx_2d = (size_t)i * frontLayer.size + front_idx;
				const float grad =
					(frontLayer.backprop[front_idx]) * layer.forwardSum[i];

				// stacking derivative
				layer.backprop[i] += v[idx_2d] * frontLayer.backprop[front_idx];

				momentum[idx_2d] =
					(momentum[idx_2d] * hyperParm.MomentumRate)
					- (hyperParm.LearningRate * grad);
				v[idx_2d] += momentum[idx_2d];

			}
		}
	}
}
