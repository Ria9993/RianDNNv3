#include "RianDNN.h"
#include <algorithm>

namespace rian
{
	void Model::Forward()
	{
		for (int layer_idx = 0; layer_idx < layers.size() - 1; layer_idx++)
		{
			Layer& src_layer = layers[layer_idx];
			Weights& now_weight = *weight[layer_idx];
			Layer& dest_layer = layers[(size_t)layer_idx + 1];

			// calculate weight
			now_weight.Forward(src_layer, dest_layer, *this);

#pragma omp parallel for
			for (int out_i = 0; out_i < dest_layer.size; out_i++)
			{
				// compute bias & act_func
				dest_layer.result[out_i] += dest_layer.bias[out_i];

				// compute activation function & derivative
				switch (dest_layer.act)
				{
				case Activation::ReLU:
					if (dest_layer.result[out_i] > 0)
						dest_layer.actDiffSum[out_i] += 1;
					else
						dest_layer.result[out_i] = 0;
					break;
				case Activation::LeakyReLU:
					if (dest_layer.result[out_i] > 0)
						dest_layer.actDiffSum[out_i] += 1;
					else
					{
						dest_layer.result[out_i] *= 0.01f;
						dest_layer.actDiffSum[out_i] += 0.01f;
					}
					break;
				case Activation::None:
					dest_layer.actDiffSum[out_i] += 1;
					break;
				}
			}

			// compute weight differentiation
#ifndef ONLY_FORWARD
			for (int i = 0; i < src_layer.size; i++)
			{
				src_layer.forwardSum[i] += src_layer.result[i];
				//src_layer.forwardSum[i] += src_layer.result[i] * src_layer.result[i];
			}
#endif
		}

		// forward Counting
		forwardCount += 1;
	}
}