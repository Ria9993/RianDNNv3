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

			// compute weight
#ifdef GPGPU
			//array_view<float, 1> in(src_layer.size, src_layer.result.data());
			//array_view<float, 2> w(src_layer.size, dest_layer.size, now_weight.v.data());
			//array_view<float, 1> out(dest_layer.size, dest_layer.result.data());
			//out.discard_data();
			
			array_view<float, 2>& w = *gpu_weight[layer_idx];
			array_view<float, 1> in(src_layer.size, src_layer.result.data());
			array_view<float, 1> out(dest_layer.size, dest_layer.result.data());
			out.discard_data();

			// calculate weight multiply
			const int src_size = src_layer.size;
			parallel_for_each(
				out.extent,
				[=](index<1> idx) restrict(amp)
				{
					out[idx] = 0;
					for (int src_i = 0; src_i < src_size; src_i++)
					{
						out[idx] += in[src_i] * w[src_i][idx];
					}
				}
			);
			out.synchronize();

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

#else // ONLY CPU

			// calculate weight
			now_weight.Forward(src_layer, dest_layer);

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
#endif

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