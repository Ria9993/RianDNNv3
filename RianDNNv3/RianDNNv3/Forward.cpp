#include "RianDNN.h"
#include <algorithm>

/* C++ AMP
	 not supported after VS2022 */
//#define GPGPU
#ifdef GPGPU
#include <amp.h>
using namespace concurrency;
#endif

/* C++ OpenMP
	support Multi-thread */
#pragma warning(disable:6993)
#include <omp.h>

namespace rian
{
	void Model::Forward()
	{
		for (int layer_idx = 0; layer_idx < layers.size() - 1; layer_idx++)
		{
			Layer& src_layer = layers[layer_idx];
			Weights& now_weight = weight[layer_idx];
			Layer& dest_layer = layers[(size_t)layer_idx + 1];

			// compute weight
#ifdef GPGPU //(Not Implemented yet)
			array_view<float, 1> in(src_layer.size, src_layer.result.data());
			array_view<float, 2> w(src_layer.size, dest_layer.size, now_weight.v.data());
			array_view<float, 1> out(dest_layer.size, dest_layer.result.data());
			out.discard_data();

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

#else // ONLY CPU
#pragma omp parallel for
			for (int out_i = 0; out_i < dest_layer.size; out_i++)
			{
				// calculate weight
				dest_layer.result[out_i] = 0;
				for (int src_i = 0; src_i < src_layer.size; src_i++)
				{
					const size_t idx_2d = (size_t)src_i * dest_layer.size + out_i;

					dest_layer.result[out_i] += src_layer.result[src_i] * now_weight.v[idx_2d];
				}

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
			}
#endif
		}

		forwardCount += 1;
	}
}