#include "RianDNN.h"

// C++ AMP
// not supported after VS2022
#include <amp.h>
using namespace concurrency;

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
#if 1 // GPGPU
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
						//out[idx] += in[src_i] * w[src_i][idx];
					}
				}
			);
			out.synchronize();

#else // ONLY CPU
#pragma omp parallel
			for (int out_i = 0; out_i < dest_layer.size; out_i++)
			{
				dest_layer.result[out_i] = 0;
				for (int src_i = 0; src_i < src_layer.size; src_i++)
				{
					dest_layer.result[out_i] += src_layer.result[src_i] * now_weight.v[(size_t)src_i * dest_layer.size + out_i];
				}
			}
#endif

			// compute bias & act_func
#pragma omp parallel
			for (int i = 0; i < dest_layer.size; i++)
			{
				dest_layer.result[i] += dest_layer.bias[i];
				dest_layer.result[i] = dest_layer.act(dest_layer.result[i]);
			}
		}
	}
}