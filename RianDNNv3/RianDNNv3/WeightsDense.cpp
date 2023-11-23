#include "WeightsDense.h"
#include "RianDNN.h"

namespace rian
{
	WeightsDense::WeightsDense(int srcSize, int destSize)
		: Weights(srcSize, destSize)
	{
	}

	void WeightsDense::Forward(Layer& src_layer, Layer& dest_layer, Model& model)
	{
#ifndef GPGPU
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

#else
		//array_view<float, 1> in(src_layer.size, src_layer.result.data());
		//array_view<float, 2> w(src_layer.size, dest_layer.size, now_weight.v.data());
		//array_view<float, 1> out(dest_layer.size, dest_layer.result.data());
		//out.discard_data();

		array_view<float, 2>& w = *model.gpu_weight[src_layer.idx];
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
#endif
	}

	void WeightsDense::Backprop(Layer& layer, Layer& frontLayer, Model& model)
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

				/*momentum[idx_2d] =
					(momentum[idx_2d] * model.hyperParm.MomentumRate)
					- (model.hyperParm.LearningRate * grad);
				v[idx_2d] += momentum[idx_2d];*/

				momentum[idx_2d] =
					(momentum[idx_2d] * model.hyperParm.MomentumRate)
					+ ((1.f - model.hyperParm.MomentumRate) * frontLayer.backprop[front_idx]);
				RMSProp[idx_2d] =
					(RMSProp[idx_2d] * model.hyperParm.RMSPropRate)
					+ ((1.f - model.hyperParm.RMSPropRate) * (frontLayer.backprop[front_idx] * frontLayer.backprop[front_idx]));

				float Epsilon = 0.00001f;
				float M = momentum[idx_2d] / (1.f - model.hyperParm.MomentumRate);
				float G = RMSProp[idx_2d] / (1.f - model.hyperParm.RMSPropRate);
				v[idx_2d] -= (model.hyperParm.LearningRate / (sqrtf(G + Epsilon))) * M;
			}
		}
#ifdef GPGPU
		array_view<float, 2>updated_weight(layer.size, frontLayer.size, v.data());
		updated_weight.copy_to(*model.gpu_weight[layer.idx]);
#endif
	}

}
