#include "WeightsConv1d.h"

namespace rian
{
	void WeightsConv1d::Forward(Layer& src_layer, Layer& dest_layer)
	{
#pragma omp parallel for
		for (int out_i = 0; out_i < dest_layer.size; out_i++)
		{
			dest_layer.result[out_i] = 0;
			for (int kernel_i = 0; kernel_i < kernelSize; kernel_i++)
			{
				int src_i = out_i * stride + kernel_i;
				dest_layer.result[out_i] += src_layer.result[src_i] * v[kernel_i];
			}
		}
	}

	void WeightsConv1d::Backprop(Layer& layer, Layer& frontLayer, HyperParm& hyperParm)
	{
		for (int i = 0; i < kernelSize; i++)
			sum_grad_v[i] = 0.f;

#pragma omp parallel for 
		for (int i = 0; i < layer.size; i++)
		{
			for (int kernel_i = i % stride; kernel_i < kernelSize; kernel_i += stride)
			{
				if (i - kernel_i < 0)
					break;
				const int front_idx = (i - kernel_i) / stride;
				if (front_idx >= frontLayer.size)
					continue;

				// stacking derivative
				layer.backprop[i] += v[kernel_i] * frontLayer.backprop[front_idx];

				// weight gradient
				const float grad =
					(frontLayer.backprop[front_idx]) * layer.forwardSum[i];
				sum_grad_v[kernel_i] += grad;
			}
		}

		for (int i = 0; i < kernelSize; i++)
		{
			momentum[i] =
				(momentum[i] * hyperParm.MomentumRate)
				- (hyperParm.LearningRate * sum_grad_v[i]);
			v[i] += momentum[i];
		}
	}
}
