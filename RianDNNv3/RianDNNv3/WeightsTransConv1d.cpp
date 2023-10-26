#include "WeightsTransConv1d.h"
#include "RianDNN.h"

namespace rian
{

	WeightsTransConv1d::WeightsTransConv1d(int srcSize, int destSize, int kernelSize0, int stride0)
		: Weights(kernelSize0, 1)
		, kernelSize(kernelSize0)
		, stride(stride0)
	{
		sum_grad_v.resize(kernelSize0);
	}

	void WeightsTransConv1d::Forward(Layer& src_layer, Layer& dest_layer, Model& model)
	{
#pragma omp parallel for 
		for (int dest_idx = 0; dest_idx < dest_layer.size; dest_idx++)
		{
			dest_layer.result[dest_idx] = 0;
			for (int kernel_i = (dest_idx + (kernelSize / 2)) % stride; kernel_i < kernelSize; kernel_i += stride)
			{
				const int src_idx = ((dest_idx + (kernelSize / 2)) - kernel_i) / stride;
				if (((dest_idx + (kernelSize / 2)) - kernel_i) < 0)
					break;
				if (src_idx >= src_layer.size)
					continue;

				dest_layer.result[dest_idx] += v[kernel_i] * src_layer.result[src_idx];
			}
		}
	}

	void WeightsTransConv1d::Backprop(Layer& layer, Layer& frontLayer, Model& model)
	{
		for (int i = 0; i < kernelSize; i++)
			sum_grad_v[i] = 0.f;

#pragma omp parallel for 
		for (int src_i = 0; src_i < layer.size; src_i++)
		{
			for (int kernel_i = 0; kernel_i < kernelSize; kernel_i++)
			{
				int front_idx = (src_i * stride + kernel_i) - (kernelSize / 2);

				if (front_idx < 0 || frontLayer.size <= front_idx)
					continue;

				// stacking derivative
				layer.backprop[src_i] = v[kernel_i] * frontLayer.backprop[front_idx];

				// weight gradient
				sum_grad_v[kernel_i] += frontLayer.backprop[front_idx] * layer.forwardSum[src_i];
			}
		}

		for (int i = 0; i < kernelSize; i++)
		{
			momentum[i] =
				(momentum[i] * model.hyperParm.MomentumRate)
				- (model.hyperParm.LearningRate * sum_grad_v[i]);
			v[i] += momentum[i];
		}
	}
}
