#include "WeightsConv1d.h"
#include "RianDNN.h"

namespace rian
{
	WeightsConv1d::WeightsConv1d(int srcSize, int destSize, int kernelSize0, int stride0)
		: Weights(kernelSize0, 1, sqrtf((float)2 / (srcSize + destSize)))
		, kernelSize(kernelSize0)
		, stride(stride0)
	{
		//v[kernelSize / 2] = 1.f;
		sum_grad_v.resize(kernelSize0);
	}

	void WeightsConv1d::Forward(Layer& src_layer, Layer& dest_layer, Model& model)
	{
#pragma omp parallel for
		for (int out_i = 0; out_i < dest_layer.size; out_i++)
		{
			dest_layer.result[out_i] = 0;
			for (int kernel_i = 0; kernel_i < kernelSize; kernel_i++)
			{
				int src_i = (out_i * stride + kernel_i) - (kernelSize / 2);

				if (src_i < 0 || src_layer.size <= src_i)
					continue;
				else
					dest_layer.result[out_i] += src_layer.result[src_i] * v[kernel_i];
			}
		}
	}

	void WeightsConv1d::Backprop(Layer& layer, Layer& frontLayer, Model& model)
	{
		for (int i = 0; i < kernelSize; i++)
			sum_grad_v[i] = 0.f;

#pragma omp parallel for 
		for (int i = 0; i < layer.size; i++)
		{
			for (int kernel_i = (i + (kernelSize / 2)) % stride; kernel_i < kernelSize; kernel_i += stride)
			{
				const int front_idx = ((i + (kernelSize / 2)) - kernel_i) / stride;
				if (((i + (kernelSize / 2)) - kernel_i) < 0)
					break;
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
			//momentum[i] =
			//	(momentum[i] * model.hyperParm.MomentumRate)
			//	- (model.hyperParm.LearningRate * sum_grad_v[i]);
			//v[i] += momentum[i];

			momentum[i] =
				(momentum[i] * model.hyperParm.MomentumRate)
				+ ((1.f - model.hyperParm.MomentumRate) * sum_grad_v[i]);
			RMSProp[i] =
				(RMSProp[i] * model.hyperParm.RMSPropRate)
				+ ((1.f - model.hyperParm.RMSPropRate) * (sum_grad_v[i] * sum_grad_v[i]));

			float Epsilon = 0.00001f;
			float M = momentum[i] / (1.f - model.hyperParm.MomentumRate);
			float G = RMSProp[i] / (1.f - model.hyperParm.RMSPropRate);
			v[i] -= (model.hyperParm.LearningRate / (sqrtf(G + Epsilon))) * M;
		}
	}
}
