#pragma once

#include <cassert>
#include <algorithm>
#include <vector>
#include <iostream>
#include "Layer.h"
#include "Weights.h"
#include "Activation.h"
#include "HyperParm.h"
#include "WeightsDense.h"
#include "WeightsConv1d.h"
#include "WeightsTransConv1d.h"
#include "WeightsRNN.h"

/* If you want enable GPGPU, define [#define GPGPU] 
	C++ AMP
	not supported after VS2019 */

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
	class Model
	{
	public:
		Model();
		Model(HyperParm hyperParm);
		~Model();

		void AddLayer(int size, Activation act);
		void AddLayerDense(int size, Activation act);
		// The size of the previous layer must be a multiple of stride. 
		void AddLayerConv1d(int kernelSize, int stride, Activation act);
		void AddLayerTransConv1d(int kernelSize, int stride, Activation act);
		void Build();
		std::vector<float>& GetInputVector();
		void Forward();
		void ComputeError(const std::vector<float>& target);
		void Optimize();
		const std::vector<float>& GetResult();

		void Save(const char* filename);
		void Load(const char* filename);

		HyperParm hyperParm;

		std::vector<Layer> layers;
		std::vector<Weights*> weight;

		

		// learning data
#ifndef ONLY_FORWARD
		int forwardCount;
		int errorComputeCount;
#endif

		/* GPGPU */
#ifdef GPGPU
		std::vector<array_view<float, 2>*> gpu_weight;
		std::vector<array_view<float, 2>*> gpu_weight_momentum;
#endif
	};
}