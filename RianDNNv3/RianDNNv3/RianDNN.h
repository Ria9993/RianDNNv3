#pragma once

#include <algorithm>
#include <vector>
#include "HyperParm.h"
#include "Layer.h"
#include "Weights.h"
#include "Activation.h"

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
		Model(HyperParm hyperParm)
		{
			this->hyperParm = hyperParm;
			this->forwardCount = 0;
			this->errorComputeCount = 0;
		}

		void AddLayer(int size, Activation act);
		void Build();
		std::vector<float>& GetInputVector();
		void Forward();
		void ComputeError(const std::vector<float>& target);
		void Optimize();
		const std::vector<float>& GetResult();

		HyperParm hyperParm;

		std::vector<Layer> layers;
		std::vector<Weights> weight;

		// learning data
#ifndef ONLY_FORWARD
		int forwardCount;
		int errorComputeCount;
#endif

		/* GPGPU */
#ifdef GPGPU
		std::vector<array_view<float, 2>*> gpu_weight;
#endif
	};
}