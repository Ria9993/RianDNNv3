#include "Layer.h"

namespace rian
{
	Layer::Layer(LayerType type, int size, int idx, Activation act, float biasInit)
	{
		this->type = type;

		this->size = size;
		this->act = act;
		this->idx = idx;

		bias.resize(size, biasInit);
		result.resize(size, 0);

		// learning data
#ifndef ONLY_FORWARD
		biasMomentum.resize(size, 0);
		biasRMSProp.resize(size, 0);
		forwardSum.resize(size, 0);
		actDiffSum.resize(size, 0);
		backprop.resize(size, 0);
#endif
	}
}