#include "Layer.h"

namespace rian
{
	Layer::Layer(int size, Activation act, float biasInit)
	{
		this->size = size;
		this->act = act;

		bias.resize(size, biasInit);
		result.resize(size, 0);

		// learning data
#ifndef ONLY_FORWARD
		biasMomentum.resize(size, 0);
		forwardSum.resize(size, 0);
		actDiffSum.resize(size, 0);
		backprop.resize(size, 0);
#endif
	}
}