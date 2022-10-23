#include "Layer.h"

namespace rian
{
	Layer::Layer(int size, float (*Activation)(float), float biasInit)
	{
		this->size = size;
		act = Activation;

		bias.resize(size, biasInit);
		result.resize(size, 0);

		// learning data
#ifndef ONLY_FORWARD
		backpropGrad.resize(size, 0);
#endif
	}
}