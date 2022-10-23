#pragma once

#include <algorithm>

namespace rian
{
	enum class Activation
	{
		ReLU,
		None
	};

	//__forceinline float ReLU(float x)
	//{
	//	return std::max(0.0f, x);
	//}

	//__forceinline float None(float x)
	//{
	//	return x;
	//}
}