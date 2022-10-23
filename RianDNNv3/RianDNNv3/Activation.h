#pragma once

#include <algorithm>

namespace rian
{
	class ReLU
	{
	public:
		float calc(float x)
		{
			return std::max(0, x);
		}
	};

	class None
	{
	public:
		float calc(float x)
		{
			return x;
		}
	};
}