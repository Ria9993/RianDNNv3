#pragma once

namespace rian
{
	class HyperParm
	{
	public:
		float LearningRate;
		float MomentumRate;
		float BiasInitValue;
		float RMSPropRate;

		HyperParm()
		{
			/* Default Value */
			LearningRate = 0.0001f;
			MomentumRate = 0.9f;
			BiasInitValue = 0.01f;
			RMSPropRate = 0.999f;
		}

		HyperParm& operator=(HyperParm& rhs) = default;
	};
}