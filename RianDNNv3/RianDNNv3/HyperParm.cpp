namespace rian
{
	class HyperParm
	{
	public:
		float LearningRate;
		float MomentumRate;
		float BiasInitValue;
		
		HyperParm()
		{
			/* Default Value */
			LearningRate = 0.0001f;
			MomentumRate = 0.9f;
			BiasInitValue = 0.01f;
		}

		HyperParm& operator=(HyperParm& rhs) = default;
	};
}