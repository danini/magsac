#pragma once

/* RANSAC Sampling */
template <typename DatumType>
class Sampler
{
public:
	Sampler() {}
	virtual ~Sampler() {}

	virtual bool sample(
		const DatumType& data_,
		const size_t data_size_,
		const size_t sample_size_,
		int *sample_) = 0;
};

