#pragma once

#include "sampler.h"
#include <random>

/* RANSAC Sampling */
template <typename DatumType>
class UniformSampler : public Sampler<DatumType>
{
public:
	UniformSampler(const size_t point_number_) :
		point_number(point_number_),
		generate(0, point_number_ - 1)
	{
		std::random_device rand_dev;
		generator = std::mt19937(rand_dev());
	}
	~UniformSampler() {}

	bool sample(
		const DatumType& data_,
		const size_t data_size_,
		const size_t sample_size_,
		int *sample_);

protected:
	size_t point_number;
    	std::mt19937 generator;
	std::uniform_int_distribution<int> generate;
};

template <typename DatumType>
bool UniformSampler<DatumType>::sample(
	const DatumType& data_,
	const size_t data_size_,
	const size_t sample_size_,
	int *sample_)
{	
	if (data_size_ < sample_size_)
		return false;

	if (sample_size_ == data_size_)
	{
		for (auto i = 0; i < data_size_; ++i)
			sample_[i] = i;
		return true;
	}

	for (auto i = 0; i < sample_size_; ++i)
	{
		do	
		{
			sample_[i] = generate(generator);
			for (auto j = 0; j < i; ++j)
				if (sample_[i] == sample_[j])
					continue;	
		} while (false);
		
	}
	return true;
}

