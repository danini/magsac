#pragma once

#include <cstddef>

/* RANSAC Scoring */
class ModelScore
{
public:
	/* number of inliers, rectangular gain function */
	size_t inlier_number;
	/* MSAC scoring, truncated quadratic gain function */
	double score;
	/* The log probability of the model considering that the inliers are normally and the outliers are uniformly distributed */
	double probability;
	/* Iteration number when it is found */
	size_t iteration;

	ModelScore() : inlier_number(0), score(0), probability(0), iteration(0) { }

	inline bool operator<(const ModelScore& score_)
	{
		return score < score_.score;
	}
};

using Score = ModelScore;
