#pragma once

/* RANSAC Scoring */
class ModelScore
{
public:
	/* number of inliers, rectangular gain function */
	size_t I;
	/* MSAC scoring, truncated quadratic gain function */
	double J;
	/* The log probability of the model considering that the inliers are normally and the outliers are uniformly distributed */
	double P;
	/* Iteration number when it is found */
	size_t iteration;

	ModelScore() : I(0), J(0), P(0), iteration(0) { }
};

using Score = ModelScore;
