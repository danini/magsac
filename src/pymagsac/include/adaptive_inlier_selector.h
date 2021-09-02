#pragma once

#include "estimators/estimator.h"
#include "model.h"

#include <vector>

template <class ModelEstimator>
class AdaptiveInlierSelector
{
public:
	virtual void selectInliers(
		const cv::Mat& points_, // The input data points
		const ModelEstimator &estimator_, // The model estrimator object
		const gcransac::Model& reference_model_, // The reference model used for the distance calculation
		std::vector<size_t> &inliers_,// The selected inliers
		double& bestThreshold_) const = 0; 
};