#pragma once

#include "estimators/estimator.h"
#include "model.h"
#include "adaptive_inlier_selector.h"

template <class ModelEstimator>
class MostSimilarInlierSelector : public AdaptiveInlierSelector<ModelEstimator>
{
protected:
	size_t minimumInlierNumber;
	double maximumThreshold;

	inline double modelToModelDistance(
		const gcransac::Model& referenceModel_,
		const gcransac::Model& model_,
		const std::vector<double>& referenceResiduals_,
		const std::vector<double>& residuals_) const;

	inline void getResiduals(
		const gcransac::Model& model_,
		const cv::Mat& points_,
		const ModelEstimator& estimator_,
		std::vector<double>& residuals_) const;

public:
	MostSimilarInlierSelector(size_t minimumInlierNumber_ = ModelEstimator::sampleSize(),
		double maximumThreshold_ = std::numeric_limits<double>::max()) :
		minimumInlierNumber(minimumInlierNumber_),
		maximumThreshold(maximumThreshold_)
	{
	}

	void selectInliers(
		const cv::Mat& points_, // The input data points
		const ModelEstimator& estimator_, // The model estrimator object
		const gcransac::Model& referenceModel_, // The reference model used for the distance calculation
		std::vector<size_t>& inliers_, // The selected inliers
		double &bestThreshold_) const
	{
		const size_t& pointNumber = points_.rows;

		// Calculate the residuals of the points given the reference model
		std::vector<double> referenceResiduals;
		getResiduals(referenceModel_,
			points_,
			estimator_,
			referenceResiduals);

		// Sort the residuals to have an ordering of the points
		std::vector<std::pair<double, size_t>> sortedResiduals;
		sortedResiduals.reserve(pointNumber);

		for (size_t pointIdx = 0; pointIdx < pointNumber; ++pointIdx)
		{
			const double& residual = referenceResiduals[pointIdx];
			if (residual > maximumThreshold)
				continue;

			sortedResiduals.emplace_back(
				std::make_pair(residual, pointIdx));
		}

		std::sort(sortedResiduals.begin(), sortedResiduals.end());

		// Initialize the inlier set
		std::vector<size_t> inliers;
		for (size_t inlierIdx = 0; inlierIdx < minimumInlierNumber - 1; ++inlierIdx)
			inliers.emplace_back(sortedResiduals[inlierIdx].second);
		
		double bestModelDistance = std::numeric_limits<double>::max();

		for (size_t inlierIdx = minimumInlierNumber - 1; inlierIdx < sortedResiduals.size(); ++inlierIdx)
		{
			// Add the next inlier to the set
			inliers.emplace_back(sortedResiduals[inlierIdx].second);

			// The estimator model parameters
			std::vector<gcransac::Model> models;

			// Do least-squares fitting
			estimator_.estimateModelNonminimal(
				points_, // All input points
				&(inliers)[0], // Points which have higher than 0 probability of being inlier
				static_cast<int>(inliers.size()), // Number of possible inliers
				&models, // Estimated models
				nullptr);

			// Calculate the model-to-model residuals
			for (const auto& model : models)
			{
				std::vector<double> residuals;
				getResiduals(model,
					points_,
					estimator_,
					residuals);

				double modelDistance =
					modelToModelDistance(referenceModel_,
						model,
						referenceResiduals,
						residuals);

				if (modelDistance < bestModelDistance)
				{
					bestModelDistance = modelDistance;
					inliers_ = inliers;
					bestThreshold_ = sortedResiduals[inlierIdx].first +
						std::numeric_limits<double>::epsilon();
				}
			}
		}
	}
};

template <class ModelEstimator>
inline double MostSimilarInlierSelector<ModelEstimator>::modelToModelDistance(
	const gcransac::Model& referenceModel_,
	const gcransac::Model& model_,
	const std::vector<double>& referenceResiduals_,
	const std::vector<double>& residuals_) const
{
	const size_t &residualNumber = referenceResiduals_.size();
	double error = 0;

	for (size_t residualIdx = 0; residualIdx < residualNumber; ++residualIdx)
		error += abs(referenceResiduals_[residualIdx] - residuals_[residualIdx]);

	return error;
}

template <class ModelEstimator>
inline void MostSimilarInlierSelector<ModelEstimator>::getResiduals(
	const gcransac::Model& model_,
	const cv::Mat& points_,
	const ModelEstimator& estimator_,
	std::vector<double> &residuals_) const
{
	const size_t& pointNumber = points_.rows;
	residuals_.reserve(pointNumber);

	for (size_t pointIdx = 0; pointIdx < pointNumber; ++pointIdx)
		residuals_.emplace_back(estimator_.squaredResidual(
			points_.row(pointIdx),
			model_));
}