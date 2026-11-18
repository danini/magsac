// Test of the weights of the final fit of MAGSAC::sigmaConsensus (the original
// MAGSAC) for an estimator that doesn't normalize the points of its non-minimal
// fits (doesNormalizationForNonMinimalFitting() is false, as for essential
// matrices): the solvers read the weight of point i at weights[i], so the weights
// must be stored by point, not by the rank of the point's residual.
// The points lie near the line y = 0, the first ones far from it (outliers).
// Checks that the points of the final fit, and only them, have a weight, and that
// reversing the order of the points doesn't change the weight of any point.
// Exits with 0 if both hold.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
#include <opencv2/core/core.hpp>
#include "magsac.h"
#include "estimators/solver_essential_matrix_five_point_stewenius.h"
#include "estimators/solver_essential_matrix_bundle_adjustment.h"
#include "utils.h"
#include "estimators.h"

// The 2D line estimator, which records the sample and the weights of its weighted fits
class RecordingEstimator : public magsac::utils::Default2DLineEstimator
{
public:
	static_assert(!doesNormalizationForNonMinimalFitting(), "the test needs weights indexed by point");
	mutable std::vector<size_t> sample;
	mutable std::vector<double> weights;

	bool estimateModelNonminimal(const cv::Mat &data_, const size_t *sample_, const size_t &sample_number_,
		std::vector<gcransac::Model> *models_, const double *weights_ = nullptr) const
	{
		if (weights_ != nullptr) // the final fit; the others run in parallel, without weights
		{
			sample.assign(sample_, sample_ + sample_number_);
			weights.assign(weights_, weights_ + data_.rows); // one per point
		}
		return magsac::utils::Default2DLineEstimator::estimateModelNonminimal(data_, sample_, sample_number_, models_, weights_);
	}
};

// Gives access to the protected sigmaConsensus
class TestMAGSAC : public MAGSAC<cv::Mat, RecordingEstimator>
{
public:
	TestMAGSAC() : MAGSAC<cv::Mat, RecordingEstimator>(MAGSAC_ORIGINAL) {}
	using MAGSAC<cv::Mat, RecordingEstimator>::sigmaConsensus;
};

constexpr int outlier_number = 20, point_number = 70;

// Runs sigmaConsensus on the points, listed in increasing or in reverse order,
// and returns the weights of the final fit, in increasing order of the points.
// Returns false if a point of the final fit has no weight, or another point has one.
static bool weightsOfPoints(bool reversed, std::vector<double> &weights)
{
	// Point i: x = i, at distance 50 + i from the line y = 0 for the first ones, below 1 for the others
	cv::Mat points(point_number, 2, CV_64F);
	for (int i = 0; i < point_number; ++i)
	{
		const int row = reversed ? point_number - 1 - i : i;
		const double distance = i < outlier_number ? 50.0 + i : 0.01 + 0.9 * ((i * 37) % 50) / 50.0;
		points.at<double>(row, 0) = i;
		points.at<double>(row, 1) = i % 2 ? distance : -distance;
	}
	gcransac::Model model, refined_model; // the line y = 0
	model.descriptor.resize(3, 1);
	model.descriptor << 0, 1, 0;

	RecordingEstimator estimator;
	TestMAGSAC magsac;
	magsac.setMaximumThreshold(10.0);
	magsac.setReferenceThreshold(1.0);
	ModelScore score, best_score;
	if (!magsac.sigmaConsensus(points, model, refined_model, score, estimator, best_score) || estimator.weights.empty())
	{
		std::printf("FAIL: no weighted fit\n");
		return false;
	}

	weights.assign(point_number, 0.0);
	std::vector<bool> in_sample(point_number, false);
	for (const size_t row : estimator.sample)
		in_sample[row] = true;
	bool ok = true;
	for (int row = 0; row < point_number; ++row)
	{
		const int i = reversed ? point_number - 1 - row : row;
		weights[i] = estimator.weights[row];
		if (in_sample[row] != (weights[i] > 0))
		{
			std::printf("FAIL (%s order): point %d is %sin the final fit, and has weight %g\n",
				reversed ? "reverse" : "increasing", i, in_sample[row] ? "" : "not ", weights[i]);
			ok = false;
		}
	}
	return ok;
}

int main()
{
	std::vector<double> forward, backward;
	bool ok = weightsOfPoints(false, forward);
	ok &= weightsOfPoints(true, backward);

	// Largest change of a point's weight, relative to the largest weight
	double largest_change = 0;
	for (int i = 0; i < point_number; ++i)
		largest_change = std::max(largest_change, std::abs(forward[i] - backward[i]));
	largest_change /= *std::max_element(forward.begin(), forward.end());
	const bool same = largest_change < 1e-9;
	std::printf("%s: largest change of a point's weight when the points are reversed: %.3g (of the largest weight)\n",
		same ? "ok  " : "FAIL", largest_change);
	return ok && same ? 0 : 1;
}
