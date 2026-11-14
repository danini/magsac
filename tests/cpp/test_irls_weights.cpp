// Test of the weights of MAGSAC++'s iteratively reweighted least squares
// (MAGSAC::sigmaConsensusPlusPlus). The MAGSAC++ paper gives the weight of a
// point with residual r in closed form; with the 2D line estimator's constants
// (degrees of freedom 4, so the upper incomplete gamma function of order 3/2):
//   w(r) = C 2^(3/2) / sigma (G(r^2 / (2 sigma^2)) - G(k^2 / 2)),  sigma = threshold / k,
//   G(x) = sqrt(x) exp(-x) + sqrt(pi) / 2 erfc(sqrt(x)).
// The test records the weights that the first fit receives, and compares them with
// w(r). Exits with 0 if they match.
#include <cmath>
#include <cstdio>
#include <vector>
#include <opencv2/core/core.hpp>
#include "magsac.h"
#include "estimators/solver_essential_matrix_five_point_stewenius.h"
#include "estimators/solver_essential_matrix_bundle_adjustment.h"
#include "utils.h"
#include "estimators.h"

using BaseEstimator = magsac::utils::Default2DLineEstimator;

// The line estimator, which records the weights of its first non-minimal fit
class RecordingLineEstimator : public BaseEstimator
{
public:
	mutable std::vector<double> weights;

	bool estimateModelNonminimal(const cv::Mat &data_, const size_t *sample_, const size_t &sample_number_,
		std::vector<gcransac::Model> *models_, const double *weights_ = nullptr) const
	{
		// This estimator gets one weight per point of data_ (doesNormalizationForNonMinimalFitting() is false)
		if (weights.empty() && weights_ != nullptr)
			weights.assign(weights_, weights_ + data_.rows);
		return BaseEstimator::estimateModelNonminimal(data_, sample_, sample_number_, models_, weights_);
	}
};

// Gives access to the protected sigmaConsensusPlusPlus
class TestMAGSAC : public MAGSAC<cv::Mat, RecordingLineEstimator>
{
public:
	using MAGSAC<cv::Mat, RecordingLineEstimator>::sigmaConsensusPlusPlus;
};

static double upperGamma32(double x)
{
	return std::sqrt(x) * std::exp(-x) + std::sqrt(M_PI) / 2.0 * std::erfc(std::sqrt(x));
}

int main()
{
	static_assert(!RecordingLineEstimator::doesNormalizationForNonMinimalFitting(), "weights are indexed by point");
	static_assert(RecordingLineEstimator::getDegreesOfFreedom() == 4, "w(r) above is for 4 degrees of freedom");
	const double k = RecordingLineEstimator::getSigmaQuantile();
	const double C = RecordingLineEstimator::getC();

	int failures = 0;
	for (double threshold : {0.5, 2.0, 10.0})
	{
		// Points around the line y = 0, at residuals from 0.05 to 0.995 times the threshold
		const std::vector<double> fractions = {0.05, -0.25, 0.5, -0.75, 0.95, -0.995};
		cv::Mat points(static_cast<int>(fractions.size()), 2, CV_64F);
		for (int i = 0; i < points.rows; ++i)
		{
			points.at<double>(i, 0) = 10.0 * i;
			points.at<double>(i, 1) = fractions[i] * threshold;
		}
		gcransac::Model model, refined_model;
		model.descriptor.resize(3, 1);
		model.descriptor << 0, 1, 0;

		RecordingLineEstimator estimator;
		TestMAGSAC magsac;
		magsac.setMaximumThreshold(threshold);
		magsac.number_of_irwls_iters = 1;
		ModelScore score, best_score;
		magsac.sigmaConsensusPlusPlus(points, model, refined_model, score, estimator, best_score);
		if (estimator.weights.size() != fractions.size())
		{
			std::printf("FAIL threshold %g: the weighted fit wasn't called\n", threshold);
			++failures;
			continue;
		}

		const double sigma = threshold / k;
		const double scale = C * std::pow(2.0, 1.5) / sigma;
		const double weight_zero = scale * (upperGamma32(0.0) - upperGamma32(k * k / 2.0));
		for (int i = 0; i < points.rows; ++i)
		{
			const double r = std::abs(fractions[i]) * threshold;
			const double expected = scale * (upperGamma32(r * r / (2.0 * sigma * sigma)) - upperGamma32(k * k / 2.0));
			// The weights come from a lookup table of the gamma function: allow for its step
			const bool ok = std::abs(estimator.weights[i] - expected) < 1e-4 * weight_zero;
			failures += !ok;
			std::printf("%s threshold %g, residual %g: weight / w(0) %.7f, expected %.7f\n", ok ? "ok  " : "FAIL",
				threshold, r, estimator.weights[i] / weight_zero, expected / weight_zero);
		}
	}
	return failures == 0 ? 0 : 1;
}
