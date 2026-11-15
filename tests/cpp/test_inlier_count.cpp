// Test of the inlier count that MAGSAC::sigmaConsensusPlusPlus returns with the
// refined model: score.inlier_number must be the number of points closer to the
// refined model than the reference threshold, not that of the model it started
// from. The points lie near the line y = 0, and the initial model, y = 0.02 x,
// is close to only part of them. Exits with 0 if the count is right.
#include <cmath>
#include <cstdio>
#include <vector>
#include <opencv2/core/core.hpp>
#include "magsac.h"
#include "estimators/solver_essential_matrix_five_point_stewenius.h"
#include "estimators/solver_essential_matrix_bundle_adjustment.h"
#include "utils.h"
#include "estimators.h"

using Estimator = magsac::utils::Default2DLineEstimator;

// Gives access to the protected sigmaConsensusPlusPlus
class TestMAGSAC : public MAGSAC<cv::Mat, Estimator>
{
public:
	using MAGSAC<cv::Mat, Estimator>::sigmaConsensusPlusPlus;
};

// The number of points closer to the model than the threshold
static size_t countClose(const cv::Mat &points, const gcransac::Model &model, const Estimator &estimator, double threshold)
{
	size_t count = 0;
	for (int i = 0; i < points.rows; ++i)
		count += estimator.residual(points.row(i), model) < threshold;
	return count;
}

int main()
{
	// 100 points at x = 0..99, alternately 0.1 above and below the line y = 0
	cv::Mat points(100, 2, CV_64F);
	for (int i = 0; i < points.rows; ++i)
	{
		points.at<double>(i, 0) = i;
		points.at<double>(i, 1) = i % 2 ? 0.1 : -0.1;
	}
	// The line 0.02 x - y = 0, normalized
	gcransac::Model model, refined_model;
	model.descriptor.resize(3, 1);
	model.descriptor << 0.02, -1, 0;
	model.descriptor /= model.descriptor.topRows<2>().norm();

	Estimator estimator;
	TestMAGSAC magsac;
	magsac.setMaximumThreshold(10.0);
	magsac.setReferenceThreshold(1.0);
	int failures = 0;
	for (size_t irls_iterations : {1, 3})
	{
		magsac.number_of_irwls_iters = irls_iterations;
		ModelScore score, best_score;
		if (!magsac.sigmaConsensusPlusPlus(points, model, refined_model, score, estimator, best_score))
		{
			std::printf("FAIL %zu IRLS iterations: no refined model\n", irls_iterations);
			++failures;
			continue;
		}
		const size_t initial = countClose(points, model, estimator, 1.0);
		const size_t refined = countClose(points, refined_model, estimator, 1.0);
		const bool ok = score.inlier_number == refined;
		failures += !ok;
		std::printf("%s %zu IRLS iterations: score.inlier_number %zu, points close to the refined model %zu "
			"(to the initial model %zu)\n", ok ? "ok  " : "FAIL", irls_iterations, score.inlier_number, refined, initial);
	}
	return failures == 0 ? 0 : 1;
}
