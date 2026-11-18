// Test of the inlier count that MAGSAC::sigmaConsensusPlusPlus returns with the
// refined model: score.inlier_number must be the number of points closer to the
// refined model than both the maximum and the reference thresholds (as for the
// input model), not that of the model it started from. The points lie near the
// line y = 0, and the initial model, y = 0.02 x, is close to only part of them.
// A second case has a reference threshold above the maximum one, and ten outliers
// between the two. Exits with 0 if the counts are right.
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
	// 100 points at x = 0..99, alternately 0.1 above and below the line y = 0, then
	// 10 outliers at x = 5, 15, ..., 95 and y = -0.75 (used by the second case only)
	cv::Mat all_points(110, 2, CV_64F);
	for (int i = 0; i < 100; ++i)
	{
		all_points.at<double>(i, 0) = i;
		all_points.at<double>(i, 1) = i % 2 ? 0.1 : -0.1;
	}
	for (int i = 0; i < 10; ++i)
	{
		all_points.at<double>(100 + i, 0) = 10 * i + 5;
		all_points.at<double>(100 + i, 1) = -0.75;
	}
	// The line 0.02 x - y = 0, normalized
	gcransac::Model model;
	model.descriptor.resize(3, 1);
	model.descriptor << 0.02, -1, 0;
	model.descriptor /= model.descriptor.topRows<2>().norm();

	struct Case { int point_number; double maximum_threshold, reference_threshold; };
	Estimator estimator;
	int failures = 0;
	for (const Case &c : { Case{ 100, 10.0, 1.0 }, Case{ 110, 0.5, 1.0 } })
	{
		const cv::Mat points = all_points.rowRange(0, c.point_number);
		const double count_threshold = std::min(c.maximum_threshold, c.reference_threshold);
		for (size_t irls_iterations : {1, 3})
		{
			TestMAGSAC magsac;
			magsac.setMaximumThreshold(c.maximum_threshold);
			magsac.setReferenceThreshold(c.reference_threshold);
			magsac.number_of_irwls_iters = irls_iterations;
			gcransac::Model refined_model;
			ModelScore score, best_score;
			if (!magsac.sigmaConsensusPlusPlus(points, model, refined_model, score, estimator, best_score))
			{
				std::printf("FAIL maximum threshold %g, reference %g, %zu IRLS iterations: no refined model\n",
					c.maximum_threshold, c.reference_threshold, irls_iterations);
				++failures;
				continue;
			}
			const size_t initial = countClose(points, model, estimator, count_threshold);
			const size_t refined = countClose(points, refined_model, estimator, count_threshold);
			const bool ok = score.inlier_number == refined;
			failures += !ok;
			std::printf("%s maximum threshold %g, reference %g, %zu IRLS iterations: score.inlier_number %zu, "
				"points close to the refined model %zu (to the initial model %zu)\n", ok ? "ok  " : "FAIL",
				c.maximum_threshold, c.reference_threshold, irls_iterations, score.inlier_number, refined, initial);
		}
	}
	return failures == 0 ? 0 : 1;
}
