#pragma once

#include <opencv2/core/core.hpp>
#include "model.h"

namespace magsac
{
	namespace estimator
	{
		// Adds what MAGSAC needs to a graph-cut-ransac estimator: the residual used for scoring
		// and the constants of the chi-square distribution, which are those of the estimators in
		// estimators.h (4 degrees of freedom). _NormalizesForNonMinimalFitting tells whether the
		// estimator's non-minimal fit copies the points it uses (so it reads one weight per used
		// point) or reads them from the full data (one weight per data point).
		template<class _Estimator, bool _NormalizesForNonMinimalFitting>
		class Wrapper : public _Estimator
		{
		public:
			using _Estimator::_Estimator;
			using _Estimator::residual;

			inline double residualForScoring(const cv::Mat& point_,
				const gcransac::Model& model_) const
			{
				return residual(point_, model_.descriptor);
			}

			static constexpr double getSigmaQuantile() { return 3.64; }
			static constexpr size_t getDegreesOfFreedom() { return 4; }
			static constexpr double getC() { return 0.25; }
			static constexpr double getGammaFunction() { return 1.0; }
			static constexpr bool doesNormalizationForNonMinimalFitting() { return _NormalizesForNonMinimalFitting; }
			// The upper incomplete gamma value of (DoF - 1) / 2 with k^2 / 2.
			static constexpr double getUpperIncompleteGammaOfK() { return 0.0036572608340910764; }
			// The lower incomplete gamma value of (DoF + 1) / 2 with k^2 / 2.
			static constexpr double getLowerIncompleteGammaOfK() { return 1.3012265540498875; }
			static constexpr double getChiSquareParamCp() { return 1.0 / (4.0 * getGammaFunction()); }
		};
	}
}
