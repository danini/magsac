#pragma once

#include "essential_estimator.h"
#include "fundamental_estimator.h"
#include "homography_estimator.h"
#include "model.h"
namespace magsac
{
	namespace estimator
	{
		// This is the estimator class for estimating a fundamental matrix between two images. 
		template<class _MinimalSolverEngine,  // The solver used for estimating the model from a minimal sample
			class _NonMinimalSolverEngine> // The solver used for estimating the model from a non-minimal sample
			class FundamentalMatrixEstimator :
			public gcransac::estimator::FundamentalMatrixEstimator<_MinimalSolverEngine, _NonMinimalSolverEngine>
		{
		public:
			FundamentalMatrixEstimator(const double minimum_inlier_ratio_in_validity_check_ = 0.5) :
				gcransac::estimator::FundamentalMatrixEstimator<_MinimalSolverEngine, _NonMinimalSolverEngine>(minimum_inlier_ratio_in_validity_check_)
			{}

			// Calculating the residual which is used for the MAGSAC score calculation.
			// Since symmetric epipolar distance is usually more robust than Sampson-error.
			// we are using it for the score calculation.
			inline double residualForScoring(const cv::Mat& point_,
                const gcransac::Model& model_) const
			{
				return std::sqrt(squaredSymmetricEpipolarDistance(point_, model_.descriptor));
			}

			static constexpr double getSigmaQuantile()
			{
				return 3.64;
			}

			static constexpr size_t getDegreesOfFreedom()
			{
				return 4;
			}

			static constexpr double getGammaFunction()
			{
				return 1.0;
			}

			static constexpr double getC()
			{
				return 0.25;
			}

			// Calculating the upper incomplete gamma value of (DoF - 1) / 2 with k^2 / 2.
			static constexpr double getUpperIncompleteGammaOfK()
			{
				return 0.0036572608340910764;
			}

			// Calculating the lower incomplete gamma value of (DoF + 1) / 2 with k^2 / 2.
			static constexpr double getLowerIncompleteGammaOfK()
			{
				return 1.3012265540498875;
			}

			static constexpr double getChiSquareParamCp()
			{
				return 1.0 / (4.0 * getGammaFunction());
			}
		};

		// This is the estimator class for estimating a fundamental matrix between two images. 
		template<class _MinimalSolverEngine,  // The solver used for estimating the model from a minimal sample
			class _NonMinimalSolverEngine> // The solver used for estimating the model from a non-minimal sample
			class EssentialMatrixEstimator :
			public gcransac::estimator::EssentialMatrixEstimator<_MinimalSolverEngine, _NonMinimalSolverEngine>
		{
		public:
			EssentialMatrixEstimator(Eigen::Matrix3d intrinsics_src_, // The intrinsic parameters of the source camera
				Eigen::Matrix3d intrinsics_dst_,  // The intrinsic parameters of the destination camera
				const double minimum_inlier_ratio_in_validity_check_ = 0.5,
				const double point_ratio_for_selecting_from_multiple_models_ = 0.05) :
				gcransac::estimator::EssentialMatrixEstimator<_MinimalSolverEngine, _NonMinimalSolverEngine>(
					intrinsics_src_,
					intrinsics_dst_,
					minimum_inlier_ratio_in_validity_check_,
					point_ratio_for_selecting_from_multiple_models_)
			{}

			// Calculating the residual which is used for the MAGSAC score calculation.
			// Since symmetric epipolar distance is usually more robust than Sampson-error.
			// we are using it for the score calculation.
			inline double residualForScoring(const cv::Mat& point_,
                const gcransac::Model& model_) const
			{
				return squaredSymmetricEpipolarDistance(point_, model_.descriptor);
			}

			static constexpr double getSigmaQuantile()
			{
				return 3.64;
			}
			
			static constexpr size_t getDegreesOfFreedom()
			{
				return 4;
			}

			static constexpr double getC()
			{
				return 0.25;
			}

			static constexpr double getGammaFunction()
			{
				return 1.0;
			}

			// Calculating the upper incomplete gamma value of (DoF - 1) / 2 with k^2 / 2.
			static constexpr double getUpperIncompleteGammaOfK()
			{
				return 0.0036572608340910764;
			}

			// Calculating the lower incomplete gamma value of (DoF + 1) / 2 with k^2 / 2.
			static constexpr double getLowerIncompleteGammaOfK()
			{
				return 1.3012265540498875;
			}

			static constexpr double getChiSquareParamCp()
			{
				return 1.0 / (4.0 * getGammaFunction());
			}
		};

		// This is the estimator class for estimating a fundamental matrix between two images. 
		template<class _MinimalSolverEngine,  // The solver used for estimating the model from a minimal sample
			class _NonMinimalSolverEngine> // The solver used for estimating the model from a non-minimal sample
			class HomographyEstimator :
			public gcransac::estimator::RobustHomographyEstimator<_MinimalSolverEngine, _NonMinimalSolverEngine>
		{
		public:
			HomographyEstimator() :
				gcransac::estimator::RobustHomographyEstimator<_MinimalSolverEngine, _NonMinimalSolverEngine>()
			{}

			// Calculating the residual which is used for the MAGSAC score calculation.
			// Since symmetric epipolar distance is usually more robust than Sampson-error.
			// we are using it for the score calculation.
			inline double residualForScoring(const cv::Mat& point_,
                const gcransac::Model& model_) const
			{
				return residual(point_, model_.descriptor);
			}
			
			static constexpr double getSigmaQuantile()
			{
				return 3.64;
			}

			static constexpr size_t getDegreesOfFreedom()
			{
				return 4;
			}

			static constexpr double getC()
			{
				return 0.25;
			}

			// Calculating the upper incomplete gamma value of (DoF - 1) / 2 with k^2 / 2.
			static constexpr double getGammaFunction()
			{
				return 1.0;
			}

			static constexpr double getUpperIncompleteGammaOfK()
			{
				return 0.0036572608340910764;
			}

			// Calculating the lower incomplete gamma value of (DoF + 1) / 2 with k^2 / 2.
			static constexpr double getLowerIncompleteGammaOfK()
			{
				return 1.3012265540498875;
			}

			static constexpr double getChiSquareParamCp()
			{
				return 1.0 / (4.0 * getGammaFunction());
			}
		};
	}

	namespace utils
	{
		// The default estimator for essential matrix fitting
		typedef estimator::EssentialMatrixEstimator<gcransac::estimator::solver::EssentialMatrixFivePointSteweniusSolver, // The solver used for fitting a model to a minimal sample
			gcransac::estimator::solver::EssentialMatrixFivePointSteweniusSolver> // The solver used for fitting a model to a non-minimal sample
			DefaultEssentialMatrixEstimator;

		// The default estimator for fundamental matrix fitting
		typedef estimator::FundamentalMatrixEstimator<gcransac::estimator::solver::FundamentalMatrixSevenPointSolver, // The solver used for fitting a model to a minimal sample
			gcransac::estimator::solver::FundamentalMatrixEightPointSolver> // The solver used for fitting a model to a non-minimal sample
			DefaultFundamentalMatrixEstimator;

		// The default estimator for homography fitting
		typedef estimator::HomographyEstimator<gcransac::estimator::solver::HomographyFourPointSolver, // The solver used for fitting a model to a minimal sample
			gcransac::estimator::solver::HomographyFourPointSolver> // The solver used for fitting a model to a non-minimal sample
			DefaultHomographyEstimator;
	}
}
