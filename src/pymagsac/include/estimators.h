#pragma once

#include "estimators/essential_estimator.h"
#include "estimators/fundamental_estimator.h"
#include "estimators/homography_estimator.h"
#include "model.h"

namespace magsac
{
	namespace estimator
	{// This is the estimator class for estimating a fundamental matrix between two images. 
		template<class _MinimalSolverEngine,  // The solver used for estimating the model from a minimal sample
			class _NonMinimalSolverEngine> // The solver used for estimating the model from a non-minimal sample
			class HomographyEstimator :
			public gcransac::estimator::RobustHomographyEstimator<_MinimalSolverEngine, _NonMinimalSolverEngine>
		{
		public:
			using gcransac::estimator::RobustHomographyEstimator<_MinimalSolverEngine, _NonMinimalSolverEngine>::residual;

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

			static constexpr bool doesNormalizationForNonMinimalFitting()
			{
				return true;
			}
		};

		// This is the estimator class for estimating a fundamental matrix between two images. 
		template<class _MinimalSolverEngine,  // The solver used for estimating the model from a minimal sample
			class _NonMinimalSolverEngine> // The solver used for estimating the model from a non-minimal sample
			class FundamentalMatrixEstimator :
			public gcransac::estimator::FundamentalMatrixEstimator<_MinimalSolverEngine, _NonMinimalSolverEngine>
		{
		protected:
			using gcransac::estimator::FundamentalMatrixEstimator<_MinimalSolverEngine, _NonMinimalSolverEngine>::use_degensac;
			using gcransac::estimator::FundamentalMatrixEstimator<_MinimalSolverEngine, _NonMinimalSolverEngine>::minimum_inlier_ratio_in_validity_check;
			using gcransac::estimator::FundamentalMatrixEstimator<_MinimalSolverEngine, _NonMinimalSolverEngine>::squared_homography_threshold;

			const double maximum_threshold;

		public:
			using gcransac::estimator::FundamentalMatrixEstimator<_MinimalSolverEngine, _NonMinimalSolverEngine>::squaredSymmetricEpipolarDistance;
			using gcransac::estimator::FundamentalMatrixEstimator<_MinimalSolverEngine, _NonMinimalSolverEngine>::sampleSize;

			FundamentalMatrixEstimator(
				const double maximum_threshold_,
				const double minimum_inlier_ratio_in_validity_check_ = 0.1,
				const bool apply_degensac_ = true,
				const double degensac_homography_threshold_ = 3.0) :
				maximum_threshold(maximum_threshold_),
				gcransac::estimator::FundamentalMatrixEstimator<_MinimalSolverEngine, _NonMinimalSolverEngine>(minimum_inlier_ratio_in_validity_check_,
					apply_degensac_,
					degensac_homography_threshold_)
			{}

			// Calculating the residual which is used for the MAGSAC score calculation.
			// Since symmetric epipolar distance is usually more robust than Sampson-error.
			// we are using it for the score calculation.
			inline double residualForScoring(const cv::Mat& point_,
                const gcransac::Model& model_) const
			{
				return std::sqrt(gcransac::estimator::FundamentalMatrixEstimator<_MinimalSolverEngine, _NonMinimalSolverEngine>::squaredSymmetricEpipolarDistance(point_, model_.descriptor));
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

			static constexpr bool doesNormalizationForNonMinimalFitting()
			{
				return true;
			}

			// Validate the model by checking the number of inlier with symmetric epipolar distance
			// instead of Sampson distance. In general, Sampson distance is more accurate but less
			// robust to degenerate solutions than the symmetric epipolar distance. Therefore,
			// every so-far-the-best model is checked if it has enough inlier with symmetric
			// epipolar distance as well. 
			bool isValidModel(gcransac::Model& model_,
				const cv::Mat& data_,
				const std::vector<size_t> &inliers_,
				const size_t *minimal_sample_,
				const double threshold_,
				bool &model_updated_) const
			{
				// Validate the model by checking the number of inlier with symmetric epipolar distance
				// instead of Sampson distance. In general, Sampson distance is more accurate but less
				// robust to degenerate solutions than the symmetric epipolar distance. Therefore,
				// every so-far-the-best model is checked if it has enough inlier with symmetric
				bool passed = false;
				size_t inlier_number = 0; // Number of inlier if using symmetric epipolar distance
				const Eigen::Matrix3d &descriptor = model_.descriptor; // The decriptor of the current model
				constexpr size_t sample_size = gcransac::estimator::FundamentalMatrixEstimator<_MinimalSolverEngine, _NonMinimalSolverEngine>::sampleSize(); // Size of a minimal sample
				// Minimum number of inliers which should be inlier as well when using symmetric epipolar distance instead of Sampson distance
				const size_t inliers_to_pass = inliers_.size() * gcransac::estimator::FundamentalMatrixEstimator<_MinimalSolverEngine, _NonMinimalSolverEngine>::minimum_inlier_ratio_in_validity_check;
				const size_t minimum_inlier_number =
					MAX(sample_size, inliers_to_pass);
				// Squared inlier-outlier threshold
				const double squared_threshold = threshold_ * threshold_;

				// Iterate through the inliers_ determined by Sampson distance
				for (const auto &idx : inliers_)
					// Calculate the residual using symmetric epipolar distance and check if
					// it is smaller than the threshold_.
					if (gcransac::estimator::FundamentalMatrixEstimator<_MinimalSolverEngine, _NonMinimalSolverEngine>::squaredSymmetricEpipolarDistance(data_.row(idx), descriptor) < squared_threshold)
						// Increase the inlier number and terminate if enough inliers_ have been found.
						if (++inlier_number >= minimum_inlier_number)
						{
							passed = true;
							break;
						}

				// If the fundamental matrix has not passed the symmetric epipolar tests,
				// terminate.
				if (!passed)
					return false;
				 
				// Validate the model by checking if the scene is dominated by a single plane.
				if (gcransac::estimator::FundamentalMatrixEstimator<_MinimalSolverEngine, _NonMinimalSolverEngine>::use_degensac)
					return applyDegensac(model_,
						data_,
						inliers_,
						minimal_sample_,
						threshold_,
						model_updated_);

				// If DEGENSAC is not applied and the model passed the previous tests,
				// assume that it is a valid model.
				return true;
			}

			//  Evaluate the H-degenerate sample test and apply DEGENSAC if needed
			inline bool applyDegensac(gcransac::Model& model_, // The input model to be tested
				const cv::Mat& data_, // All data points
				const std::vector<size_t> &inliers_, // The inliers of the input model
				const size_t *minimal_sample_, // The minimal sample used for estimating the model
				const double threshold_, // The inlier-outlier threshold
				bool &model_updated_) const // A flag saying if the model has been updated here
			{
				// Set the flag initially to false since the model has not been yet updated.
				model_updated_ = false;

				// The possible triplets of points
				constexpr size_t triplets[] = {
					0, 1, 2,
					3, 4, 5,
					0, 1, 6,
					3, 4, 6,
					2, 5, 6 };
				constexpr size_t number_of_triplets = 5; // The number of triplets to be tested
				const size_t columns = data_.cols; // The number of columns in the data matrix

				// The fundamental matrix coming from the minimal sample
				const Eigen::Matrix3d &fundamental_matrix =
					model_.descriptor.block<3, 3>(0, 0);

				// Applying SVD decomposition to the estimated fundamental matrix
				Eigen::JacobiSVD<Eigen::Matrix3d> svd(
					fundamental_matrix,
					Eigen::ComputeFullU | Eigen::ComputeFullV);

				// Calculate the epipole in the second image
				const Eigen::Vector3d epipole =
					svd.matrixU().rightCols<1>().head<3>() / svd.matrixU()(2, 2);

				// The calculate the cross-produced matrix of the epipole
				Eigen::Matrix3d epipolar_cross;
				epipolar_cross << 0, -epipole(2), epipole(1),
					epipole(2), 0, -epipole(0),
					-epipole(1), epipole(0), 0;

				const Eigen::Matrix3d A =
					epipolar_cross * model_.descriptor;

				// A flag deciding if the sample is H-degenerate
				bool h_degenerate_sample = false;
				// The homography which the H-degenerate part of the sample implies
				Eigen::Matrix3d best_homography;
				// Iterate through the triplets of points in the sample
				for (size_t triplet_idx = 0; triplet_idx < number_of_triplets; ++triplet_idx)
				{
					// The index of the first point of the triplet
					const size_t triplet_offset = triplet_idx * 3;
					// The indices of the other points
					const size_t point_1_idx = minimal_sample_[triplets[triplet_offset]],
						point_2_idx = minimal_sample_[triplets[triplet_offset + 1]],
						point_3_idx = minimal_sample_[triplets[triplet_offset + 2]];

					// A pointer to the first point's first coordinate
					const double *point_1_ptr =
						reinterpret_cast<double *>(data_.data) + point_1_idx * columns;
					// A pointer to the second point's first coordinate
					const double *point_2_ptr =
						reinterpret_cast<double *>(data_.data) + point_2_idx * columns;
					// A pointer to the third point's first coordinate
					const double *point_3_ptr =
						reinterpret_cast<double *>(data_.data) + point_3_idx * columns;

					// Copy the point coordinates into Eigen vectors
					Eigen::Vector3d point_1_1, point_1_2, point_1_3,
						point_2_1, point_2_2, point_2_3;

					point_1_1 << point_1_ptr[0], point_1_ptr[1], 1;
					point_2_1 << point_1_ptr[2], point_1_ptr[3], 1;
					point_1_2 << point_2_ptr[0], point_2_ptr[1], 1;
					point_2_2 << point_2_ptr[2], point_2_ptr[3], 1;
					point_1_3 << point_3_ptr[0], point_3_ptr[1], 1;
					point_2_3 << point_3_ptr[2], point_3_ptr[3], 1;

					// Calculate the cross-product of the epipole end each point
					Eigen::Vector3d point_1_cross_epipole = point_2_1.cross(epipole);
					Eigen::Vector3d point_2_cross_epipole = point_2_2.cross(epipole);
					Eigen::Vector3d point_3_cross_epipole = point_2_3.cross(epipole);

					Eigen::Vector3d b;
					b << point_2_1.cross(A * point_1_1).transpose() * point_1_cross_epipole / point_1_cross_epipole.squaredNorm(),
						point_2_2.cross(A * point_1_2).transpose() * point_2_cross_epipole / point_2_cross_epipole.squaredNorm(),
						point_2_3.cross(A * point_1_3).transpose() * point_3_cross_epipole / point_3_cross_epipole.squaredNorm();

					Eigen::Matrix3d M;
					M << point_1_1(0), point_1_1(1), point_1_1(2),
						point_1_2(0), point_1_2(1), point_1_2(2),
						point_1_3(0), point_1_3(1), point_1_3(2);

					Eigen::Matrix3d homography =
						A - epipole * (M.inverse() * b).transpose();

					// The number of point consistent with the implied homography
					size_t inlier_number = 3;

					// Count the inliers of the homography
					for (size_t i = 0; i < sampleSize(); ++i)
					{
						// Get the point's index from the minimal sample
						size_t idx = minimal_sample_[i];

						// Check if the point is not included in the current triplet
						if (idx == point_1_idx ||
							idx == point_2_idx ||
							idx == point_3_idx)
							continue; // If yes, the error does not have to be calculated

						// Calculate the re-projection error
						const double *point_ptr =
							reinterpret_cast<double *>(data_.data) + idx * columns;

						const double &x1 = point_ptr[0], // The x coordinate in the first image
							&y1 = point_ptr[1], // The y coordinate in the first image
							&x2 = point_ptr[2], // The x coordinate in the second image
							&y2 = point_ptr[3]; // The y coordinate in the second image

						// Calculating H * p
						const double t1 = homography(0, 0) * x1 + homography(0, 1) * y1 + homography(0, 2),
							t2 = homography(1, 0) * x1 + homography(1, 1) * y1 + homography(1, 2),
							t3 = homography(2, 0) * x1 + homography(2, 1) * y1 + homography(2, 2);

						// Calculating the difference of the projected and original points
						const double d1 = x2 - (t1 / t3),
							d2 = y2 - (t2 / t3);

						// Calculating the squared re-projection error
						const double squared_residual = d1 * d1 + d2 * d2;

						// If the squared re-projection error is smaller than the threshold, 
						// consider the point inlier.
						if (squared_residual < squared_homography_threshold)
							++inlier_number;
					}

					// If at least 5 points are correspondences are consistent with the homography,
					// consider the sample as H-degenerate.
					if (inlier_number >= 5)
					{
						// Saving the parameters of the homography
						best_homography = homography;
						// Setting the flag of being a h-degenerate sample
						h_degenerate_sample = true;
						break;
					}
				}

				// If the sample is H-degenerate
				if (h_degenerate_sample)
				{
					// Declare a homography estimator to be able to calculate the residual and the homography from a non-minimal sample
					static const magsac::estimator::HomographyEstimator<
						gcransac::estimator::solver::HomographyFourPointSolver, // The solver used for fitting a model to a minimal sample
						gcransac::estimator::solver::HomographyFourPointSolver> homography_estimator;

					// The inliers of the homography
					std::vector<size_t> homography_inliers;
					homography_inliers.reserve(inliers_.size());

					// Iterate through the inliers of the fundamental matrix
					// and select those which are inliers of the homography as well.
					//for (size_t inlier_idx = 0; inlier_idx < data_.rows; ++inlier_idx)
					for (const size_t &inlier_idx : inliers_)
						if (homography_estimator.squaredResidual(data_.row(inlier_idx), best_homography) < squared_homography_threshold)
							homography_inliers.emplace_back(inlier_idx);

					// If the homography does not have enough inliers to be estimated, terminate.
					if (homography_inliers.size() < homography_estimator.nonMinimalSampleSize())
						return false;

					// The set of estimated homographies. For all implemented solvers,
					// this should be of size 1.
					std::vector<gcransac::Model> homographies;

					// Estimate the homography parameters from the provided inliers.
					homography_estimator.estimateModelNonminimal(data_, // All data points
						&homography_inliers[0], // The inliers of the homography
						homography_inliers.size(), // The number of inliers
						&homographies); // The estimated homographies

					// If the number of estimated homographies is not 1, there is some problem
					// and, thus, terminate.
					if (homographies.size() != 1)
						return false;

					// Get the reference of the homography fit to the non-minimal sample
					const Eigen::Matrix3d &nonminimal_homography =
						homographies[0].descriptor;

					// Do a local GC-RANSAC to determine the parameters of the fundamental matrix by
					// the plane-and-parallax algorithm using the determined homography.
					magsac::estimator::FundamentalMatrixEstimator<
						gcransac::estimator::solver::FundamentalMatrixPlaneParallaxSolver, // The solver used for fitting a model to a minimal sample
						gcransac::estimator::solver::FundamentalMatrixEightPointSolver> estimator(maximum_threshold, 0.0, false);
					estimator.getMinimalSolver()->setHomography(&nonminimal_homography);

					std::vector<int> inliers;
					gcransac::Model model;

					MAGSAC<cv::Mat, magsac::estimator::FundamentalMatrixEstimator<
						gcransac::estimator::solver::FundamentalMatrixPlaneParallaxSolver, // The solver used for fitting a model to a minimal sample
						gcransac::estimator::solver::FundamentalMatrixEightPointSolver>> magsac;
					magsac.setMaximumThreshold(maximum_threshold); // The maximum noise scale sigma allowed
					magsac.setReferenceThreshold(threshold_);
					magsac.setIterationLimit(1e4); // Iteration limit to interrupt the cases when the algorithm run too long.

					gcransac::sampler::UniformSampler sampler(&data_); // The local optimization sampler is used inside the local optimization

					int iteration_number = 0; // Number of iterations required
					ModelScore score;
					const bool success = magsac.run(data_, // The data points
						0.99, // The required confidence in the results
						estimator, // The used estimator
						sampler, // The sampler used for selecting minimal samples in each iteration
						model, // The estimated model
						iteration_number, // The number of iterations
						score);
					
					// If more inliers are found the what initially was given,
					// update the model parameters.
					if (score.inlier_number >= inliers_.size())
					{
						// Consider the model to be updated
						model_updated_ = true;
						// Update the parameters
						model_ = model;	
					}
				}

				// If we get to this point, the procedure was successfull
				return true;
			}
		};

		// This is the estimator class for estimating a fundamental matrix between two images. 
		template<class _MinimalSolverEngine,  // The solver used for estimating the model from a minimal sample
			class _NonMinimalSolverEngine> // The solver used for estimating the model from a non-minimal sample
			class EssentialMatrixEstimator :
			public gcransac::estimator::EssentialMatrixEstimator<_MinimalSolverEngine, _NonMinimalSolverEngine>
		{
		public:
			using gcransac::estimator::EssentialMatrixEstimator<_MinimalSolverEngine, _NonMinimalSolverEngine>::squaredSymmetricEpipolarDistance;

			EssentialMatrixEstimator(Eigen::Matrix3d intrinsics_src_, // The intrinsic parameters of the source camera
				Eigen::Matrix3d intrinsics_dst_,  // The intrinsic parameters of the destination camera
				const double minimum_inlier_ratio_in_validity_check_ = 0.1,
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

			static constexpr double getC()
			{
				return 0.25;
			}

			static constexpr double getGammaFunction()
			{
				return 1.0;
			}

			static constexpr bool doesNormalizationForNonMinimalFitting()
			{
				return false;
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
	}

	namespace utils
	{
		// The default estimator for essential matrix fitting
		typedef estimator::EssentialMatrixEstimator<gcransac::estimator::solver::EssentialMatrixFivePointSteweniusSolver, // The solver used for fitting a model to a minimal sample
			gcransac::estimator::solver::EssentialMatrixBundleAdjustmentSolver> // The solver used for fitting a model to a non-minimal sample
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
