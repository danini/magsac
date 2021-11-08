#pragma once

#include <limits>
#include <chrono>
#include <memory>
#include "model.h"
#include "model_score.h"
#include "samplers/sampler.h"
#include "samplers/uniform_sampler.h"
#include <math.h> 
#include "gamma_values.cpp"

#ifdef _WIN32 
	#include <ppl.h>
#endif

#include <gflags/gflags.h>
#include <glog/logging.h>

template <class DatumType, class ModelEstimator>
class MAGSAC  
{
public:
	enum Version { 
		// The original version of MAGSAC. It works well, however, can be quite slow in many cases.
		MAGSAC_ORIGINAL, 
		// The recently proposed MAGSAC++ algorithm which keeps the accuracy of the original MAGSAC but is often orders of magnitude faster.
		MAGSAC_PLUS_PLUS }; 

	MAGSAC(const Version magsac_version_ = Version::MAGSAC_PLUS_PLUS) :
		time_limit(std::numeric_limits<double>::max()), // 
		desired_fps(-1),
		iteration_limit(std::numeric_limits<size_t>::max()),
		maximum_threshold(10.0),
		apply_post_processing(true),
		mininum_iteration_number(50),
		partition_number(5),
		core_number(1),
		number_of_irwls_iters(1),
		interrupting_threshold(1.0),
		last_iteration_number(0),
		log_confidence(0),
		point_number(0),
		save_samples(false),
		magsac_version(magsac_version_)
	{ 
	}

	~MAGSAC() {}

	// A function to run MAGSAC.
	bool run(
		const cv::Mat &points_, // The input data points
		const double confidence_, // The required confidence in the results
		ModelEstimator& estimator_, // The model estimator
		gcransac::sampler::Sampler<cv::Mat, size_t> &sampler_, // The sampler used
		gcransac::Model &obtained_model_, // The estimated model parameters
		int &iteration_number_, // The number of iterations done
		ModelScore &model_score_); // The score of the estimated model
		
	// A function to set the maximum inlier-outlier threshold 
	void setMaximumThreshold(const double maximum_threshold_) 
	{
		maximum_threshold = maximum_threshold_;
	}

	void setSampleSavingFlag(const bool save_samples_)
	{
		save_samples = save_samples_;	
	}

	// A function to set the inlier-outlier threshold used for speeding up the procedure
	// and for determining the required number of iterations.
	void setReferenceThreshold(const double threshold_)
	{
		interrupting_threshold = threshold_;
	}

	double getReferenceThreshold()
	{
		return interrupting_threshold;
	}

	const std::vector<std::vector<size_t>> &getMinimalSamples() const
	{
		return minimal_samples;
	}

	std::vector<std::vector<size_t>> &getMutableMinimalSamples()
	{
		return minimal_samples;
	}

	// Setting the flag determining if post-processing is needed
	void applyPostProcessing(bool value_) 
	{
		apply_post_processing = value_;
	}

	// A function to set the maximum number of iterations
	void setIterationLimit(size_t iteration_limit_)
	{
		iteration_limit = iteration_limit_;
	}

	// A function to set the minimum number of iterations
	void setMinimumIterationNumber(size_t mininum_iteration_number_)
	{
		mininum_iteration_number = mininum_iteration_number_;
	}

	// A function to set the number of cores used in the original MAGSAC algorithm.
	// In MAGSAC++, it is not used. Note that when multiple MAGSACs run in parallel,
	// it is beneficial to keep the core number one for each independent MAGSAC.
	// Otherwise, the threads will act weirdly.
	void setCoreNumber(size_t core_number_)
	{
		//if (magsac_version == MAGSAC_PLUS_PLUS)
		//	LOG(INFO) << "Setting the core number for MAGSAC++ is deprecated.";
		core_number = core_number_;
	}

	// Setting the number of partitions used in the original MAGSAC algorithm
	// to speed up the procedure. In MAGSAC++, this parameter is not used.
	void setPartitionNumber(size_t partition_number_)
	{
		//if (magsac_version == MAGSAC_PLUS_PLUS)
		//	LOG(INFO) << "Setting the partition number for MAGSAC++ is deprecated.";
		partition_number = partition_number_;
	}

	// A function to set a desired minimum frames-per-second (FPS) value.
	void setFPS(int fps_) 
	{ 
		desired_fps = fps_; // The required FPS.
		// The time limit which the FPS implies
		time_limit = fps_ <= 0 ? 
			std::numeric_limits<double>::max() : 
			1.0 / fps_;
	}

	// The post-processing algorithm applying sigma-consensus to the input model once.
	bool postProcessing(
		const cv::Mat &points, // All data points
		const gcransac::Model &so_far_the_best_model, // The input model to be improved
		gcransac::Model &output_model, // The improved model parameters
		ModelScore &output_score, // The score of the improved model
		const ModelEstimator &estimator); // The model estimator

	// The function determining the quality/score of a model using the original MAGSAC
	// criterion. Note that this function is significantly slower than the quality
	// function of MAGSAC++.
	void getModelQuality(
		const cv::Mat& points_, // All data points
		const gcransac::Model& model_, // The input model
		const ModelEstimator& estimator_, // The model estimator
		double& marginalized_iteration_number_, // The required number of iterations marginalized over the noise scale
		double& score_); // The score/quality of the model

	// The function determining the quality/score of a 
	// model using the MAGSAC++ criterion.
	void getModelQualityPlusPlus(
		const cv::Mat &points_, // All data points
		const gcransac::Model &model_, // The model parameter
		const ModelEstimator &estimator_, // The model estimator class
		double &score_, // The score to be calculated
		const double &previous_best_score_); // The score of the previous so-far-the-best model

	size_t number_of_irwls_iters;
protected:
	Version magsac_version; // The version of MAGSAC used
	size_t iteration_limit; // Maximum number of iterations allowed
	size_t mininum_iteration_number; // Minimum number of iteration before terminating
	double maximum_threshold; // The maximum sigma value
	size_t core_number; // Number of core used in sigma-consensus
	double time_limit; // A time limit after the algorithm is interrupted
	int desired_fps; // The desired FPS (TODO: not tested with MAGSAC)
	bool apply_post_processing; // Decides if the post-processing step should be applied
	int point_number; // The current point number
	int last_iteration_number; // The iteration number implied by the last run of sigma-consensus
	double log_confidence; // The logarithm of the required confidence
	size_t partition_number; // Number of partitions used to speed up sigma-consensus
	double interrupting_threshold; // A threshold to speed up MAGSAC by interrupting the sigma-consensus procedure whenever there is no chance of being better than the previous so-far-the-best model
	
	bool save_samples;
	std::vector<std::vector<size_t>> minimal_samples;

	bool sigmaConsensus(
		const cv::Mat& points_,
		const gcransac::Model& model_,
		gcransac::Model& refined_model_,
		ModelScore& score_,
		const ModelEstimator& estimator_,
		const ModelScore& best_score_);

	bool sigmaConsensusPlusPlus(
		const cv::Mat &points_,
		const gcransac::Model& model_,
		gcransac::Model& refined_model_,
		ModelScore &score_,
		const ModelEstimator &estimator_,
		const ModelScore &best_score_);
};

template <class DatumType, class ModelEstimator>
bool MAGSAC<DatumType, ModelEstimator>::run(
	const cv::Mat& points_,
	const double confidence_,
	ModelEstimator& estimator_,
	gcransac::sampler::Sampler<cv::Mat, size_t> &sampler_,
	gcransac::Model& obtained_model_,
	int& iteration_number_,
	ModelScore &model_score_)
{
	// Initialize variables
	std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measuring: start and end times
	std::chrono::duration<double> elapsed_seconds; // Variables for time measuring: elapsed time
	log_confidence = log(1.0 - confidence_); // The logarithm of 1 - confidence
	point_number = points_.rows; // Number of points
	constexpr size_t sample_size = estimator_.sampleSize(); // The sample size required for the estimation
	size_t max_iteration = iteration_limit; // The maximum number of iterations initialized to the iteration limit
	int iteration = 0; // Current number of iterations
	gcransac::Model so_far_the_best_model; // Current best model
	ModelScore so_far_the_best_score; // The score of the current best model
	std::unique_ptr<size_t[]> minimal_sample(new size_t[sample_size]); // The sample used for the estimation

	std::vector<size_t> pool(points_.rows);
	for (size_t point_idx = 0; point_idx < point_number; ++point_idx)
		pool[point_idx] = point_idx;
	
	if (points_.rows < sample_size)
	{	
		LOG(WARNING) << "There are not enough points for applying robust estimation. Minimum is "
			<< static_cast<int>(sample_size) 
			<< "; while " 
			<< static_cast<int>(points_.rows) 
			<< " are given.";
		return false;
	}

	// Set the start time variable if there is some time limit set
	if (desired_fps > -1)
		start = std::chrono::system_clock::now();

	constexpr size_t max_unsuccessful_model_generations = 50;

	// Save the minimal samples if needed that can be used for training
	if (save_samples)
		minimal_samples.reserve(max_iteration);

	// Main MAGSAC iteration
	while (mininum_iteration_number > iteration ||
		iteration < max_iteration)
	{
		// Increase the current iteration number
		++iteration;
				
		// Sample a minimal subset
		std::vector<gcransac::Model> models; // The set of estimated models
		size_t unsuccessful_model_generations = 0; // The number of unsuccessful model generations
		// Try to select a minimal sample and estimate the implied model parameters
		while (++unsuccessful_model_generations < max_unsuccessful_model_generations)
		{
			// Get a minimal sample randomly
			if (!sampler_.sample(pool, // The index pool from which the minimal sample can be selected
				minimal_sample.get(), // The minimal sample
				sample_size)) // The size of a minimal sample
			{
				//printf("Invalid sample 1\n");
				sampler_.update(
					minimal_sample.get(),
					sample_size,
					iteration,
					0.0);
				continue;
			}

			// Check if the selected sample is valid before estimating the model
			// parameters which usually takes more time. 
			if (!estimator_.isValidSample(points_, // All points
				minimal_sample.get())) // The current sample
			{
				//printf("Invalid sample 1\n");
				sampler_.update(
					minimal_sample.get(),
					sample_size,
					iteration,
					0.0);
				continue;
			}

			// Estimate the model from the minimal sample
 			if (estimator_.estimateModel(points_, // All data points
				minimal_sample.get(), // The selected minimal sample
				&models)) // The estimated models
				break; 
				
			sampler_.update(
				minimal_sample.get(),
				sample_size,
				iteration,
				0.0);
		}         

		// Saving the minimal sample if needed
		if (save_samples)
		{
			minimal_samples.emplace_back(std::vector<size_t>(sample_size));
			for (size_t sample_idx = 0; sample_idx < sample_size; ++sample_idx)
				minimal_samples.back()[sample_idx] = minimal_sample[sample_idx];
		}

		// If the method was not able to generate any usable models, break the cycle.
		iteration += unsuccessful_model_generations - 1;

		// Select the so-far-the-best from the estimated models
		for (const auto &model : models)
		{
			ModelScore score; // The score of the current model
			gcransac::Model refined_model; // The refined model parameters

			// Apply sigma-consensus to refine the model parameters by marginalizing over the noise level sigma
			bool success;
			if (magsac_version == Version::MAGSAC_ORIGINAL)
				success = sigmaConsensus(points_,
					model,
					refined_model,
					score,
					estimator_,
					so_far_the_best_score);
			else
				success = sigmaConsensusPlusPlus(points_,
					model,
					refined_model,
					score,
					estimator_,
					so_far_the_best_score);

			// Continue if the model was rejected
			if (!success || score.score == -1)
				continue;

			// Save the iteration number when the current model is found
			score.iteration = iteration;
						
			// Update the best model parameters if needed
			if (so_far_the_best_score < score)
			{
				so_far_the_best_model = refined_model; // Update the best model parameters
				so_far_the_best_score = score; // Update the best model's score
				max_iteration = MIN(max_iteration, last_iteration_number); // Update the max iteration number, but do not allow to increase
			}
		}		
				
		sampler_.update(
			minimal_sample.get(),
			sample_size,
			iteration,
			0.0);

		// Update the time parameters if a time limit is set
		if (desired_fps > -1)
		{
			end = std::chrono::system_clock::now();
			elapsed_seconds = end - start;

			// Interrupt if the time limit is exceeded
			if (elapsed_seconds.count() > time_limit)
				break;
		}
	}
	
	// Apply sigma-consensus as a post processing step if needed and the estimated model is valid
	if (apply_post_processing)
	{
		// TODO
	}
	
	obtained_model_ = so_far_the_best_model;
	iteration_number_ = iteration;
	model_score_ = so_far_the_best_score;

	return so_far_the_best_score.score > 0;
}

template <class DatumType, class ModelEstimator>
bool MAGSAC<DatumType, ModelEstimator>::postProcessing(
	const cv::Mat &points_,
	const gcransac::Model &model_,
	gcransac::Model &refined_model_,
	ModelScore &refined_score_,
	const ModelEstimator &estimator_)
{
	LOG(WARNING) << "Sigma-consensus++ is not implemented yet as post-processing.";
	return false;
}


template <class DatumType, class ModelEstimator>
bool MAGSAC<DatumType, ModelEstimator>::sigmaConsensus(
	const cv::Mat &points_,
	const gcransac::Model& model_,
	gcransac::Model& refined_model_,
	ModelScore &score_,
	const ModelEstimator &estimator_,
	const ModelScore &best_score_)
{
	// Set up the parameters
	constexpr double L = 1.05;
	constexpr double k = ModelEstimator::getSigmaQuantile();
	constexpr double threshold_to_sigma_multiplier = 1.0 / k;
	constexpr size_t sample_size = estimator_.sampleSize();
	static auto comparator = [](std::pair<double, int> left, std::pair<double, int> right) { return left.first < right.first; };
	const int point_number = points_.rows;
	double current_maximum_sigma = this->maximum_threshold;

	// Calculating the residuals
	std::vector< std::pair<double, size_t> > all_residuals;
	all_residuals.reserve(point_number);

	// If it is not the first run, consider the previous best and interrupt the validation when there is no chance of being better
	if (best_score_.inlier_number > 0)
	{
		// Number of inliers which should be exceeded
		int points_remaining = best_score_.inlier_number;

		// Collect the points which are closer than the threshold which the maximum sigma implies
		for (int point_idx = 0; point_idx < point_number; ++point_idx)
		{
			// Calculate the residual of the current point
			const double residual = estimator_.residual(points_.row(point_idx), model_);
			if (current_maximum_sigma > residual)
			{
				// Store the residual of the current point and its index
				all_residuals.emplace_back(std::make_pair(residual, point_idx));

				// Count points which are closer than a reference threshold to speed up the procedure
				if (residual < interrupting_threshold)
					--points_remaining;
			}

			// Interrupt if there is no chance of being better
			// TODO: replace this part by SPRT test
			if (point_number - point_idx < points_remaining)
				return false;
		}

		// Store the number of really close inliers just to speed up the procedure
		// by interrupting the next verifications.
		score_.inlier_number = best_score_.inlier_number - points_remaining;
	}
	else
	{
		// The number of really close points
		size_t points_close = 0;

		// Collect the points which are closer than the threshold which the maximum sigma implies
		for (size_t point_idx = 0; point_idx < point_number; ++point_idx)
		{
			// Calculate the residual of the current point
			const double residual = estimator_.residual(points_.row(point_idx), model_);
			if (current_maximum_sigma > residual)
			{
				// Store the residual of the current point and its index
				all_residuals.emplace_back(std::make_pair(residual, point_idx));

				// Count points which are closer than a reference threshold to speed up the procedure
				if (residual < interrupting_threshold)
					++points_close;
			}
		}

		// Store the number of really close inliers just to speed up the procedure
		// by interrupting the next verifications.
		score_.inlier_number = points_close;
	}

	std::vector<gcransac::Model> sigma_models;
	std::vector<size_t> sigma_inliers;
	std::vector<double> final_weights;
	
	// The number of possible inliers
	const size_t possible_inlier_number = all_residuals.size();

	// Sort the residuals in ascending order
	std::sort(all_residuals.begin(), all_residuals.end(), comparator);

	// The maximum threshold is set to be slightly bigger than the distance of the
	// farthest possible inlier.
	current_maximum_sigma =
		all_residuals.back().first + std::numeric_limits<double>::epsilon();

	const double sigma_step = current_maximum_sigma / partition_number;

	last_iteration_number = 10000;

	score_.score = 0;

	// The weights calculated by each parallel process
	std::vector<std::vector<double>> point_weights_par(partition_number, std::vector<double>(possible_inlier_number, 0));

	// If OpenMP is used, calculate things in parallel
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(core_number)
	for (int partition_idx = 0; partition_idx < partition_number; ++partition_idx)
	{
		// The maximum sigma value in the current partition
		const double max_sigma = (partition_idx + 1) * sigma_step;

		// Find the last element which has smaller distance than 'max_threshold'
		// Since the vector is ordered binary search can be used to find that particular element.
		const auto &last_element = std::upper_bound(all_residuals.begin(), all_residuals.end(), std::make_pair(max_sigma, 0), comparator);
		const size_t sigma_inlier_number = last_element - all_residuals.begin();

		// Put the indices into a vector
		std::vector<size_t> sigma_inliers;
		sigma_inliers.reserve(sigma_inlier_number);

		// Store the points which are closer than the current sigma limit
		for (size_t relative_point_idx = 0; relative_point_idx < sigma_inlier_number; ++relative_point_idx)
			sigma_inliers.emplace_back(all_residuals[relative_point_idx].second);

		// Check if there are enough inliers to fit a model
		if (sigma_inliers.size() > sample_size)
		{
			// Estimating the model which the current set of inliers imply
			std::vector<gcransac::Model> sigma_models;
			estimator_.estimateModelNonminimal(points_,
				&(sigma_inliers)[0],
				sigma_inlier_number,
				&sigma_models);

			// If the estimation was successful calculate the implied probabilities
			if (sigma_models.size() == 1)
			{
				const double max_sigma_squared_2 = 2 * max_sigma * max_sigma;
				double residual_i_2, // The residual of the i-th point
					probability_i; // The probability of the i-th point

				// Iterate through all points to estimate the related probabilities
				for (size_t relative_point_idx = 0; relative_point_idx < sigma_inliers.size(); ++relative_point_idx)
				{
					// TODO: Replace with Chi-square instead of normal distribution
					const size_t &point_idx = sigma_inliers[relative_point_idx];

					// Calculate the residual of the current point
					residual_i_2 = estimator_.squaredResidual(points_.row(point_idx),
						sigma_models[0]);

					// Calculate the probability of the i-th point assuming Gaussian distribution
					// TODO: replace by Chi-square distribution
					probability_i = exp(-residual_i_2 / max_sigma_squared_2);

					// Store the probability of the i-th point coming from the current partition
					point_weights_par[partition_idx][relative_point_idx] += probability_i;


				}
			}
		}
	}
#else
	LOG(ERROR) << "Not implemented yet.";
#endif

	// The weights used for the final weighted least-squares fitting
	// If point normalization is applied the indexing of the weights differs.
	// In that case
	//		final_weights[i] is the weight of inlier[i]-th point
	// Otherwise,
	//		final_weights[i] is the weight of i-th point
	if constexpr (ModelEstimator::doesNormalizationForNonMinimalFitting())
		final_weights.reserve(possible_inlier_number);
	else
		final_weights.resize(point_number, 0);

	// Collect all points which has higher probability of being inlier than zero
	sigma_inliers.reserve(possible_inlier_number);
	for (size_t point_idx = 0; point_idx < possible_inlier_number; ++point_idx)
	{
		// Calculate the weight of the current point
		double weight = 0.0;
		for (size_t partition_idx = 0; partition_idx < partition_number; ++partition_idx)
			weight += point_weights_par[partition_idx][point_idx];

		// If the weight is approx. zero, continue.
		if (weight < std::numeric_limits<double>::epsilon())
			continue;

		// Store the index and weight of the current point
		sigma_inliers.emplace_back(all_residuals[point_idx].second);

		if constexpr (ModelEstimator::doesNormalizationForNonMinimalFitting())
			final_weights.emplace_back(weight);
		else
			final_weights[point_idx] = weight;
	}

	// If there are fewer inliers than the size of the minimal sample interupt the procedure
	if (sigma_inliers.size() < sample_size)
		return false;

	// Estimate the model parameters using weighted least-squares fitting
	if (!estimator_.estimateModelNonminimal(
		points_, // All input points
		&(sigma_inliers)[0], // Points which have higher than 0 probability of being inlier
		static_cast<int>(sigma_inliers.size()), // Number of possible inliers
		&sigma_models, // Estimated models
		&(final_weights)[0])) // Weights of points 
		return false;

	bool is_model_updated = false;
	
	if (sigma_models.size() == 1 && // If only a single model is estimated
		estimator_.isValidModel(sigma_models.back(),
			points_,
			sigma_inliers,
			&(sigma_inliers)[0],
			interrupting_threshold,
			is_model_updated)) // and it is valid
	{
		// Return the refined model
		refined_model_ = sigma_models.back();

		// Calculate the score of the model and the implied iteration number
		double marginalized_iteration_number;
		getModelQuality(points_, // All the input points
			refined_model_, // The estimated model
			estimator_, // The estimator
			marginalized_iteration_number, // The marginalized inlier ratio
			score_.score); // The marginalized score

		if (marginalized_iteration_number < 0 || std::isnan(marginalized_iteration_number))
			last_iteration_number = std::numeric_limits<int>::max();
		else
			last_iteration_number = static_cast<int>(round(marginalized_iteration_number));
		return true;
	}
	return false;
}

template <class DatumType, class ModelEstimator>
bool MAGSAC<DatumType, ModelEstimator>::sigmaConsensusPlusPlus(
	const cv::Mat &points_,
	const gcransac::Model& model_,
	gcransac::Model& refined_model_,
	ModelScore &score_,
	const ModelEstimator &estimator_,
	const ModelScore &best_score_)
{
	// The degrees of freedom of the data from which the model is estimated.
	// E.g., for models coming from point correspondences (x1,y1,x2,y2), it is 4.
	constexpr size_t degrees_of_freedom = ModelEstimator::getDegreesOfFreedom();
	// A 0.99 quantile of the Chi^2-distribution to convert sigma values to residuals
	constexpr double k = ModelEstimator::getSigmaQuantile();
	// A multiplier to convert residual values to sigmas
	constexpr double threshold_to_sigma_multiplier = 1.0 / k;
	// Calculating k^2 / 2 which will be used for the estimation and, 
	// due to being constant, it is better to calculate it a priori.
	constexpr double squared_k_per_2 = k * k / 2.0;
	// Calculating (DoF - 1) / 2 which will be used for the estimation and, 
	// due to being constant, it is better to calculate it a priori.
	constexpr double dof_minus_one_per_two = (degrees_of_freedom - 1.0) / 2.0;
	// TODO: check
	constexpr double C = ModelEstimator::getC();
	// The size of a minimal sample used for the estimation
	constexpr size_t sample_size = estimator_.sampleSize();
	// Calculating 2^(DoF - 1) which will be used for the estimation and, 
	// due to being constant, it is better to calculate it a priori.
	static const double two_ad_dof = std::pow(2.0, dof_minus_one_per_two);
	// Calculating C * 2^(DoF - 1) which will be used for the estimation and, 
	// due to being constant, it is better to calculate it a priori.
	static const double C_times_two_ad_dof = C * two_ad_dof;
	// Calculating the gamma value of (DoF - 1) / 2 which will be used for the estimation and, 
	// due to being constant, it is better to calculate it a priori.
	static const double gamma_value = tgamma(dof_minus_one_per_two);
	// Calculating the upper incomplete gamma value of (DoF - 1) / 2 with k^2 / 2.
	constexpr double gamma_k = ModelEstimator::getUpperIncompleteGammaOfK();
	// Calculating the lower incomplete gamma value of (DoF - 1) / 2 which will be used for the estimation and, 
	// due to being constant, it is better to calculate it a priori.
	static const double gamma_difference = gamma_value - gamma_k;
	// The number of points provided
	const int point_number = points_.rows;
	// The manually set maximum inlier-outlier threshold
	double current_maximum_sigma = this->maximum_threshold;
	// Calculating the pairs of (residual, point index).
	std::vector< std::pair<double, size_t> > residuals;
	// Occupy the maximum required memory to avoid doing it later.
	residuals.reserve(point_number);

	// If it is not the first run, consider the previous best and interrupt the validation when there is no chance of being better
	if (best_score_.inlier_number > 0)
	{
		// Number of points close to the previous so-far-the-best model. 
		// This model should have more inliers.
		int points_remaining = best_score_.inlier_number;

		// Collect the points which are closer than the threshold which the maximum sigma implies
		for (int point_idx = 0; point_idx < point_number; ++point_idx)
		{
			// Calculate the residual of the current point
			const double residual = estimator_.residual(points_.row(point_idx), model_);
			if (current_maximum_sigma > residual)
			{
				// Store the residual of the current point and its index
				residuals.emplace_back(std::make_pair(residual, point_idx));

				// Count points which are closer than a reference threshold to speed up the procedure
				if (residual < interrupting_threshold)
					--points_remaining;
			}

			// Interrupt if there is no chance of being better
			// TODO: replace this part by SPRT test
			if (point_number - point_idx < points_remaining)
				return false;
		}

		// Store the number of really close inliers just to speed up the procedure
		// by interrupting the next verifications.
		score_.inlier_number = best_score_.inlier_number - points_remaining;
	}
	else
	{
		// The number of really close points
		size_t points_close = 0;

		// Collect the points which are closer than the threshold which the maximum sigma implies
		for (size_t point_idx = 0; point_idx < point_number; ++point_idx)
		{
			// Calculate the residual of the current point
			const double residual = estimator_.residual(points_.row(point_idx), model_);
			if (current_maximum_sigma > residual)
			{
				// Store the residual of the current point and its index
				residuals.emplace_back(std::make_pair(residual, point_idx));

				// Count points which are closer than a reference threshold to speed up the procedure
				if (residual < interrupting_threshold)
					++points_close;
			}
		}

		// Store the number of really close inliers just to speed up the procedure
		// by interrupting the next verifications.
		score_.inlier_number = points_close;
	}

	// Models fit by weighted least-squares fitting
	std::vector<gcransac::Model> sigma_models;
	// Points used in the weighted least-squares fitting
	std::vector<size_t> sigma_inliers;
	// Weights used in the the weighted least-squares fitting
	std::vector<double> sigma_weights;
	// Number of points considered in the fitting
	const size_t possible_inlier_number = residuals.size();
	// Occupy the memory to avoid doing it inside the calculation possibly multiple times
	sigma_inliers.reserve(possible_inlier_number);
	// Occupy the memory to avoid doing it inside the calculation possibly multiple times
	sigma_weights.reserve(possible_inlier_number);

	// Calculate 2 * \sigma_{max}^2 a priori
	const double squared_sigma_max_2 = current_maximum_sigma * current_maximum_sigma * 2.0;
	// Divide C * 2^(DoF - 1) by \sigma_{max} a priori
	const double one_over_sigma = C_times_two_ad_dof / current_maximum_sigma;
	// Calculate the weight of a point with 0 residual (i.e., fitting perfectly) a priori
	const double weight_zero = one_over_sigma * gamma_difference;

	// Initialize the polished model with the initial one
	gcransac::Model polished_model = model_;
	// A flag to determine if the initial model has been updated
	bool updated = false;

	// Do the iteratively re-weighted least squares fitting
	for (size_t iterations = 0; iterations < number_of_irwls_iters; ++iterations)
	{
		// If the current iteration is not the first, the set of possibly inliers 
		// (i.e., points closer than the maximum threshold) have to be recalculated. 
		if (iterations > 0)
		{
			// The number of points close to the model
			size_t points_close = 0;
			// Remove everything from the residual vector
			residuals.clear();

			// Collect the points which are closer than the maximum threshold
			for (size_t point_idx = 0; point_idx < point_number; ++point_idx)
			{
				// Calculate the residual of the current point
				const double residual = estimator_.residual(points_.row(point_idx), polished_model);
				if (current_maximum_sigma > residual)
				{
					// Store the residual of the current point and its index
					residuals.emplace_back(std::make_pair(residual, point_idx));

					// Count points which are closer than a reference threshold to speed up the procedure
					if (residual < interrupting_threshold)
						++points_close;
				}
			}

			// Store the number of really close inliers just to speed up the procedure
			// by interrupting the next verifications.
			score_.inlier_number = points_close;

			// Number of points closer than the threshold
			const size_t possible_inlier_number = residuals.size();

			// Clear the inliers and weights
			sigma_inliers.clear();
			sigma_weights.clear();

			// Occupy the memory for the inliers and weights
			sigma_inliers.reserve(possible_inlier_number);
			sigma_weights.reserve(possible_inlier_number);
		}

		if constexpr (!ModelEstimator::doesNormalizationForNonMinimalFitting())
			sigma_weights.resize(point_number, 0);

		// Calculate the weight of each point
		for (const auto &[residual, idx] : residuals)
		{
			// The weight
			double weight = 0.0;
			// If the residual is ~0, the point fits perfectly and it is handled differently
			if (residual < std::numeric_limits<double>::epsilon())
				weight = weight_zero;
			else
			{
				// Calculate the squared residual
				const double squared_residual = residual * residual;
				// Get the position of the gamma value in the lookup table
				size_t x = round(precision_of_stored_gammas * squared_residual / squared_sigma_max_2);
				// Put the index of the point into the vector of points used for the least squares fitting
				sigma_inliers.emplace_back(idx);

				// If the sought gamma value is not stored in the lookup, return the closest element
				if (stored_gamma_number < x)
					x = stored_gamma_number;

				// Calculate the weight of the point
				weight = one_over_sigma * (stored_gamma_values[x] - gamma_k);
			}

			// Store the weight of the point 
			if constexpr (ModelEstimator::doesNormalizationForNonMinimalFitting())
				sigma_weights.emplace_back(weight);
			else
				sigma_weights[idx] = weight;
		}

		// If there are fewer than the minimum point close to the model,
		// terminate.
		if (sigma_inliers.size() < sample_size)
			return false;

		// Estimate the model parameters using weighted least-squares fitting
		if (!estimator_.estimateModelNonminimal(
			points_, // All input points
			&(sigma_inliers)[0], // Points which have higher than 0 probability of being inlier
			static_cast<int>(sigma_inliers.size()), // Number of possible inliers
			&sigma_models, // Estimated models
			&(sigma_weights)[0])) // Weights of points 
		{
			// If the estimation failed and the iteration was never successfull,
			// terminate with failure.
			if (iterations == 0)
				return false;
			// Otherwise, if the iteration was successfull at least once,
			// simply break it. 
			break;
		}

		// Update the model parameters
		polished_model = sigma_models[0];
		// Clear the vector of models and keep only the best
		sigma_models.clear();
		// The model has been updated
		updated = true;
	}

	bool is_model_updated = false;

	if (updated && // If the model has been updated
		estimator_.isValidModel(polished_model,
			points_,
			sigma_inliers,
			&(sigma_inliers[0]),
			interrupting_threshold,
			is_model_updated)) // and it is valid
	{
		// Return the refined model
		refined_model_ = polished_model;

		// Calculate the score of the model and the implied iteration number
		double marginalized_iteration_number;
		getModelQualityPlusPlus(points_, // All the input points
			refined_model_, // The estimated model
			estimator_, // The estimator
			score_.score, // The marginalized score
			best_score_.score); // The score of the previous so-far-the-best model
			
		// Update the iteration number
		last_iteration_number =
			log_confidence / log(1.0 - std::pow(static_cast<double>(score_.inlier_number) / point_number, sample_size));
		return true;
	}
	return false;
}

template <class DatumType, class ModelEstimator>
void MAGSAC<DatumType, ModelEstimator>::getModelQualityPlusPlus(
	const cv::Mat &points_, // All data points
	const gcransac::Model &model_, // The model parameter
	const ModelEstimator &estimator_, // The model estimator class
	double &score_, // The score to be calculated
	const double &previous_best_score_) // The score of the previous so-far-the-best model 
{
	// The degrees of freedom of the data from which the model is estimated.
	// E.g., for models coming from point correspondences (x1,y1,x2,y2), it is 4.
	constexpr size_t degrees_of_freedom = ModelEstimator::getDegreesOfFreedom();
	// A 0.99 quantile of the Chi^2-distribution to convert sigma values to residuals
	constexpr double k = ModelEstimator::getSigmaQuantile();
	// A multiplier to convert residual values to sigmas
	constexpr double threshold_to_sigma_multiplier = 1.0 / k;
	// Calculating k^2 / 2 which will be used for the estimation and, 
	// due to being constant, it is better to calculate it a priori.
	constexpr double squared_k_per_2 = k * k / 2.0;
	// Calculating (DoF - 1) / 2 which will be used for the estimation and, 
	// due to being constant, it is better to calculate it a priori.
	constexpr double dof_minus_one_per_two = (degrees_of_freedom - 1.0) / 2.0;
	// Calculating (DoF + 1) / 2 which will be used for the estimation and, 
	// due to being constant, it is better to calculate it a priori.
	constexpr double dof_plus_one_per_two = (degrees_of_freedom + 1.0) / 2.0;
	// TODO: check
	constexpr double C = 0.25;
	// Calculating 2^(DoF - 1) which will be used for the estimation and, 
	// due to being constant, it is better to calculate it a priori.
	static const double two_ad_dof_minus_one = std::pow(2.0, dof_minus_one_per_two);
	// Calculating 2^(DoF + 1) which will be used for the estimation and, 
	// due to being constant, it is better to calculate it a priori.
	static const double two_ad_dof_plus_one = std::pow(2.0, dof_plus_one_per_two);
	// Calculate the gamma value of k
	constexpr double gamma_value_of_k = ModelEstimator::getUpperIncompleteGammaOfK();
	// Calculate the lower incomplete gamma value of k
	constexpr double lower_gamma_value_of_k = ModelEstimator::getLowerIncompleteGammaOfK();
	// The number of points provided
	const int point_number = points_.rows;
	// The previous best loss
	const double previous_best_loss = 1.0 / previous_best_score_;
	// Convert the maximum threshold to a sigma value
	const double maximum_sigma = threshold_to_sigma_multiplier * maximum_threshold;
	// Calculate the squared maximum sigma
	const double maximum_sigma_2 = maximum_sigma * maximum_sigma;
	// Calculate \sigma_{max}^2 / 2
	const double maximum_sigma_2_per_2 = maximum_sigma_2 / 2.0;
	// Calculate 2 * \sigma_{max}^2
	const double maximum_sigma_2_times_2 = maximum_sigma_2 * 2.0;
	// Calculate the loss implied by an outlier
	const double outlier_loss = maximum_sigma * two_ad_dof_minus_one  * lower_gamma_value_of_k;
	// Calculating 2^(DoF + 1) / \sigma_{max} which will be used for the estimation and, 
	// due to being constant, it is better to calculate it a priori.
	const double two_ad_dof_plus_one_per_maximum_sigma = two_ad_dof_plus_one / maximum_sigma;
	// The loss which a point implies
	double loss = 0.0,
		// The total loss regarding the current model
		total_loss = 0.0;

	// Iterate through all points to calculate the implied loss
	for (size_t point_idx = 0; point_idx < point_number; ++point_idx)
	{
		// Calculate the residual of the current point
		const double residual =
			estimator_.residualForScoring(points_.row(point_idx), model_.descriptor);

		// If the residual is smaller than the maximum threshold, consider it outlier
		// and add the loss implied to the total loss.
		if (maximum_threshold < residual)
			loss = outlier_loss;
		else // Otherwise, consider the point inlier, and calculate the implied loss
		{
			// Calculate the squared residual
			const double squared_residual = residual * residual;
			// Divide the residual by the 2 * \sigma^2
			const double squared_residual_per_sigma = squared_residual / maximum_sigma_2_times_2;
			// Get the position of the gamma value in the lookup table
			size_t x = round(precision_of_stored_incomplete_gammas * squared_residual_per_sigma);
			// If the sought gamma value is not stored in the lookup, return the closest element
			if (stored_incomplete_gamma_number < x)
				x = stored_incomplete_gamma_number;

			// Calculate the loss implied by the current point
			loss = maximum_sigma_2_per_2 * stored_lower_incomplete_gamma_values[x] +
				squared_residual / 4.0 * (stored_complete_gamma_values[x] -
					gamma_value_of_k);
			loss = loss * two_ad_dof_plus_one_per_maximum_sigma;
		}

		// Update the total loss
		total_loss += loss;

		// Break the validation if there is no chance of being better than the previous
		// so-far-the-best model.
		if (previous_best_loss < total_loss)
			break;
	}

	// Calculate the score of the model from the total loss
	score_ = 1.0 / total_loss;
}

template <class DatumType, class ModelEstimator>
void MAGSAC<DatumType, ModelEstimator>::getModelQuality(
	const cv::Mat &points_, // All data points
	const gcransac::Model &model_, // The model parameter
	const ModelEstimator &estimator_, // The model estimator class
	double &marginalized_iteration_number_, // The marginalized iteration number to be calculated
	double &score_) // The score to be calculated
{
	// Set up the parameters
	constexpr size_t sample_size = estimator_.sampleSize();
	const size_t point_number = points_.rows;

	// Getting the inliers
	std::vector<std::pair<double, size_t>> all_residuals;
	all_residuals.reserve(point_number);

	double max_distance = 0;
	for (size_t point_idx = 0; point_idx < point_number; ++point_idx)
	{
		// Calculate the residual of the current point
		const double residual =
			estimator_.residualForScoring(points_.row(point_idx), model_.descriptor);
		// If the residual is smaller than the maximum threshold, add it to the set of possible inliers
		if (maximum_threshold > residual)
		{
			max_distance = MAX(max_distance, residual);
			all_residuals.emplace_back(std::make_pair(residual, point_idx));
		}
	}

	// Set the maximum distance to be slightly bigger than that of the farthest possible inlier
	max_distance = max_distance +
		std::numeric_limits<double>::epsilon();

	// Number of possible inliers
	const size_t possible_inlier_number = all_residuals.size();

	// The extent of a partition
	const double threshold_step = max_distance / partition_number;

	// The maximum threshold considered in each partition
	std::vector<double> thresholds(partition_number);
	std::vector<double> thresholds_squared(partition_number);
	std::vector<double> thresholds_2_squared(partition_number);

	// Calculating the thresholds for each partition
	for (size_t i = 0; i < partition_number; ++i)
	{
		thresholds[i] = (i + 1) * threshold_step;
		thresholds_squared[i] = thresholds[i] * thresholds[i];
		thresholds_2_squared[i] = 2 * thresholds_squared[i];
	}

	double residual_i, // Residual of the i-th point
		residual_i_squared, // Squared residual of the i-th poin 
		probability_i; // Probability of the i-th point given the model

	std::vector<double> inliers(partition_number, 0), // RANSAC score for each partition
		probabilities(partition_number, 1); // Probabilities for each partition
	for (size_t point_idx = 0; point_idx < possible_inlier_number; ++point_idx)
	{
		residual_i = all_residuals[point_idx].first;
		residual_i_squared = residual_i * residual_i;

		for (size_t i = 0; i < partition_number; ++i)
		{
			if (residual_i < thresholds[i])
			{
				probability_i = 1.0 - residual_i_squared / thresholds_squared[i];
				++inliers[i];
				probabilities[i] += probability_i;
			}
		}
	}

	score_ = 0;
	marginalized_iteration_number_ = 0.0;
	for (auto i = 0; i < partition_number; ++i)
	{
		score_ += probabilities[i];
		marginalized_iteration_number_ += log_confidence / log(1.0 - std::pow(inliers[i] / point_number, sample_size));
	}
	marginalized_iteration_number_ = marginalized_iteration_number_ / partition_number;
}
