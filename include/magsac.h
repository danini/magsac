#pragma once

#include <limits>
#include <cv.h>
#include <chrono>
#include <memory>
#include "model.h"
#include "model_score.h"
#include "sampler.h"
#include "uniform_sampler.h"

#ifdef _WIN32 
	#include <ppl.h>
#endif

template <class DatumType, class ModelEstimator>
class MAGSAC  
{
public:
	MAGSAC() : 
		time_limit(std::numeric_limits<double>::max()),
		desired_fps(-1),
		iteration_limit(std::numeric_limits<size_t>::max()),
		maximum_threshold(10.0),
		apply_post_processing(true),
		mininum_iteration_number(50),
		partition_number(5),
		core_number(1),
		interrupting_threshold(1.0),
		last_iteration_number(0),
		log_confidence(0),
		point_number(0)
	{ 
	}

	~MAGSAC() {}

	bool run(
		const cv::Mat &points_, 
		const double confidence_,
		ModelEstimator& estimator_,
		gcransac::sampler::Sampler<cv::Mat, size_t> &sampler_,
		gcransac::Model &obtained_model_,  
		int &iteration_number_);

	bool scoreLess(
		const ModelScore &score_1_, 
		const ModelScore &score_2_)
	{ 
		return score_1_.score < score_2_.score; 
	}
	
	void setMaximumThreshold(const double maximum_threshold_) 
	{
		maximum_threshold = maximum_threshold_;
	}

	void setReferenceThreshold(const double threshold_)
	{
		reference_inlier_outlier_threshold = threshold_;
	}

	void applyPostProcessing(bool value_) 
	{
		apply_post_processing = value_;
	}

	void setIterationLimit(size_t iteration_limit_)
	{
		iteration_limit = iteration_limit_;
	}

	void setCoreNumber(size_t core_number_)
	{
		core_number = core_number_;
	}

	void setPartitionNumber(size_t partition_number_)
	{
		partition_number = partition_number_;
	}

	void setMinimumIterationNumber(size_t mininum_iteration_number_)
	{
		mininum_iteration_number = mininum_iteration_number_;
	}

	void setFPS(int fps_) { desired_fps = fps_; time_limit = fps_ <= 0 ? std::numeric_limits<double>::max() : 1.0 / fps_; }

	bool postProcessing(
		const cv::Mat &points,
		const gcransac::Model &so_far_the_best_model,
		gcransac::Model &output_model,
		ModelScore &output_score,
		const ModelEstimator &estimator);

	void getSigmaScore(
		const cv::Mat& points_,
		const gcransac::Model& model_,
		const ModelEstimator& estimator_,
		double& avg_inlier_ratio_,
		double& score_);

protected:
	size_t iteration_limit; // Maximum number of iterations allowed
	size_t mininum_iteration_number; // Minimum number of iteration before terminating
	double reference_inlier_outlier_threshold; // An inlier-outlier threshold to speed up the procedure by interrupting sigma-consensus if needed
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

	bool sigmaConsensus(
		const cv::Mat& points_,
		const gcransac::Model& model_,
		gcransac::Model& refined_model_,
		ModelScore& score_,
		const ModelEstimator& estimator_,
		const ModelScore& best_score_);
};

template <class DatumType, class ModelEstimator>
bool MAGSAC<DatumType, ModelEstimator>::run(
	const cv::Mat& points_,
	const double confidence_,
	ModelEstimator& estimator_,
	gcransac::sampler::Sampler<cv::Mat, size_t> &sampler_,
	gcransac::Model& obtained_model_,
	int& iteration_number_)
{
	// Initialize variables
	std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measuring: start and end times
	std::chrono::duration<double> elapsed_seconds; // Variables for time measuring: elapsed time
	log_confidence = log(1.0 - confidence_); // The logarithm of 1 - confidence
	point_number = points_.rows; // Number of points
	const int sample_size = estimator_.sampleSize(); // The sample size required for the estimation
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
		fprintf(stderr, "There are not enough points for applying robust estimation. Minimum is %d; while %d are given.\n", 
			sample_size, points_.rows);
		return false;
	}

	// Set the start time variable if there is some time limit set
	if (desired_fps > -1)
		start = std::chrono::system_clock::now();

	constexpr size_t max_unsuccessful_model_generations = 50;

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
				continue;
						 
			// Estimate the model from the minimal sample
 			if (estimator_.estimateModel(points_, // All data points
				minimal_sample.get(), // The selected minimal sample
				&models)) // The estimated models
				break; 
		}         

		// If the method was not able to generate any usable models, break the cycle.
		if (unsuccessful_model_generations >= max_unsuccessful_model_generations)
			break;

		// Select the so-far-the-best from the estimated models
		for (const auto &model : models)
		{
			ModelScore score; // The score of the current model
			gcransac::Model refined_model; // The refined model parameters

			// Apply sigma-consensus to refine the model parameters by marginalizing over the noise level sigma
			const bool success = sigmaConsensus(points_,
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
			if (scoreLess(so_far_the_best_score, score))
			{
				so_far_the_best_model = refined_model; // Update the best model parameters
				so_far_the_best_score = score; // Update the best model's score
				max_iteration = MIN(max_iteration, last_iteration_number); // Update the max iteration number, but do not allow to increase
			}
		}

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
	if (apply_post_processing &&
		estimator_.isValidModel(so_far_the_best_model,
			points_,
			std::vector<size_t>(),
			reference_inlier_outlier_threshold))
	{
		gcransac::Model refined_model;
		/*if (postProcessing(points_,
			so_far_the_best_model,
			refined_model,
			so_far_the_best_score,
			estimator_))
			so_far_the_best_model = refined_model;*/
	}
	
	obtained_model_ = so_far_the_best_model;
	iteration_number_ = iteration;

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
	// Set up the parameters
	constexpr double threshold_to_sigma_multiplier = 1.0 / 3.64;
	double threshold = this->maximum_threshold;
	const size_t point_number = points_.rows;
	const size_t sample_size = estimator_.sampleSize();

	// Collect the points which are closer than the maximum threshold
	std::vector<std::pair<double, size_t>> all_residuals;
	all_residuals.reserve(point_number);
	for (auto pt_idx = 0; pt_idx < point_number; ++pt_idx)
	{
		// The residual of the current point given the model
		const double residual = estimator_.residual(points_.row(pt_idx), model_);

		// If the residual is higher than the threshold consider it when doing sigma-consensus
		if (threshold > residual)
			all_residuals.emplace_back(
				std::make_pair(residual * threshold_to_sigma_multiplier, pt_idx));
	}

	// Number of points closer than the maximum distance
	const size_t Ni = all_residuals.size();

	if (Ni < sample_size)
		return false;

	// Sort the (residual, point index) pairs in ascending order
	const auto comparator = [](std::pair<double, int> left, std::pair<double, int> right) { return left.first < right.first; };
	std::sort(all_residuals.begin(), all_residuals.end(), comparator);

	// Set the threshold to be the distance of the farthest point which has lower residual than the maximum sigma
	threshold = all_residuals.back().first + 
		std::numeric_limits<double>::epsilon();

	// The inliers of the current sigma
	std::vector<size_t> sigma_inliers;
	sigma_inliers.reserve(Ni);

	const int step_size = (Ni - sample_size) / core_number;

	std::vector<double> final_weights(Ni, 0);
	std::vector<std::vector<double>> point_weights(core_number, 
		std::vector<double>(Ni, 0));
	const double divisions_per_process = partition_number / core_number;
	const double sigma_step = threshold / partition_number;

	for (auto process = 0; process < core_number; ++process)
	{
		const double last_sigma = (process + 1) * divisions_per_process * sigma_step;
		double prev_sigma = process * divisions_per_process * sigma_step;
		double next_sigma = prev_sigma + sigma_step;		
		double next_sigma_2 = 2 * next_sigma * next_sigma;

		int sigma_idx;
		std::vector<size_t> temp_sigma_inliers;
		
		for (sigma_idx = 0; sigma_idx < Ni; ++sigma_idx)
		{
			const std::pair<double, size_t> &next = all_residuals[sigma_idx];
			const double sigma = next.first;

			// Collecting the points while the next step is not achieved
			if (sigma < next_sigma)
			{
				temp_sigma_inliers.emplace_back(all_residuals[sigma_idx].second);
				continue;
			}

			// Estimating model(sigma)
			if (temp_sigma_inliers.size() > sample_size)
			{
				std::vector<gcransac::Model> model_sigma;
				estimator_.estimateModelNonminimal(points_, 
					&(temp_sigma_inliers)[0], 
					static_cast<int>(temp_sigma_inliers.size()),
					&model_sigma);

				if (model_sigma.size() > 0)
				{
					double ri, pi;
					for (auto pt_idx = 0; pt_idx < temp_sigma_inliers.size(); ++pt_idx)
					{
						const auto real_pt_idx = temp_sigma_inliers[pt_idx];
						ri = estimator_.residual(points_.row(real_pt_idx), model_sigma[0].descriptor);
						pi = exp(-ri * ri / next_sigma_2);
						pi = sigma_step * pi;
						point_weights[process][pt_idx] += pi;
					}
				}
			} 
			
			// Update the next sigma
			prev_sigma = next_sigma;
			next_sigma += sigma_step;

			// Break if all divisions have been processed
			if (next_sigma > last_sigma)
				break;

			temp_sigma_inliers.emplace_back(all_residuals[sigma_idx].second);
			next_sigma_2 = 2 * next_sigma * next_sigma;
		}
	}

	// Collect the points which have higher than 0 probability of being inlier
	for (auto sigma_idx = 0; sigma_idx < Ni; sigma_idx += 1)
		sigma_inliers.emplace_back(all_residuals[sigma_idx].second);

	// Accumulate the sigmas
	for (auto process = 0; process < core_number; ++process)
		for (auto pt_idx = 0; pt_idx < Ni; ++pt_idx)
			final_weights[pt_idx] += point_weights[process][pt_idx];

	// Estimate the model by weighted least-squares using the posterior probabilities as weights
	std::vector<gcransac::Model> sigma_models;
	estimator_.estimateModelNonminimal(
		points_, // All input points
		&(sigma_inliers)[0], // Points which have higher than 0 probability of being inlier
		static_cast<int>(sigma_inliers.size()), // Number of possible inliers
		&sigma_models, // Estimated models
		&(final_weights)[0]); // Weights of points 

	// Update the model if needed
	if (sigma_models.size() == 1)
		refined_model_ = sigma_models[0];
	else
		refined_model_ = model_;
	return true;
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
	constexpr double L = 1.05,
		threshold_to_sigma_multiplier = 1.0 / 3.64;
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
				all_residuals.emplace_back(std::make_pair(residual * threshold_to_sigma_multiplier, point_idx));

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
				all_residuals.emplace_back(std::make_pair(residual * threshold_to_sigma_multiplier, point_idx));

				// Count points which are closer than a reference threshold to speed up the procedure
				if (residual < interrupting_threshold)
					++points_close;
			}
		}

		// Store the number of really close inliers just to speed up the procedure
		// by interrupting the next verifications.
		score_.inlier_number = points_close;
	}

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
	fprintf(stderr, "Not implemented yet.\n");
#endif

	// The weights used for the final weighted least-squares fitting
	std::vector<double> final_weights;
	final_weights.reserve(possible_inlier_number);

	// Collect all points which has higher probability of being inlier than zero
	std::vector<size_t> sigma_inliers; // The points with higher than 0 probability
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
		final_weights.emplace_back(weight);
	}

	// If there are fewer inliers than the size of the minimal sample interupt the procedure
	if (sigma_inliers.size() < sample_size)
		return false;

	// Estimate the model parameters using weighted least-squares fitting
	std::vector<gcransac::Model> sigma_models;
	if (!estimator_.estimateModelNonminimal(
		points_, // All input points
		&(sigma_inliers)[0], // Points which have higher than 0 probability of being inlier
		static_cast<int>(sigma_inliers.size()), // Number of possible inliers
		&sigma_models, // Estimated models
		&(final_weights)[0])) // Weights of points 
		return false;
	
	if (sigma_models.size() == 1 && // If only a single model is estimated
		estimator_.isValidModel(sigma_models.back(),
			points_,
			sigma_inliers,
			interrupting_threshold)) // and it is valid
	{
		// Return the refined model
		refined_model_ = sigma_models.back();

		// Calculate the score of the model and the implied iteration number
		double marginalized_iteration_number;
		getSigmaScore(points_, // All the input points
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
void MAGSAC<DatumType, ModelEstimator>::getSigmaScore(
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
