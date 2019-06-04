#pragma once

#include <vector>
#include <cv.h>

// Templated class for estimating a model for RANSAC. This class is purely a
// virtual class and should be implemented for the specific task that RANSAC is
// being used for. Two methods must be implemented: estimateModel and Error. All
// other methods are optional, but will likely enhance the quality of the RANSAC
// output.
template <typename DatumType, typename ModelType> class Estimator
{
public:
	typedef DatumType Datum;
	typedef ModelType Model;

	cv::Mat last_estimation_data;
	cv::Mat next_estimation_data;

	Estimator() {}
	virtual ~Estimator() {}

	virtual std::string modelName() const = 0;

	// Get the minimum number of samples needed to generate a model.
	virtual int sampleSize() const = 0;

	void finalize(Model &model) const {}

	virtual bool isErrorSquared() const = 0;

	// Given a set of data points, Estimate the model. Users should implement this
	// function appropriately for the task being solved. Returns true for
	// successful model esticv::Mation (and outputs model), false for failed
	// esticv::Mation. Typically, this is a minimal set, but it is not required to be.
	virtual bool estimateModel(
		const cv::Mat& data_,
		const int *sample_, 
		std::vector<Model>* model_) const = 0;

	// Estimate a model from a non-minimal sampling of the data. E.g. for a line,
	// use SVD on a set of points instead of constructing a line from two points.
	// By default, this simply implements the minimal case.
	virtual bool estimateModelNonminimal(
		const cv::Mat& data,
		const int *sample_,
		size_t sample_number_,
		std::vector<Model>* model_) const = 0;
		
	virtual bool estimateModelNonminimalWeighted(
		const cv::Mat& data_,
		const int *sample_,
		const double *weights_,
		size_t sample_size_,
		std::vector<Model>* models_) const = 0;

	// Given a model and a data point, calculate the error. Users should implement
	// this function appropriately for the task being solved.
	virtual double error(const Datum& data_, const Model& model_) const = 0;
	virtual double error(const Datum& data_, const cv::Mat& model_) const = 0;

	// Compute the residuals of many data points. By default this is just a loop
	// that calls Error() on each data point, but this function can be useful if
	// the errors of multiple points may be Estimated simultanesously (e.g.,
	// cv::Matrix multiplication to compute the reprojection error of many points at
	// once).
	virtual std::vector<double> residuals(
		const std::vector<Datum>& data_,
		const Model& model_) const 
	{
		std::vector<double> residuals(data_.size());
		for (auto i = 0; i < data_.size(); i++) 
			residuals[i] = error(data_[i], model_);
		return residuals;
	}

	// Returns the set inliers of the data set based on the error threshold
	// provided.
	std::vector<int> getInliers(const std::vector<Datum>& data_,
		const Model& model_,
		double error_threshold_) const 
	{
		std::vector<int> inliers;
		inliers.reserve(data_.size());

		for (auto i = 0; i < data_.size(); i++)
		{
			if (error(data_[i], model_) < error_threshold_) 
				inliers.emplace_back(i);
		}

		return inliers;
	}

	// Enable a quick check to see if the model is valid. This can be a geometric
	// check or some other verification of the model structure.
	virtual bool validModel(const Model& model_) const { return true; }
};

