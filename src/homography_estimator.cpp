#include <iostream>
#include <math.h>
#include <chrono>
#include <random>
#include <vector>
#include <cv.h>
#include "opencv2/calib3d/calib3d.hpp"

#include "estimator.h"

struct Homography
{
	cv::Mat descriptor;

	Homography(cv::Mat descriptor_) : descriptor(descriptor_) {}
	Homography() {}
	Homography(const Homography& other)
	{
		descriptor = other.descriptor.clone();
	}
};

// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
class RobustHomographyEstimator : public Estimator < cv::Mat, Homography >
{
protected:

public:
	RobustHomographyEstimator() {}
	~RobustHomographyEstimator() {}

	// Number of correspondences required to estimate a homography
	int sampleSize() const {
		return 4;
	}

	// Name of the model
	std::string modelName() const
	{
		return "homography";
	}

	bool isErrorSquared() const
	{
		return true;
	}

	// The method estimating the homography from a minimal sample
	bool estimateModel(const cv::Mat& data, // All data points
		const int *sample, // The current sample consising of four correspondences
		std::vector<Homography>* models) const // Estimated model(s)
	{
		static const auto M = sampleSize(); // Size of a minimal sample
		static std::vector<cv::Point2d> points_src(M), // Points in the first image
			points_dst(M); // Points in the second image

		// Copying the sample coordinates
		for (auto i = 0; i < M; ++i)
		{
			points_src[i].x = static_cast<double>(data.at<double>(sample[i], 0));
			points_src[i].y = static_cast<double>(data.at<double>(sample[i], 1));
			points_dst[i].x = static_cast<double>(data.at<double>(sample[i], 3));
			points_dst[i].y = static_cast<double>(data.at<double>(sample[i], 4));
		}

		// Estimate the homography using OpenCV's function
		// TODO: replace it by a faster algorithm
		cv::Mat H = findHomography(points_src, points_dst);
		H.convertTo(H, CV_64F); // Convert to double because OpenCV returns float matrix

		// If the estimation failed, there are no rows in H 
		if (H.rows == 0)
			return false;

		Homography model(H); // Initilize the model
		models->emplace_back(model);
		return true;
	}
	
	bool estimateModelNonminimalWeighted(const cv::Mat& data, // All data points
		const int *sample, // The current sample consising of four correspondences
		const double *weights, // The weights of the points
		size_t sample_number, // Number of correspondences
		std::vector<Homography>* models) const // Estimated model(s)
	{
		// If there are fewer points than the minimal sample interrupt the estimation
		static const size_t M = sampleSize();

		if (sample_number < M)
			return false;

		const size_t N = sample_number; // Number of points
		cv::Mat normalized_points(data.size(), data.type()); // The normalized point coordinates
		cv::Mat normalizing_transformation_src, normalizing_transformation_dst; // The normalizing transformations in the first and second images
		normalizing_transformation_src = normalizing_transformation_dst = cv::Mat::zeros(3, 3, data.type());
	
		// Normalize the points
		if (!normalizePoints(data,
			sample,
			N,
			normalized_points,
			normalizing_transformation_src,
			normalizing_transformation_dst))
			return false;

		// Estimate the homography using the normalized DLT algorithm
		solverFourPoint(normalized_points,
			NULL,
			weights,
			N,
			models);

		// Denormalize the homography matrix
		models->at(0).descriptor = normalizing_transformation_dst.inv() * 
			models->at(0).descriptor * 
			normalizing_transformation_src;
		return true;
	}

	bool normalizePoints(const cv::Mat& data_,
		const int *sample_,
		size_t sample_size_,
		cv::Mat &normalized_points_,
		cv::Mat &normalizing_transformation_src_,
		cv::Mat &normalizing_transformation_dst_) const
	{
		const auto columns = data_.cols;
		double *normalized_points_ptr = reinterpret_cast<double *>(normalized_points_.data); 
		const double *data_ptr = reinterpret_cast<double*>(data_.data);

		// Get the mass point of each point cloud
		double mass1[2], mass2[2];
		mass1[0] = mass1[1] = mass2[0] = mass2[1] = 0.0f;

		for (auto i = 0; i < sample_size_; ++i)
		{
			if (sample_[i] >= data_.rows)
				return false;

			// Pointer to the current point's row
			const double *d_idx = data_ptr + columns * sample_[i];

			mass1[0] += *(d_idx);
			mass1[1] += *(d_idx + 1);
			mass2[0] += *(d_idx + 3);
			mass2[1] += *(d_idx + 4);
		}

		mass1[0] /= sample_size_;
		mass1[1] /= sample_size_;
		mass2[0] /= sample_size_;
		mass2[1] /= sample_size_;

		// Get the average distance of each point cloud from its mass point
		double mean_distance_src = 0.0, 
			mean_distance_dst = 0.0;

		for (auto i = 0; i < sample_size_; ++i)
		{
			// Pointer to the current point's row
			const double *d_idx = data_ptr + columns * sample_[i];

			const double x1 = *(d_idx);
			const double y1 = *(d_idx + 1);
			const double x2 = *(d_idx + 3);
			const double y2 = *(d_idx + 4);

			const double dx1 = mass1[0] - x1;
			const double dy1 = mass1[1] - y1;
			const double dx2 = mass2[0] - x2;
			const double dy2 = mass2[1] - y2;

			mean_distance_src += sqrt(dx1 * dx1 + dy1 * dy1);
			mean_distance_dst += sqrt(dx2 * dx2 + dy2 * dy2);
		}

		mean_distance_src /= sample_size_;
		mean_distance_dst /= sample_size_;

		static const double sqrt_2 = sqrt(2.0);
		const double ratio1 = sqrt_2 / mean_distance_src; // Normalizing value to set the average distance to sqrt(2)
		const double ratio2 = sqrt_2 / mean_distance_dst;

		// Compute the normalized coordinates
		int n_idx = 0;
		for (auto i = 0; i < sample_size_; ++i)
		{
			// Pointer to the current point's row
			const double *d_idx = data_ptr + columns * sample_[i];

			const double x1 = *(d_idx);
			const double y1 = *(d_idx + 1);
			const double x2 = *(d_idx + 3);
			const double y2 = *(d_idx + 4);

			*normalized_points_ptr++ = (x1 - mass1[0]) * ratio1;
			*normalized_points_ptr++ = (y1 - mass1[1]) * ratio1;
			*normalized_points_ptr++ = 1;
			*normalized_points_ptr++ = (x2 - mass2[0]) * ratio2;
			*normalized_points_ptr++ = (y2 - mass2[1]) * ratio2;
			*normalized_points_ptr++ = 1;
		}
		
		normalizing_transformation_src_ = (cv::Mat_<double>(3, 3) << ratio1, 0, -ratio1 * mass1[0],
			0, ratio1, -ratio1 * mass1[1],
			0, 0, 1);

		normalizing_transformation_dst_ = (cv::Mat_<double>(3, 3) << ratio2, 0, -ratio2 * mass2[0],
			0, ratio2, -ratio2 * mass2[1],
			0, 0, 1);
		return true;
	}

	bool estimateModelNonminimal(const cv::Mat& data,
		const int *sample,
		size_t sample_number,
		std::vector<Homography>* models) const
	{
		if (sample_number < sampleSize())
			return false;
		
		const auto N = sample_number;

		std::vector<cv::Point2d> pts1(N);
		std::vector<cv::Point2d> pts2(N);

		for (auto i = 0; i < N; ++i)
		{
			pts1[i].x = static_cast<double>(data.at<double>(sample[i], 0));
			pts1[i].y = static_cast<double>(data.at<double>(sample[i], 1));
			pts2[i].x = static_cast<double>(data.at<double>(sample[i], 3));
			pts2[i].y = static_cast<double>(data.at<double>(sample[i], 4));
		}

		cv::Mat H = findHomography(pts1, pts2, NULL, 0);
		if (H.cols == 0)
			return false;

		H.convertTo(H, CV_64F);

		Homography model;
		model.descriptor = H / norm(H);
		models->emplace_back(model);
		return true;
	}

	double error(const cv::Mat& point, 
		const Homography& model) const
	{
		return error(point, 
			model.descriptor);
	}

	double error(const cv::Mat& point, 
		const cv::Mat& descriptor) const
	{
		const double* s = reinterpret_cast<double*>(point.data);
		const double* p = reinterpret_cast<double*>(descriptor.data);

		const double x1 = *s;
		const double y1 = *(s + 1);
		const double x2 = *(s + 3);
		const double y2 = *(s + 4);

		const double t1 = *p * x1 + *(p + 1) * y1 + *(p + 2);
		const double t2 = *(p + 3) * x1 + *(p + 4) * y1 + *(p + 5);
		const double t3 = *(p + 6) * x1 + *(p + 7) * y1 + *(p + 8);

		const double d1 = x2 - (t1 / t3);
		const double d2 = y2 - (t2 / t3);

		return d1*d1 + d2*d2;
	}

	bool solverFourPoint(const cv::Mat& data,
		const int *sample,
		const double* weights,
		size_t sample_number,
		std::vector<Homography>* models) const
	{
		// model calculation 
		const int M = sample_number;
		const int cols = data.cols;
		double *data_ptr = reinterpret_cast<double*>(data.data);

		double x1, y1, x2, y2, w = 1;
		int smpl;

		cv::Mat_<double> A(2 * M, 9);
		double * A_ptr = (double*)A.data;

		for (int i = 0; i < sample_number; i++)
		{
			if (sample != NULL)
				smpl = cols * sample[i];
			else
				smpl = cols * i;
			if (weights != NULL)
				w = weights[i];

			x1 = data_ptr[smpl];
			y1 = data_ptr[smpl + 1];

			x2 = data_ptr[smpl + 3];
			y2 = data_ptr[smpl + 4];

			(*A_ptr++) = -x1 * w;
			(*A_ptr++) = -y1 * w;
			(*A_ptr++) = -w;
			(*A_ptr++) = 0;
			(*A_ptr++) = 0;
			(*A_ptr++) = 0;
			(*A_ptr++) = x2 * x1 * w;
			(*A_ptr++) = x2 * y1 * w;
			(*A_ptr++) = x2 * w;

			(*A_ptr++) = 0;
			(*A_ptr++) = 0;
			(*A_ptr++) = 0;
			(*A_ptr++) = -x1 * w;
			(*A_ptr++) = -y1 * w;
			(*A_ptr++) = -w;
			(*A_ptr++) = y2 * x1 * w;
			(*A_ptr++) = y2 * y1 * w;
			(*A_ptr++) = y2 * w;
		}

		cv::Mat AtA = A.t() * A;
		cv::Mat evals, evecs;
		cv::eigen(AtA, evals, evecs);

		Homography model;
		model.descriptor = cv::Mat(3, 3, CV_64F);
		memcpy(model.descriptor.data, evecs.row(evecs.rows - 1).data, sizeof(double) * 9);
		models->emplace_back(model);

		return true;
	}

	// Enable a quick check to see if the model is valid. This can be a geometric
	// check or some other verification of the model structure.
	bool validModel(const Model& model) const
	{ 
		return model.descriptor.rows == 3 && 
			model.descriptor.cols == 3;
	}

};