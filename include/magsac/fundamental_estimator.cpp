#include "stdafx.h"

#include <iostream>
#include <math.h>
#include <chrono>
#include <random>
#include <vector>
#include <cv.h>

#include "estimator.h"

using namespace std;
using namespace cv;

struct FundamentalMatrix
{
	cv::Mat descriptor;

	FundamentalMatrix() {}
	FundamentalMatrix(const FundamentalMatrix& other)
	{
		descriptor = other.descriptor.clone();
	}
};

// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
class FundamentalMatrixEstimator : public Estimator < Mat, FundamentalMatrix >
{
protected:

public:
	FundamentalMatrixEstimator() {}
	~FundamentalMatrixEstimator() {}

	int sampleSize() const { 
		return 7;
	}

	std::string modelName() const
	{
		return "fundamental matrix";
	}

	void finalize(Model &model) const
	{
		cv::Mat F = model.descriptor;
		cv::SVD svd(F);

		if (abs(svd.w.at<double>(2)) < FLT_EPSILON)
			return;

		cv::Mat e = cv::Mat::zeros(3, 3, F.type());
		e.at<double>(0, 0) = svd.w.at<double>(0);
		e.at<double>(1, 1) = svd.w.at<double>(1);

		cv::Mat F2 = svd.u * e *  svd.vt;
		memcpy(model.descriptor.data, F2.data, sizeof(double) * 9);
	}

	bool estimateModel(
		const cv::Mat& data,
		const int *sample,
		std::vector<FundamentalMatrix>* models) const
	{
		// model calculation 
		int M = sampleSize();

		solverSevenPoint(data, sample, M, models);
		if (models->size() == 0)
			return false;
		return true;
	}
	
	bool estimateModelNonminimalWeighted(
		const cv::Mat& data,
		const int *sample,
		const double *weights,
		size_t sample_number,
		std::vector<FundamentalMatrix>* models) const
	{
		// model calculation 
		int M = sample_number;

		if (sample_number < 9)
			return false;

		const int N = sample_number;
		Mat normalized_points(data.size(), data.type());
		Mat T1, T2;
		T1 = T2 = Mat::zeros(3, 3, data.type());

		if (!normalizePoints(data, sample, sample_number, normalized_points, T1, T2))
			return false;
		
		solverEightPoint(normalized_points,
			NULL,
			weights,
			sample_number, 
			models);

		for (size_t model_idx = 0; model_idx < models->size(); ++model_idx)
			models->at(model_idx).descriptor = T2.t() * models->at(model_idx).descriptor * T1;

		return true;
	}

	bool estimateModelNonminimal(
		const cv::Mat& data,
		const int *sample,
		size_t sample_number,
		std::vector<FundamentalMatrix>* models) const
	{
		// model calculation 
		int M = sample_number;

		if (sample_number < 8)
			return false;

		const int N = sample_number;
		Mat normalized_points(data.size(), data.type());
		Mat T1, T2;
		T1 = T2 = Mat::zeros(3, 3, data.type());
		
		if (!normalizePoints(data, sample, sample_number, normalized_points, T1, T2))
			return false;

		solverEightPoint(normalized_points,
			NULL,
			NULL, 
			sample_number,
			models);

		for (size_t model_idx = 0; model_idx < models->size(); ++model_idx)
			models->at(model_idx).descriptor = T2.t() * models->at(model_idx).descriptor * T1;

		return true;
	}

	bool normalizePoints(
		const cv::Mat& data,
		const int *sample,
		int sample_number,
		cv::Mat &normalized_points,
		cv::Mat &T1,
		cv::Mat &T2) const
	{
		const int cols = data.cols;
		double *npts = reinterpret_cast<double *>(normalized_points.data); 
		const double *d = (double *)data.data;

		// Get mass point
		double mass1[2], mass2[2];
		mass1[0] = mass1[1] = mass2[0] = mass2[1] = 0.0f;

		for (int i = 0; i < sample_number; ++i)
		{
			//cout << sample_number << " " << sample[i] << " " << data.rows << endl;
			if (sample[i] >= data.rows)
				return false;

			const double *d_idx = d + cols * sample[i];

			mass1[0] += *(d_idx);
			mass1[1] += *(d_idx + 1);
			mass2[0] += *(d_idx + 3);
			mass2[1] += *(d_idx + 4);
		}

		mass1[0] /= sample_number;
		mass1[1] /= sample_number;
		mass2[0] /= sample_number;
		mass2[1] /= sample_number;

		// Get mean distance
		double mean_dist1 = 0.0f, mean_dist2 = 0.0f;
		for (int i = 0; i < sample_number; ++i)
		{
			const double *d_idx = d + cols * sample[i];

			const double x1 = *(d_idx);
			const double y1 = *(d_idx + 1);
			const double x2 = *(d_idx + 3);
			const double y2 = *(d_idx + 4);

			const double dx1 = mass1[0] - x1;
			const double dy1 = mass1[1] - y1;
			const double dx2 = mass2[0] - x2;
			const double dy2 = mass2[1] - y2;

			mean_dist1 += sqrt(dx1 * dx1 + dy1 * dy1);
			mean_dist2 += sqrt(dx2 * dx2 + dy2 * dy2);
		}

		mean_dist1 /= sample_number;
		mean_dist2 /= sample_number;

		const double ratio1 = sqrt(2) / mean_dist1;
		const double ratio2 = sqrt(2) / mean_dist2;

		// Compute the normalized coordinates
		int n_idx = 0;
		for (int i = 0; i < sample_number; ++i)
		{
			const double *d_idx = d + cols * sample[i];

			const double x1 = *(d_idx);
			const double y1 = *(d_idx + 1);
			const double x2 = *(d_idx + 3);
			const double y2 = *(d_idx + 4);

			*npts++ = (x1 - mass1[0]) * ratio1;
			*npts++ = (y1 - mass1[1]) * ratio1;
			*npts++ = 1;
			*npts++ = (x2 - mass2[0]) * ratio2;
			*npts++ = (y2 - mass2[1]) * ratio2;
			*npts++ = 1;
		}

		T1 = (Mat_<double>(3, 3) << ratio1, 0, -ratio1 * mass1[0],
			0, ratio1, -ratio1 * mass1[1],
			0, 0, 1);

		T2 = (Mat_<double>(3, 3) << ratio2, 0, -ratio2 * mass2[0],
			0, ratio2, -ratio2 * mass2[1],
			0, 0, 1);
		return true;
	}

	// Calculate the Sampson-distance of a point and a fundamental matrix
	double error(
		const cv::Mat& point, 
		const FundamentalMatrix& model) const
	{
		return error(point, model.descriptor);
	}

	// Calculate the Sampson-distance of a point and a fundamental matrix
	double error(
		const cv::Mat& point,
		const cv::Mat& descriptor) const
	{
		const double* p = (double*)descriptor.data;
		const double* s = (double*)point.data;
		const double x1 = *s;
		const double y1 = *(s + 1);
		const double x2 = *(s + 3);
		const double y2 = *(s + 4);

		double rxc = *(p)* x2 + *(p + 3) * y2 + *(p + 6);
		double ryc = *(p + 1) * x2 + *(p + 4) * y2 + *(p + 7);
		double rwc = *(p + 2) * x2 + *(p + 5) * y2 + *(p + 8);
		double r = (x1 * rxc + y1 * ryc + rwc);
		double rx = *(p)* x1 + *(p + 1) * y1 + *(p + 2);
		double ry = *(p + 3) * x1 + *(p + 4) * y1 + *(p + 5);

		return sqrt(static_cast<double>(r * r / (rxc * rxc + ryc * ryc + rx * rx + ry * ry)));
	}

	bool solverEightPoint(const cv::Mat& data,
		const int *sample,
		const double *weights,
		int sample_number,
		vector<FundamentalMatrix>* models) const
	{
		double f[9];
		cv::Mat evals(1, 9, CV_64F), evecs(9, 9, CV_64F);
		cv::Mat A(sample_number, 9, CV_64F);
		cv::Mat F(3, 3, CV_64F, f);
		int i;

		const int M = sample_number;
		const int cols = data.cols;
		double *data_ptr = reinterpret_cast<double*>(data.data);

		double x0, y0, x1, y1, w = 1;

		// form a linear system: i-th row of A(=a) represents
		// the equation: (m2[i], 1)'*F*(m1[i], 1) = 0
		for (i = 0; i < sample_number; i++)
		{
			int smpl;
			if (sample != NULL)
				smpl = cols * sample[i];
			else
				smpl = cols * i;

			if (weights != NULL)
				w = sample != NULL ? weights[sample[i]] : weights[i];

			x0 = data_ptr[smpl];
			y0 = data_ptr[smpl + 1];

			x1 = data_ptr[smpl + 3];
			y1 = data_ptr[smpl + 4];

			A.at<double>(i, 0) = w * x1 * x0;
			A.at<double>(i, 1) = w * x1 * y0;
			A.at<double>(i, 2) = w * x1;
			A.at<double>(i, 3) = w * y1 * x0;
			A.at<double>(i, 4) = w * y1 * y0;
			A.at<double>(i, 5) = w * y1;
			A.at<double>(i, 6) = w * x0;
			A.at<double>(i, 7) = w * y0;
			A.at<double>(i, 8) = w;
		}

		// A*(f11 f12 ... f33)' = 0 is singular (7 equations for 9 variables), so
		// the solution is linear subspace of dimensionality 2.
		// => use the last two singular vectors as a basis of the space
		// (according to SVD properties)		
		cv::Mat cov = A.t() * A;
		eigen(cov, evals, evecs);

		for (i = 0; i < 9; ++i)
			f[i] = evecs.at<double>(8, i);

		// Enforce rank-2 constraint
		cv::SVD svd2(F);
		cv::Mat e = cv::Mat::zeros(3, 3, F.type());
		e.at<double>(0, 0) = svd2.w.at<double>(0);
		e.at<double>(1, 1) = svd2.w.at<double>(1);

		F = svd2.u * e *  svd2.vt;

		/* orient. constr. */
		if (!isOrientationValid(F, data, sample, sample_number)) {
			return false;
		}
		
		Model model;
		model.descriptor = F;
		models->push_back(model);
		return true;
	}

	bool solverSevenPoint(const Mat& data,
		const int *sample,
		int sample_number,
		vector<FundamentalMatrix>* models) const
	{
		double a[7 * 9], v[9 * 9], c[4], r[3];
		double *f1, *f2;
		double t0, t1, t2;
		cv::Mat evals, evecs(9, 9, CV_64F, v);
		cv::Mat A(7, 9, CV_64F, a);
		cv::Mat coeffs(1, 4, CV_64F, c);
		cv::Mat roots(1, 3, CV_64F, r);
		int i, k, n;

		// form a linear system: i-th row of A(=a) represents
		// the equation: (m2[i], 1)'*F*(m1[i], 1) = 0
		for (i = 0; i < 7; i++)
		{
			const double x0 = data.at<double>(sample[i], 0), y0 = data.at<double>(sample[i], 1);
			const double x1 = data.at<double>(sample[i], 3), y1 = data.at<double>(sample[i], 4);

			a[i * 9 + 0] = x1*x0;
			a[i * 9 + 1] = x1*y0;
			a[i * 9 + 2] = x1;
			a[i * 9 + 3] = y1*x0;
			a[i * 9 + 4] = y1*y0;
			a[i * 9 + 5] = y1;
			a[i * 9 + 6] = x0;
			a[i * 9 + 7] = y0;
			a[i * 9 + 8] = 1;
		}

		// A*(f11 f12 ... f33)' = 0 is singular (7 equations for 9 variables), so
		// the solution is linear subspace of dimensionality 2.
		// => use the last two singular vectors as a basis of the space
		// (according to SVD properties)
		cv::Mat cov = A.t() * A;
		eigen(cov, evals, evecs);
		f1 = v + 7 * 9;
		f2 = v + 8 * 9;

		// f1, f2 is a basis => lambda*f1 + mu*f2 is an arbitrary f. matrix.
		// as it is determined up to a scale, normalize lambda & mu (lambda + mu = 1),
		// so f ~ lambda*f1 + (1 - lambda)*f2.
		// use the additional constraint det(f) = det(lambda*f1 + (1-lambda)*f2) to find lambda.
		// it will be a cubic equation.
		// find c - polynomial coefficients.
		for (i = 0; i < 9; i++)
			f1[i] -= f2[i];

		t0 = f2[4] * f2[8] - f2[5] * f2[7];
		t1 = f2[3] * f2[8] - f2[5] * f2[6];
		t2 = f2[3] * f2[7] - f2[4] * f2[6];

		c[3] = f2[0] * t0 - f2[1] * t1 + f2[2] * t2;

		c[2] = f1[0] * t0 - f1[1] * t1 + f1[2] * t2 -
			f1[3] * (f2[1] * f2[8] - f2[2] * f2[7]) +
			f1[4] * (f2[0] * f2[8] - f2[2] * f2[6]) -
			f1[5] * (f2[0] * f2[7] - f2[1] * f2[6]) +
			f1[6] * (f2[1] * f2[5] - f2[2] * f2[4]) -
			f1[7] * (f2[0] * f2[5] - f2[2] * f2[3]) +
			f1[8] * (f2[0] * f2[4] - f2[1] * f2[3]);

		t0 = f1[4] * f1[8] - f1[5] * f1[7];
		t1 = f1[3] * f1[8] - f1[5] * f1[6];
		t2 = f1[3] * f1[7] - f1[4] * f1[6];

		c[1] = f2[0] * t0 - f2[1] * t1 + f2[2] * t2 -
			f2[3] * (f1[1] * f1[8] - f1[2] * f1[7]) +
			f2[4] * (f1[0] * f1[8] - f1[2] * f1[6]) -
			f2[5] * (f1[0] * f1[7] - f1[1] * f1[6]) +
			f2[6] * (f1[1] * f1[5] - f1[2] * f1[4]) -
			f2[7] * (f1[0] * f1[5] - f1[2] * f1[3]) +
			f2[8] * (f1[0] * f1[4] - f1[1] * f1[3]);

		c[0] = f1[0] * t0 - f1[1] * t1 + f1[2] * t2;

		// solve the cubic equation; there can be 1 to 3 roots ...
		n = cv::solveCubic(coeffs, roots);

		if (n < 1 || n > 3)
			return false;

		for (k = 0; k < n; k++)
		{
			double f[9];
			cv::Mat F(3, 3, CV_64F, f);

			// for each root form the fundamental matrix
			double lambda = r[k], mu = 1.;
			double s = f1[8] * r[k] + f2[8];

			// normalize each matrix, so that F(3,3) (~fmatrix[8]) == 1
			if (fabs(s) > DBL_EPSILON)
			{
				mu = 1. / s;
				lambda *= mu;

				for (int i = 0; i < 8; ++i)
					f[i] = f1[i] * lambda + f2[i] * mu;
				f[8] = 1.0;

				/* orient. constr. */
				if (!isOrientationValid(F, data, sample, sample_number)) {
					continue;
				}

				Model model;
				model.descriptor = F;
				models->push_back(model);
			}
		}

		return true;
	}

	// Enable a quick check to see if the model is valid. This can be a geometric
	// check or some other verification of the model structure.
	bool validModel(const Model& model) const
	{
		return model.descriptor.rows == 3 &&
			model.descriptor.cols == 3;
	}

	/************** oriented constraints ******************/
	void getEpipole(
		cv::Mat &epipole_, 
		const cv::Mat &fundamental_matrix_) const
	{
		epipole_ = fundamental_matrix_.row(0).cross(fundamental_matrix_.row(2));

		for (auto i = 0; i < 3; i++)
			if ((epipole_.at<double>(i) > 1.9984e-15) || (epipole_.at<double>(i) < -1.9984e-15)) return;
		epipole_ = fundamental_matrix_.row(1).cross(fundamental_matrix_.row(2));
	}

	double getOrientationSign(
		const cv::Mat &fundamental_matrix_, 
		const cv::Mat &epipole_, 
		const cv::Mat &point_) const
	{
		const double s1 = fundamental_matrix_.at<double>(0,0) * point_.at<double>(3) +
			fundamental_matrix_.at<double>(1,0) * point_.at<double>(4) +
			fundamental_matrix_.at<double>(2,0) * point_.at<double>(5);

		const double s2 = epipole_.at<double>(1) * point_.at<double>(2) -
			epipole_.at<double>(2) * point_.at<double>(1);
		return s1 * s2;
	}

	bool isOrientationValid(
		const cv::Mat &fundamental_matrix_, 
		const cv::Mat &data_, 
		const int *sample_, 
		int sample_size_) const
	{
		Mat epipole;
		double sig, sig1;
		getEpipole(epipole, fundamental_matrix_);

		if (sample_ == nullptr)
		{
			sig1 = getOrientationSign(fundamental_matrix_,
				epipole,
				data_.row(0));

			for (auto i = 1; i < sample_size_; i++)
			{
				sig = getOrientationSign(fundamental_matrix_,
					epipole,
					data_.row(i));

				if (sig1 * sig < 0)
					return false;
			}
		} else
		{
			sig1 = getOrientationSign(fundamental_matrix_,
				epipole,
				data_.row(sample_[0]));

			for (auto i = 1; i < sample_size_; i++)
			{
				sig = getOrientationSign(fundamental_matrix_,
					epipole,
					data_.row(sample_[i]));

				if (sig1 * sig < 0)
					return false;
			}
		}
		return true;
	}
};