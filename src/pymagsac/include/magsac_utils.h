#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "estimators/estimator.h"
#include "model.h"

#include <vector>
#include <fstream>

#include <gflags/gflags.h>
#include <glog/logging.h>

/**************************************************
Declaration
**************************************************/
void readAnnotatedPoints(
	const std::string& path_,
	cv::Mat& points_,
	std::vector<int>& labels_);

void loadMatrix(
	const std::string& path_,
	cv::Mat& matrix_);

template<typename T>
void drawMatches(
	const cv::Mat& points_,
	const std::vector<int>& labeling_,
	const cv::Mat& image1_,
	const cv::Mat& image2_,
	cv::Mat& out_image_);

bool savePointsToFile(
	const std::string& path_,
	std::vector<cv::Point2d>& src_points_, 
	std::vector<cv::Point2d>& dst_points_);

bool loadPointsFromFile(
	const std::string& path_,
	std::vector<cv::Point2d>& src_points_, 
	std::vector<cv::Point2d>& dst_points_);

void showImage(const cv::Mat& image_,
	std::string window_name_,
	int max_width_ = std::numeric_limits<int>::max(),
	int max_height_ = std::numeric_limits<int>::max(),
	bool wait_ = false);

template <typename Model, typename Estimator>
void refineManualLabeling(
	const cv::Mat& points_,
	std::vector<int>& labeling_,
	const Estimator &estimator_,
	const double threshold_);

/**************************************************
Implementation
**************************************************/
template <typename Model, typename Estimator>
void refineManualLabeling(
	const cv::Mat& points_, // All data points
	std::vector<int>& labeling_, // The labeling to be refined
	const Estimator &estimator_, // The estimator object which has an error(...) function to calculate the residual of each point
	const double threshold_) // The inlier-outlier threshold
{
	// Collect the points labeled as inliers
	std::vector<size_t> inliers; // The container for the inliers
	const size_t point_number = labeling_.size(); // The number of points
	inliers.reserve(point_number);

	// Iterating through all points. If a point is labeling inliers, store it.
	for (size_t i = 0; i < point_number; ++i)
		if (labeling_[i])
			inliers.emplace_back(i);

	// Number of inliers
	const size_t inlier_number = inliers.size();

	// Estimate a model from the manually selected inliers
	std::vector<gcransac::Model> models;
	estimator_.estimateModelNonminimal(points_, // All data points
		&(inliers[0]), // The inlier indices
		inlier_number, // The number of inliers
		&models); // The estimated models

	if (models.size() != 1)
	{
		LOG(ERROR) << "A problem occured when refining the manual annotation.";
		return;
	}

	// The squared threshold used to select the new inliers
	const double squared_threshold = threshold_ * threshold_;

	// Select the new inliers of the estimated model
	for (size_t i = 0; i < point_number; ++i)
		labeling_[i] = estimator_.squaredResidual(points_.row(i), models[0]) < squared_threshold;
}

template <typename LabelType>
std::vector<LabelType> getSubsetFromLabeling(const std::vector<LabelType>& labeling_,
	const LabelType label_)
{
	std::vector<LabelType> results;
	for (auto idx = 0; idx < labeling_.size(); ++idx)
		if (labeling_[idx] == label_)
			results.emplace_back(idx);
	return results;
}

void loadMatrix(
	const std::string& path_,
	cv::Mat& matrix_)
{
	std::ifstream file(path_);

	for (int r = 0; r < matrix_.rows; ++r)
		for (int c = 0; c < matrix_.cols; ++c)
			file >> matrix_.at<double>(r, c);
	file.close();
}

bool loadPointsFromFile(
	const std::string& path_,
	std::vector<cv::Point2d>& src_points_, 
	std::vector<cv::Point2d>& dst_points_)
{
	std::ifstream infile(path_);

	if (!infile.is_open())
		return false;

	double x1, y1, x2, y2;

	std::string line;
	while (getline(infile, line))
	{
		std::istringstream split(line);
		split >> x1 >> y1 >> x2 >> y2;

		src_points_.emplace_back(cv::Point2d(x1, y1));
		dst_points_.emplace_back(cv::Point2d(x2, y2));
	}

	infile.close();
	return true;
}

bool savePointsToFile(
	const std::string& path_,
	std::vector<cv::Point2d>& src_points_, 
	std::vector<cv::Point2d>& dst_points_)
{
	std::ofstream outfile(path_);

	for (auto i = 0; i < src_points_.size(); ++i)
	{
		outfile << src_points_[i].x << " " << src_points_[i].y << " ";
		outfile << dst_points_[i].x << " " << dst_points_[i].y << " ";
		outfile << std::endl;
	}

	outfile.close();

	return true;
}

void readAnnotatedPoints(
	const std::string& path_,
	cv::Mat& points_,
	std::vector<int>& labels_)
{
	std::ifstream file(path_);

	double x1, y1, x2, y2, a, s;
	std::string str;

	std::vector<cv::Point2d> pts1;
	std::vector<cv::Point2d> pts2;
	if (path_.find("extremeview") != std::string::npos) // For extremeview dataset
	{
		while (file >> x1 >> y1 >> x2 >> y2 >> s >> s >> str >> str >> a)
		{
			pts1.emplace_back(cv::Point2d(x1, y1));
			pts2.emplace_back(cv::Point2d(x2, y2));
			labels_.emplace_back(a > 0 ? 1 : 0);
		}
	}
	else
	{
		while (file >> x1 >> y1 >> s >> x2 >> y2 >> s >> a)
		{
			pts1.emplace_back(cv::Point2d(x1, y1));
			pts2.emplace_back(cv::Point2d(x2, y2));
			labels_.emplace_back(a > 0 ? 1 : 0);
		}
	}

	file.close();

	points_.create(static_cast<int>(pts1.size()), 4, CV_64F);
	for (int i = 0; i < pts1.size(); ++i)
	{
		points_.at<double>(i, 0) = pts1[i].x;
		points_.at<double>(i, 1) = pts1[i].y;
		points_.at<double>(i, 2) = pts2[i].x;
		points_.at<double>(i, 3) = pts2[i].y;
	}
}

template<size_t _ColumnNumber>
void readPoints(
	const std::string& path_,
	cv::Mat& points_)
{
	std::ifstream file(path_);

	constexpr size_t dimension_number = _ColumnNumber / 2;
	double x1, y1, x2, y2, a, s;
	std::string str;

	std::vector<cv::Point2d> pts1;
	std::vector<cv::Point2d> pts2;

	size_t point_number = 0;
	file >> point_number;

	for (size_t i = 0; i < point_number; ++i)
	{
		for (size_t dim = 0; dim < dimension_number; ++dim)
			if (dim == 0)
				file >> x1;
			else if (dim == 1)
				file >> y1;

		for (size_t dim = 0; dim < dimension_number; ++dim)
			if (dim == 0)
				file >> x2;
			else if (dim == 1)
				file >> y2;

		pts1.emplace_back(cv::Point2d(x1, y1));
		pts2.emplace_back(cv::Point2d(x2, y2));
	}
	
	file.close();

	points_.create(static_cast<int>(pts1.size()), 4, CV_64F);
	for (int i = 0; i < pts1.size(); ++i)
	{
		points_.at<double>(i, 0) = pts1[i].x;
		points_.at<double>(i, 1) = pts1[i].y;
		points_.at<double>(i, 2) = pts2[i].x;
		points_.at<double>(i, 3) = pts2[i].y;
	}
}

template<typename T, typename LabelType>
void drawMatches(
	const cv::Mat &points_, 
	const std::vector<LabelType>& labeling_,
	const cv::Mat& image1_,
	const cv::Mat& image2_,
	cv::Mat& out_image_)
{
	const size_t N = points_.rows;
	std::vector< cv::KeyPoint > keypoints1, keypoints2;
	std::vector< cv::DMatch > matches;

	keypoints1.reserve(N);
	keypoints2.reserve(N);
	matches.reserve(N);

	// Collect the points which has label 1 (i.e. inlier)
	for (auto pt_idx = 0; pt_idx < N; ++pt_idx)
	{
		if (!labeling_[pt_idx])
			continue;

		const T x1 = points_.at<T>(pt_idx, 0);
		const T y1 = points_.at<T>(pt_idx, 1);
		const T x2 = points_.at<T>(pt_idx, 2);
		const T y2 = points_.at<T>(pt_idx, 3);
		const size_t n = keypoints1.size();

		keypoints1.emplace_back(
			cv::KeyPoint(cv::Point_<T>(x1, y1), 0));
		keypoints2.emplace_back(
			cv::KeyPoint(cv::Point_<T>(x2, y2), 0));
		matches.emplace_back(cv::DMatch(static_cast<int>(n), static_cast<int>(n), 0));
	}

	// Draw the matches using OpenCV's built-in function
	cv::drawMatches(image1_,
		keypoints1,
		image2_,
		keypoints2,
		matches,
		out_image_);
}

void showImage(const cv::Mat &image_,
	std::string window_name_,
	int max_width_,
	int max_height_,
	bool wait_)
{
	// Resizing the window to fit into the screen if needed
	int window_width = image_.cols,
		window_height = image_.rows;
	if (static_cast<double>(image_.cols) / max_width_ > 1.0 &&
		static_cast<double>(image_.cols) / max_width_ >
		static_cast<double>(image_.rows) / max_height_)
	{
		window_width = max_width_;
		window_height = static_cast<int>(window_width * static_cast<double>(image_.rows) / static_cast<double>(image_.cols));
	}
	else if (static_cast<double>(image_.rows) / max_height_ > 1.0 &&
		static_cast<double>(image_.cols) / max_width_ <
		static_cast<double>(image_.rows) / max_height_)
	{
		window_height = max_height_;
		window_width = static_cast<int>(window_height * static_cast<double>(image_.cols) / static_cast<double>(image_.rows));
	}

    cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
	cv::resizeWindow(window_name_, window_width, window_height);
	cv::imshow(window_name_, image_);
	if (wait_)
		cv::waitKey(0);
}
