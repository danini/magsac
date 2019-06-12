#include <string.h>
#include <fstream>
#include <vector>
#include <ctime>
#include <chrono>
#include <cstddef>

#include <cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "utils.h"
#include "magsac.h"
#include "fundamental_estimator.cpp"
#include "homography_estimator.cpp"

enum SceneType { FundamentalMatrixScene, HomographyScene };

// A method applying MAGSAC for fundamental matrix estimation to one of the built-in scenes
void testFundamentalMatrixFitting(
	double ransac_confidence_,
	double sigma_max_,
	std::string test_scene_,
	bool draw_results_ = false,
	double drawing_threshold_ = 2);

// A method applying MAGSAC for homography estimation to one of the built-in scenes
void testHomographyFitting(
	double ransac_confidence_,
	double sigma_max_,
	std::string test_scene_,
	bool draw_results_ = false,
	double drawing_threshold_ = 2);

// The names of built-in scenes
std::vector<std::string> getAvailableTestScenes(
	SceneType scene_type_);

int main(int argc, const char* argv[])
{
	/*
		This is an example showing how MAGSAC is applied to homography or fundamental matrix estimation tasks.
		The paper is readable here: https://arxiv.org/pdf/1803.07469.pdf
		This implementation is not the one used in the experiments of the paper. 
	*/
	const double ransac_confidence = 0.99; // The required confidence in the results
	const double sigma_max = 10; // The maximum sigma value allowed in MAGSAC
	const bool draw_results = true; // A flag to draw and show the results 
	// The inlier threshold for visualization. This threshold is not used by the algorithm,
	// it is simply for selecting the inliers to be drawn after MAGSAC finished.
	const double drawing_threshold = 1.0;

	// Test scenes for homography estimation
	for (const auto& scene : getAvailableTestScenes(SceneType::HomographyScene))
		testHomographyFitting(ransac_confidence,
			sigma_max, // The maximum sigma value allowed in MAGSAC
			scene, // The scene type
			draw_results, // A flag to draw and show the results 
			drawing_threshold); // The inlier threshold for visualization.

	// Test scenes for fundamental matrix estimation
	for (const auto& scene : getAvailableTestScenes(SceneType::FundamentalMatrixScene))
		testFundamentalMatrixFitting(ransac_confidence, // The required confidence in the results
			sigma_max, // The maximum sigma value allowed in MAGSAC
			scene, // The scene type
			draw_results, // A flag to draw and show the results 
			drawing_threshold); // The inlier threshold for visualization.


	return 0;
} 

std::vector<std::string> getAvailableTestScenes(SceneType scene_type_)
{
	switch (scene_type_)
	{
	case SceneType::HomographyScene: // Available test scenes for homography estimation
		return { "LePoint1", "LePoint2", "LePoint3", 
			"graf", "ExtremeZoom", "city", 
			"CapitalRegion", "BruggeTower", "BruggeSquare", 
			"BostonLib", "boat", "adam", 
			"WhiteBoard", "Eiffel", "Brussels", 
			"Boston"};

	case SceneType::FundamentalMatrixScene:
		return {"corr", "booksh", "box",
			"castle", "graff", "head",
			"kampa", "leafs", "plant",
			"rotunda", "shout", "valbonne",
			"wall", "wash", "zoom",
			"Kyoto", "barrsmith", "bonhall",
			"bonython", "elderhalla", "elderhallb",
			"hartley", "johnssonb", "ladysymon", 
			"library", "napiera", "napierb", 
			"nese", "oldclassicswing", "physics", 
			"sene", "unihouse", "unionhouse", 
			"boxesandbooks", "glasscaseb", "stairs"};

	default:
		break;
	}
}

void testFundamentalMatrixFitting(
	double ransac_confidence_,
	double sigma_max_,
	std::string test_scene_,
	bool draw_results_,
	double drawing_threshold_)
{
	printf("Processed scene = '%s'.\n", test_scene_.c_str());

	// Load the images of the current test scene
	cv::Mat image1 = cv::imread("data/fundamental_matrix/" + test_scene_ + "A.png");
	cv::Mat image2 = cv::imread("data/fundamental_matrix/" + test_scene_ + "B.png");
	if (image1.cols == 0)
	{
		image1 = cv::imread("data/fundamental_matrix/" + test_scene_ + "A.jpg");
		image2 = cv::imread("data/fundamental_matrix/" + test_scene_ + "B.jpg");
	}

	if (image1.cols == 0)
	{
		fprintf(stderr, "A problem occured when loading the images for test scene '%s'\n", test_scene_.c_str());
		return;
	}

	cv::Mat points; // The point correspondences, each is of format x1 y1 1 x2 y2 1
	std::vector<int> ground_truth_labels; // The ground truth labeling provided in the dataset

	// A function loading the points from files
	readAnnotatedPoints("data/fundamental_matrix/" + test_scene_ + "_pts.txt",
		points,
		ground_truth_labels);

	// The number of points in the datasets
	const size_t N = points.rows; // The number of points in the scene

	if (N == 0) // If there are no points, return
	{
		fprintf(stderr, "A problem occured when loading the annotated points for test scene '%s'\n", test_scene_.c_str());
		return; 
	}

	FundamentalMatrixEstimator estimator; // The robust homography estimator class containing the function for the fitting and residual calculation
	FundamentalMatrix model; // The estimated model

	// In this used datasets, the manually selected inliers are not all inliers but a subset of them.
	// Therefore, the manually selected inliers are augmented as follows: 
	// (i) First, the implied model is estimated from the manually selected inliers.
	// (ii) Second, the inliers of the ground truth model are selected.
	refineManualLabeling<FundamentalMatrix, FundamentalMatrixEstimator>(
		points,
		ground_truth_labels,
		estimator,
		0.35); // Threshold value from the LO*-RANSAC paper

	// Select the inliers from the labeling
	std::vector<int> ground_truth_inliers = getSubsetFromLabeling(ground_truth_labels, 1);
	const size_t I = static_cast<double>(ground_truth_inliers.size());

	printf("Estimated model = '%s'.\n", estimator.modelName().c_str());
	printf("Number of correspondences loaded = %d.\n", static_cast<int>(N));
	printf("Number of ground truth inliers = %d.\n", static_cast<int>(I));
	printf("Theoretical RANSAC iteration number at %.2f confidence = %d.\n",
		ransac_confidence_, static_cast<int>(log(1.0 - ransac_confidence_) / log(1.0 - pow(static_cast<double>(I) / static_cast<double>(N), 4))));
	
	MAGSAC<FundamentalMatrixEstimator, FundamentalMatrix> magsac;
	magsac.setSigmaMax(sigma_max_); // The maximum noise scale sigma allowed
	magsac.setCoreNumber(4); // The number of cores used to speed up sigma-consensus
	magsac.setPartitionNumber(5); // The number partitions used for speeding up sigma consensus. As the value grows, the algorithm become slower and, usually, more accurate.

	int iteration_number = 0; // Number of iterations required

	std::chrono::time_point<std::chrono::system_clock> end,
		start = std::chrono::system_clock::now();
	magsac.run(points, // The data points
		ransac_confidence_, // The required confidence in the results
		estimator, // The used estimator
		model, // The estimated model
		iteration_number); // The number of iterations
	end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::time_t end_time = std::chrono::system_clock::to_time_t(end);

	printf("Actual number of iterations drawn by MAGSAC at %.2f confidence: %d\n", ransac_confidence_, iteration_number);
	printf("Elapsed time: %f secs\n", elapsed_seconds.count());

	// Compute the RMSE given the ground truth inliers
	double rmse = 0, error;
	for (const auto &inlier_idx : ground_truth_inliers)
	{
		error = estimator.error(points.row(inlier_idx), model);
		rmse += error * error;
	}
	rmse = sqrt(rmse / static_cast<double>(I));
	printf("RMSE error: %f px\n", rmse);

	// Visualization part.
	// Inliers are selected using threshold and the estimated model. 
	// This part is not necessary and is only for visualization purposes. 
	if (draw_results_)
	{
		std::vector<int> obtained_labeling(points.rows, 0);

		for (auto pt_idx = 0; pt_idx < points.rows; ++pt_idx)
		{
			// Computing the residual of the point given the estimated model
			auto residual = estimator.error(points.row(pt_idx),
				model.descriptor);
			
			// Change the label to 'inlier' if the residual is smaller than the threshold
			if (drawing_threshold_ >= residual)
				obtained_labeling[pt_idx] = 1;
		}

		// Draw the matches to the images
		cv::Mat out_image;
		drawMatches<double>(points, obtained_labeling, image1, image2, out_image);

		// Show the matches
		std::string window_name = "Visualization with threshold = " + std::to_string(drawing_threshold_) + " px";
		printf("Press a button to continue.\n\n");
		showImage(out_image,
			window_name,
			1600,
			900);
		out_image.release();
	}

	// Clean up the memory occupied by the images
	image1.release();
	image2.release();
}

void testHomographyFitting(
	double ransac_confidence_,
	double sigma_max_,
	std::string test_scene_,
	bool draw_results_,
	double drawing_threshold_)
{
	printf("Processed scene = '%s'.\n", test_scene_.c_str());

	// Load the images of the current test scene
	cv::Mat image1 = cv::imread("data/homography/" + test_scene_ + "A.png");
	cv::Mat image2 = cv::imread("data/homography/" + test_scene_ + "B.png");
	if (image1.cols == 0)
	{
		image1 = cv::imread("data/homography/" + test_scene_ + "A.jpg");
		image2 = cv::imread("data/homography/" + test_scene_ + "B.jpg");
	}

	if (image1.cols == 0)
	{
		fprintf(stderr, "A problem occured when loading the images for test scene '%s'\n", test_scene_.c_str());
		return;
	}

	cv::Mat points; // The point correspondences, each is of format x1 y1 1 x2 y2 1
	std::vector<int> ground_truth_labels; // The ground truth labeling provided in the dataset

	// A function loading the points from files
	readAnnotatedPoints("data/homography/" + test_scene_ + "_pts.txt", 
		points, 
		ground_truth_labels);

	// The number of points in the datasets
	const size_t N = points.rows; // The number of points in the scene

	if (N == 0) // If there are no points, return
	{
		fprintf(stderr, "A problem occured when loading the annotated points for test scene '%s'\n", test_scene_.c_str());
		return;
	}

	RobustHomographyEstimator estimator; // The robust homography estimator class containing the function for the fitting and residual calculation
	Homography model; // The estimated model

	// In this used datasets, the manually selected inliers are not all inliers but a subset of them.
	// Therefore, the manually selected inliers are augmented as follows: 
	// (i) First, the implied model is estimated from the manually selected inliers.
	// (ii) Second, the inliers of the ground truth model are selected.
	refineManualLabeling<Homography, RobustHomographyEstimator>(
		points,
		ground_truth_labels,
		estimator,
		2.0);

	// Select the inliers from the labeling
	std::vector<int> ground_truth_inliers = getSubsetFromLabeling(ground_truth_labels, 1);
	const size_t I = static_cast<double>(ground_truth_inliers.size());

	printf("Estimated model = '%s'.\n", estimator.modelName().c_str());
	printf("Number of correspondences loaded = %d.\n", static_cast<int>(N));
	printf("Number of ground truth inliers = %d.\n", static_cast<int>(I));
	printf("Theoretical RANSAC iteration number at %.2f confidence = %d.\n", 
		ransac_confidence_, static_cast<int>(log(1.0 - ransac_confidence_) / log(1.0 - pow(static_cast<double>(I) / static_cast<double>(N), 4))));

	MAGSAC<RobustHomographyEstimator, Homography> magsac;
	magsac.setSigmaMax(sigma_max_); // The maximum noise scale sigma allowed
	magsac.setCoreNumber(4); // The number of cores used to speed up sigma-consensus
	magsac.setPartitionNumber(10); // The number partitions used for speeding up sigma consensus. As the value grows, the algorithm become slower and, usually, more accurate.

	int iteration_number = 0; // Number of iterations required

	std::chrono::time_point<std::chrono::system_clock> end, 
		start = std::chrono::system_clock::now();
	magsac.run(points, // The data points
		ransac_confidence_, // The required confidence in the results
		estimator, // The used estimator
		model, // The estimated model
		iteration_number); // The number of iterations
	end = std::chrono::system_clock::now();
	 
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::time_t end_time = std::chrono::system_clock::to_time_t(end);

	printf("Actual number of iterations drawn by MAGSAC at %.2f confidence: %d\n", ransac_confidence_, iteration_number);
	printf("Elapsed time: %f secs\n", elapsed_seconds.count());
	 
	// Compute the RMSE given the ground truth inliers
	double rmse = 0, error;
	size_t inlier_number = 0;
	for (const auto& inlier_idx : ground_truth_inliers)
	{
		error = estimator.error(points.row(inlier_idx), model);
		rmse += error;
	}
	rmse = sqrt(rmse / static_cast<double>(I));
	printf("RMSE error: %f px\n", rmse);

	// Visualization part.
	// Inliers are selected using threshold and the estimated model. 
	// This part is not necessary and is only for visualization purposes. 
	if (draw_results_)
	{
		std::vector<int> obtained_labeling(points.rows, 0);

		for (auto pt_idx = 0; pt_idx < points.rows; ++pt_idx)
		{
			// Computing the residual of the point given the estimated model
			auto residual = sqrt(estimator.error(points.row(pt_idx),
				model.descriptor));
			
			// Change the label to 'inlier' if the residual is smaller than the threshold
			if (drawing_threshold_ >= residual)
				obtained_labeling[pt_idx] = 1;
		}

		// Draw the matches to the images
		cv::Mat out_image;
		drawMatches<double>(points, obtained_labeling, image1, image2, out_image);

		// Show the matches
		std::string window_name = "Visualization with threshold = " + std::to_string(drawing_threshold_) + " px";
		printf("Press a button to continue.\n\n");
		showImage(out_image,
			window_name,
			1600,
			900);
		out_image.release();
	}

	// Clean up the memory occupied by the images
	image1.release();
	image2.release();
}
