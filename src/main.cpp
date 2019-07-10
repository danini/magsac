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
#include "uniform_sampler.h"
#include "fundamental_estimator.cpp"
#include "homography_estimator.cpp"

enum SceneType { FundamentalMatrixScene, HomographyScene };
enum Dataset { kusvod2, extremeview, homogr, adelaidermf, multih };

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

// A method applying OpenCV for homography estimation to one of the built-in scenes
void opencvHomographyFitting(
	double ransac_confidence_,
	double threshold_,
	std::string test_scene_,
	bool draw_results_ = false);

// A method applying OpenCV for fundamental matrix estimation to one of the built-in scenes
void opencvFundamentalMatrixFitting(
	double ransac_confidence_,
	double threshold_,
	std::string test_scene_,
	bool draw_results_ = false);

// The names of built-in scenes
std::vector<std::string> getAvailableTestScenes(
	SceneType scene_type_,
	Dataset dataset_);

// Running tests on the selected dataset
void runTest(SceneType scene_type_, 
	Dataset dataset_,
	const double ransac_confidence_,
	const bool draw_results_,
	const double drawing_threshold_);

// Returns the name of the selected dataset
std::string dataset2str(Dataset dataset_);

int main(int argc, const char* argv[])
{
	/*
		This is an example showing how MAGSAC is applied to homography or fundamental matrix estimation tasks.
		The paper is readable here: https://arxiv.org/pdf/1803.07469.pdf
		This implementation is not the one used in the experiments of the paper. 
	*/
	const double ransac_confidence = 0.99; // The required confidence in the results
	const bool draw_results = true; // A flag to draw and show the results 
	// The inlier threshold for visualization. This threshold is not used by the algorithm,
	// it is simply for selecting the inliers to be drawn after MAGSAC finished.
	const double drawing_threshold = 1.0;

	runTest(SceneType::FundamentalMatrixScene, Dataset::kusvod2, ransac_confidence, draw_results, drawing_threshold);
	runTest(SceneType::FundamentalMatrixScene, Dataset::adelaidermf, ransac_confidence, draw_results, drawing_threshold);
	runTest(SceneType::FundamentalMatrixScene, Dataset::multih, ransac_confidence, draw_results, drawing_threshold);
	runTest(SceneType::HomographyScene, Dataset::extremeview, ransac_confidence, draw_results, drawing_threshold);
	runTest(SceneType::HomographyScene, Dataset::homogr, ransac_confidence, draw_results, drawing_threshold);

	return 0;
} 

void runTest(SceneType scene_type_, 
	Dataset dataset_,
	const double ransac_confidence_,
	const bool draw_results_,
	const double drawing_threshold_)
{
	const std::string dataset_name = dataset2str(dataset_);
	const std::string problem_name = scene_type_ == SceneType::HomographyScene ?
		"Homography" : 
		"Fundamental matrix";	

	// Test scenes for homography estimation
	for (const auto& scene : getAvailableTestScenes(scene_type_, dataset_))
	{
		printf("--------------------------------------------------------------\n");
		printf("%s estimation on scene \"%s\" from dataset \"%s\".\n", 
			problem_name.c_str(), scene.c_str(), dataset_name.c_str());
		printf("--------------------------------------------------------------\n");

		if (scene_type_ == SceneType::HomographyScene)
		{
			printf("- Running OpenCV's RANSAC with threshold %f px\n", 3.0);
			opencvHomographyFitting(ransac_confidence_,
				3, // The maximum sigma value allowed in MAGSAC
				scene, // The scene type
				false); // A flag to draw and show the results

			printf("\n- Running MAGSAC with reasonably set maximum threshold (%f px)\n", 3.0);
			testHomographyFitting(ransac_confidence_,
				3, // The maximum sigma value allowed in MAGSAC
				scene, // The scene type
				draw_results_, // A flag to draw and show the results 
				drawing_threshold_); // The inlier threshold for visualization.

			printf("\n- Running MAGSAC with extreme maximum threshold (%f px)\n", 10.0);
			testHomographyFitting(ransac_confidence_,
				10, // The maximum sigma value allowed in MAGSAC
				scene, // The scene type
				draw_results_, // A flag to draw and show the results 
				drawing_threshold_); // The inlier threshold for visualization.
		} else
		{
			printf("- Running OpenCV's RANSAC with threshold %f px\n", 3.0);
			opencvFundamentalMatrixFitting(ransac_confidence_,
				3, // The maximum sigma value allowed in MAGSAC
				scene, // The scene type
				false); // A flag to draw and show the results

			printf("\n- Running MAGSAC with reasonably set maximum threshold (%f px)\n", 3.0);
			testFundamentalMatrixFitting(ransac_confidence_, // The required confidence in the results
				3, // The maximum sigma value allowed in MAGSAC
				scene, // The scene type
				draw_results_, // A flag to draw and show the results 
				drawing_threshold_); // The inlier threshold for visualization.


			printf("\n- Running MAGSAC with extreme maximum threshold (%f px)\n", 10.0);
			testFundamentalMatrixFitting(ransac_confidence_, // The required confidence in the results
				10, // The maximum sigma value allowed in MAGSAC
				scene, // The scene type
				draw_results_, // A flag to draw and show the results 
				drawing_threshold_); // The inlier threshold for visualization.
		}

		printf("Press a button to continue.\n\n");
		cv::waitKey(0);
	}
}

std::string dataset2str(Dataset dataset_)
{
	switch (dataset_)	
	{	
		case Dataset::homogr:
			return "homogr";
		case Dataset::extremeview:
			return "extremeview";
		case Dataset::kusvod2:
			return "kusvod2";
		case Dataset::adelaidermf:
			return "adelaidermf";
		case Dataset::multih:
			return "multih";
		default:
			return "unknown";
	}
}

std::vector<std::string> getAvailableTestScenes(SceneType scene_type_, 
	Dataset dataset_)
{
	switch (scene_type_)
	{
	case SceneType::HomographyScene: // Available test scenes for homography estimation
		switch (dataset_)	
		{	
			case Dataset::homogr:
				return { "LePoint1", "LePoint2", "LePoint3", // "homogr" dataset
					"graf", "ExtremeZoom", "city", 
					"CapitalRegion", "BruggeTower", "BruggeSquare", 
					"BostonLib", "boat", "adam", 
					"WhiteBoard", "Eiffel", "Brussels", 
					"Boston"};
			case Dataset::extremeview:
				return {"extremeview/adam", "extremeview/cafe", "extremeview/cat", // "EVD" (i.e. extremeview) dataset
					"extremeview/dum", "extremeview/face", "extremeview/fox", 
					"extremeview/girl", "extremeview/graf", "extremeview/grand", 
					"extremeview/index", "extremeview/mag", "extremeview/pkk", 
					"extremeview/shop", "extremeview/there", "extremeview/vin"};

			default:
				return std::vector<std::string>();
		}

	case SceneType::FundamentalMatrixScene:
		switch (dataset_)	
		{	
			case Dataset::kusvod2:
				return {"corr", "booksh", "box",
					"castle", "graff", "head",
					"kampa", "leafs", "plant",
					"rotunda", "shout", "valbonne",
					"wall", "wash", "zoom",
					"Kyoto"};
			case Dataset::adelaidermf:
				return {"barrsmith", "bonhall",
					"bonython", "elderhalla", "elderhallb",
					"hartley", "johnssonb", "ladysymon", 
					"library", "napiera", "napierb", 
					"nese", "oldclassicswing", "physics", 
					"sene", "unihouse", "unionhouse"};
			case Dataset::multih:
				return {"boxesandbooks", "glasscaseb", "stairs"};
			default:
				return std::vector<std::string>();
		}
	default:
		return std::vector<std::string>();
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

	// Initialize the sampler used for selecting minimal samples
	UniformSampler<cv::Mat> sampler(N);
	
	MAGSAC<cv::Mat, FundamentalMatrixEstimator, FundamentalMatrix> magsac;
	magsac.setSigmaMax(sigma_max_); // The maximum noise scale sigma allowed
	magsac.setCoreNumber(5); // The number of cores used to speed up sigma-consensus
	magsac.setPartitionNumber(5); // The number partitions used for speeding up sigma consensus. As the value grows, the algorithm become slower and, usually, more accurate.
	magsac.setIterationLimit(1e5); // Iteration limit to interrupt the cases when the algorithm run too long.

	int iteration_number = 0; // Number of iterations required

	std::chrono::time_point<std::chrono::system_clock> end,
		start = std::chrono::system_clock::now();
	magsac.run(points, // The data points
		ransac_confidence_, // The required confidence in the results
		estimator, // The used estimator
		sampler, // The sampler used for selecting minimal samples in each iteration
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
		drawMatches<double, int>(points, obtained_labeling, image1, image2, out_image);

		// Show the matches
		std::string window_name = "Visualization with threshold = " + std::to_string(drawing_threshold_) + " px; Maximum threshold is = " + std::to_string(sigma_max_);
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

	// Initialize the sampler used for selecting minimal samples
	UniformSampler<cv::Mat> sampler(N);

	MAGSAC<cv::Mat, RobustHomographyEstimator, Homography> magsac;
	magsac.setSigmaMax(sigma_max_); // The maximum noise scale sigma allowed
	magsac.setCoreNumber(5); // The number of cores used to speed up sigma-consensus
	magsac.setPartitionNumber(5); // The number partitions used for speeding up sigma consensus. As the value grows, the algorithm become slower and, usually, more accurate.
	magsac.setIterationLimit(1e5); // Iteration limit to interrupt the cases when the algorithm run too long.

	int iteration_number = 0; // Number of iterations required

	std::chrono::time_point<std::chrono::system_clock> end, 
		start = std::chrono::system_clock::now();
	magsac.run(points, // The data points
		ransac_confidence_, // The required confidence in the results
		estimator, // The used estimator
		sampler, // The sampler used for selecting minimal samples in each iteration
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
		drawMatches<double, int>(points, obtained_labeling, image1, image2, out_image);

		// Show the matches
		std::string window_name = "Visualization with threshold = " + std::to_string(drawing_threshold_) + " px; Maximum threshold is = " + std::to_string(sigma_max_);
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

void opencvHomographyFitting(
	double ransac_confidence_,
	double threshold_,
	std::string test_scene_,
	bool draw_results_)
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

	// Define location of sub matrices in data matrix
	cv::Rect roi1( 0, 0, 3, N );  
	cv::Rect roi2( 3, 0, 3, N );   

	std::vector<int> obtained_labeling(points.rows, 0);
	std::chrono::time_point<std::chrono::system_clock> end, 
		start = std::chrono::system_clock::now();
	cv::Mat homography = cv::findHomography(cv::Mat(points, roi1), 
		cv::Mat(points, roi2),
		CV_RANSAC,
		threshold_,
		obtained_labeling);
	end = std::chrono::system_clock::now();
	 
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::time_t end_time = std::chrono::system_clock::to_time_t(end);

	printf("Elapsed time: %f secs\n", elapsed_seconds.count());
	 
	// Compute the RMSE given the ground truth inliers
	double rmse = 0, error;
	size_t inlier_number = 0;
	for (const auto& inlier_idx : ground_truth_inliers)
	{
		error = estimator.error(points.row(inlier_idx), homography);
		rmse += error;
	}
	rmse = sqrt(rmse / static_cast<double>(I));
	printf("RMSE error: %f px\n", rmse);

	// Visualization part.
	// Inliers are selected using threshold and the estimated model. 
	// This part is not necessary and is only for visualization purposes. 
	if (draw_results_)
	{
		// Draw the matches to the images
		cv::Mat out_image;
		drawMatches<double, int>(points, obtained_labeling, image1, image2, out_image);

		// Show the matches
		std::string window_name = "OpenCV's RANSAC";
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

void opencvFundamentalMatrixFitting(
	double ransac_confidence_,
	double threshold_,
	std::string test_scene_,
	bool draw_results_)
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
		0.35);

	// Select the inliers from the labeling
	std::vector<int> ground_truth_inliers = getSubsetFromLabeling(ground_truth_labels, 1);
	const size_t I = static_cast<double>(ground_truth_inliers.size());

	printf("Estimated model = '%s'.\n", estimator.modelName().c_str());
	printf("Number of correspondences loaded = %d.\n", static_cast<int>(N));
	printf("Number of ground truth inliers = %d.\n", static_cast<int>(I));

	// Define location of sub matrices in data matrix
	cv::Rect roi1( 0, 0, 3, N );  
	cv::Rect roi2( 3, 0, 3, N );   

	std::vector<uchar> obtained_labeling(points.rows, 0);
	std::chrono::time_point<std::chrono::system_clock> end, 
		start = std::chrono::system_clock::now();
	cv::Mat fundamental_matrix = cv::findFundamentalMat(cv::Mat(points, roi1), 
		cv::Mat(points, roi2),
		CV_FM_RANSAC,
		threshold_,
		ransac_confidence_,
		obtained_labeling);
	end = std::chrono::system_clock::now();
	 
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::time_t end_time = std::chrono::system_clock::to_time_t(end);

	printf("Elapsed time: %f secs\n", elapsed_seconds.count());
	 
	// Compute the RMSE given the ground truth inliers
	double rmse = 0, error;
	size_t inlier_number = 0;
	for (const auto& inlier_idx : ground_truth_inliers)
	{
		error = estimator.error(points.row(inlier_idx), fundamental_matrix);
		rmse += error;
	}
	rmse = sqrt(rmse / static_cast<double>(I));
	printf("RMSE error: %f px\n", rmse);

	// Visualization part.
	// Inliers are selected using threshold and the estimated model. 
	// This part is not necessary and is only for visualization purposes. 
	if (draw_results_)
	{
		// Draw the matches to the images
		cv::Mat out_image;
		drawMatches<double, uchar>(points, obtained_labeling, image1, image2, out_image);

		// Show the matches
		std::string window_name = "OpenCV's RANSAC";
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
