#include <string>
#include <vector>
#include <opencv2/core/core.hpp>

int adaptiveInlierSelection_(
    const std::vector<double>& srcPts_,
    const std::vector<double>& dstPts_,
    const std::vector<double>& model_,
    std::vector<bool>& inliers_,
    double& bestThreshold_,
    int problemType_,
    double maximumThreshold_,
    int minimumInlierNumber_);

// The common parameters of the functions below are those of findLine2D_ (further down).
// Each returns the number of inliers, or 0 if no model was found; matrices are returned row-major.

// Fits a camera pose [R | t] (3x4) to rows (u, v, x, y, z): a point (u, v) in normalized
// image coordinates and the 3D point (x, y, z) it sees. The threshold is in normalized coordinates.
int findPnP_(std::vector<double>& correspondences,
    std::vector<bool>& inliers,
    std::vector<double>& pose,
    std::vector<double>& inlier_probabilities,
    int sampler_id,
    bool use_magsac_plus_plus,
    double sigma_max,
    double conf,
    int min_iters,
    int max_iters,
    int partition_num);

// Fits a homography H and the radial distortion (lambda_1, lambda_2) of the division model
// of each image to rows (x1, y1, x2, y2), with coordinates relative to the distortion centres.
int findRadialHomography_(std::vector<double>& correspondences,
    std::vector<bool>& inliers,
    std::vector<double>& H,
    std::vector<double>& distortion,
    std::vector<double>& inlier_probabilities,
    int sampler_id,
    bool use_magsac_plus_plus,
    double sigma_max,
    double conf,
    int min_iters,
    int max_iters,
    int partition_num);

// Fits a homography to affine correspondences, rows (x1, y1, x2, y2, a11, a12, a21, a22),
// where A = [a11 a12; a21 a22] maps a neighbourhood of (x1, y1) to one of (x2, y2).
int findHomographyAffine_(std::vector<double>& correspondences,
    std::vector<bool>& inliers,
    std::vector<double>& H,
    std::vector<double>& inlier_probabilities,
    int sampler_id,
    bool use_magsac_plus_plus,
    double sigma_max,
    double conf,
    int min_iters,
    int max_iters,
    int partition_num);

// Fits an essential matrix to rows (x1, y1, x2, y2) in pixels, assuming a planar motion
// (rotation about the vertical axis, translation in the horizontal plane).
int findEssentialMatrixPlanar_(std::vector<double>& correspondences,
    std::vector<bool>& inliers,
    std::vector<double>& E,
    std::vector<double>& src_K,
    std::vector<double>& dst_K,
    std::vector<double>& inlier_probabilities,
    int sampler_id,
    bool use_magsac_plus_plus,
    double sigma_max,
    double conf,
    int min_iters,
    int max_iters,
    int partition_num);

// Fits an essential matrix to rows (x1, y1, x2, y2) in pixels, given the rotations that
// align each camera's vertical axis with gravity.
int findEssentialMatrixGravity_(std::vector<double>& correspondences,
    std::vector<bool>& inliers,
    std::vector<double>& E,
    std::vector<double>& src_K,
    std::vector<double>& dst_K,
    std::vector<double>& gravity_src,
    std::vector<double>& gravity_dst,
    std::vector<double>& inlier_probabilities,
    int sampler_id,
    bool use_magsac_plus_plus,
    double sigma_max,
    double conf,
    int min_iters,
    int max_iters,
    int partition_num);

int findLine2D_(std::vector<double>& points,
    std::vector<bool>& inliers,
    std::vector<double>& line,
    std::vector<double>& inlier_probabilities,
    double imageWidth,
    double imageHeight,
    int sampler_id,
    bool use_magsac_plus_plus,
    double sigma_max,
    double conf,
    int min_iters,
    int max_iters,
    int partition_num);

int findFundamentalMatrix_(
    std::vector<double>& correspondences,
    std::vector<bool>& inliers,
    std::vector<double>& F,
    std::vector<double>& inlier_probabilities,
    double sourceImageWidth,
    double sourceImageHeight,
    double destinationImageWidth,
    double destinationImageHeight,
    int sampler_id,
    bool use_magsac_plus_plus,
    double sigma_max,
    double conf,
    int min_iters,
    int max_iters,
    int partition_num);

int findHomography_(std::vector<double>& correspondences,
                    std::vector<bool>& inliers,
                    std::vector<double>& H,
                    std::vector<double>& inlier_probabilities,
                    double sourceImageWidth,
                    double sourceImageHeight,
                    double destinationImageWidth,
                    double destinationImageHeight,
                    int sampler_id,
					bool use_magsac_plus_plus,
                    double sigma_max,
                    double conf,
                    int min_iters,
                    int max_iters,
                    int partition_num);

int findEssentialMatrix_(std::vector<double>& correspondences,
    std::vector<bool>& inliers,
    std::vector<double>& E,
    std::vector<double>& src_K,
    std::vector<double>& dst_K,
    std::vector<double>& inlier_probabilities,
    double sourceImageWidth,
    double sourceImageHeight,
    double destinationImageWidth,
    double destinationImageHeight,
    int sampler_id,
    bool use_magsac_plus_plus,
    double sigma_max,
    double conf,
    int min_iters,
    int max_iters,
    int partition_num);

    
int findRigidTransformation_(
    std::vector<double>& correspondences,
    std::vector<bool>& inliers,
    std::vector<double>& F,
    std::vector<double>& inlier_probabilities,
    int sampler_id,
    bool use_magsac_plus_plus,
    double sigma_max,
    double conf,
    //double neighborhood_size,
    int min_iters,
    int max_iters,
    int partition_num);