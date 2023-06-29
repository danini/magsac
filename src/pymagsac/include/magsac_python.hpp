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