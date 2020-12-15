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

int findFundamentalMatrix_(std::vector<double>& srcPts,
                           std::vector<double>& dstPts,
                           std::vector<bool>& inliers,
                           std::vector<double>&  F,
						   bool use_magsac_plus_plus = true,
                           double sigma_th = 3.0,
                           double conf = 0.99,
                           int max_iters = 10000,
                           int partition_num = 5);


                
int findHomography_(std::vector<double>& srcPts,
                    std::vector<double>& dstPts,
                    std::vector<bool>& inliers,
                    std::vector<double>& H,
					bool use_magsac_plus_plus = true,
                    double sigma_th = 3.0,
                    double conf = 0.99,
                    int max_iters = 10000,
                    int partition_num = 5);

int findEssentialMatrix_(std::vector<double>& srcPts,
    std::vector<double>& dstPts,
    std::vector<bool>& inliers,
    std::vector<double>& E,
    std::vector<double>& intrinsics_src,
    std::vector<double>& intrinsics_dst,
    bool use_magsac_plus_plus,
    double sigma_max,
    double conf,
    int max_iters,
    int partition_num);