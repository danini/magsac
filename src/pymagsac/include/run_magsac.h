#pragma once

#include <cstdio>
#include <memory>
#include <vector>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include "magsac.h"
#include "model.h"
#include "samplers/uniform_sampler.h"
#include "samplers/prosac_sampler.h"
#include "samplers/importance_sampler.h"
#include "samplers/adaptive_reordering_sampler.h"

// Used by the Python functions of graph-cut-ransac's estimators (see estimator_wrapper.h).
// Runs MAGSAC (or MAGSAC++) on the data points and marks the points closer than the
// threshold to the estimated model as inliers. Returns the number of inliers, or 0 if
// no model was found. Samplers: 0 (uniform), 1 (PROSAC, the points must be sorted by
// quality), 3 (NG-RANSAC) and 4 (AR-Sampler), the last two using the inlier probabilities,
// one per point (checked by the caller).
template <class Estimator>
int runMagsac(const cv::Mat& points,
    Estimator& estimator,
    std::vector<double>& inlier_probabilities,
    int sampler_id,
    bool use_magsac_plus_plus,
    double threshold,
    double reference_threshold,
    double conf,
    int min_iters,
    int max_iters,
    int partition_num,
    gcransac::Model& model,
    std::vector<bool>& inliers)
{
    inliers.assign(points.rows, false);

    std::unique_ptr<gcransac::sampler::Sampler<cv::Mat, size_t>> sampler;
    if (sampler_id == 0)
        sampler.reset(new gcransac::sampler::UniformSampler(&points));
    else if (sampler_id == 1)
        sampler.reset(new gcransac::sampler::ProsacSampler(&points, estimator.sampleSize()));
    else if (sampler_id == 3)
        sampler.reset(new gcransac::sampler::ImportanceSampler(&points, inlier_probabilities, estimator.sampleSize()));
    else if (sampler_id == 4)
    {
        double max_prob = 0;
        for (const auto& prob : inlier_probabilities)
            max_prob = MAX(max_prob, prob);
        for (auto& prob : inlier_probabilities)
            prob /= max_prob;
        sampler.reset(new gcransac::sampler::AdaptiveReorderingSampler(&points, inlier_probabilities, estimator.sampleSize(), 0.1));
    }
    else
    {
        fprintf(stderr, "Unknown sampler identifier: %d. The accepted samplers are 0 (uniform sampling), 1 (PROSAC sampling), 3 (NG-RANSAC sampler), 4 (AR-Sampler)\n",
            sampler_id);
        return 0;
    }

    MAGSAC<cv::Mat, Estimator> magsac(use_magsac_plus_plus ?
        MAGSAC<cv::Mat, Estimator>::MAGSAC_PLUS_PLUS :
        MAGSAC<cv::Mat, Estimator>::MAGSAC_ORIGINAL);
    magsac.setMaximumThreshold(threshold);
    magsac.setReferenceThreshold(reference_threshold);
    magsac.setCoreNumber(1);
    magsac.setPartitionNumber(partition_num);
    magsac.setIterationLimit(max_iters);
    magsac.setMinimumIterationNumber(min_iters);

    ModelScore score;
    int iterations = max_iters;
    if (!magsac.run(points, conf, estimator, *sampler, model, iterations, score))
        return 0;

    int num_inliers = 0;
    for (int i = 0; i < points.rows; ++i)
        if (estimator.residual(points.row(i), model.descriptor) <= threshold)
        {
            inliers[i] = true;
            ++num_inliers;
        }
    return num_inliers;
}

// Copies a matrix to a row-major vector.
inline std::vector<double> toRowMajor(const Eigen::MatrixXd& matrix)
{
    std::vector<double> values;
    for (Eigen::Index row = 0; row < matrix.rows(); ++row)
        for (Eigen::Index col = 0; col < matrix.cols(); ++col)
            values.push_back(matrix(row, col));
    return values;
}
