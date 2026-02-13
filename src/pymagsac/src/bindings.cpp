#include <stdexcept>
#include "array_support.h"
#include "magsac_python.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;
using namespace nb::literals;

nb::tuple adaptiveInlierSelection(
    NDArray<double, 2> x1y1_,
    NDArray<double, 2> x2y2_,
    NDArray<double, 2> modelParameters_,
    double maximumThreshold_,
    int problemType_,
    int minimumInlierNumber_)
{
    if (problemType_ < 0 || problemType_ > 2)
        throw std::invalid_argument("Variable 'problemType' should be in interval [0,2]");

    size_t NUM_TENTS = x1y1_.shape(0);
    size_t DIM = x1y1_.shape(1);
    if (DIM != 2) {
        throw std::invalid_argument("x1y1 should be an array with dims [n,2]");
    }

    size_t NUM_TENTSa = x2y2_.shape(0);
    size_t DIMa = x2y2_.shape(1);
    if (DIMa != 2) {
        throw std::invalid_argument("x2y2 should be an array with dims [n,2]");
    }

    if (NUM_TENTSa != NUM_TENTS) {
        throw std::invalid_argument("x1y1 and x2y2 should be the same size");
    }

    size_t DIMModelX = modelParameters_.shape(0);
    size_t DIMModelY = modelParameters_.shape(1);
    if (DIMModelX != 3 || DIMModelY != 3)
        throw std::invalid_argument("The model should be a 3*3 matrix.");

    // convert inputs to vectors
    double* ptr1 = x1y1_.data();
    std::vector<double> x1y1(ptr1, ptr1 + x1y1_.size());

    double* ptr2 = x2y2_.data();
    std::vector<double> x2y2(ptr2, ptr2 + x2y2_.size());

    double* ptrModel = modelParameters_.data();
    std::vector<double> modelParameters(ptrModel, ptrModel + modelParameters_.size());

    std::vector<bool> inliers(NUM_TENTS);
    double bestThreshold;

    int inlierNumber = adaptiveInlierSelection_(
        x1y1,
        x2y2,
        modelParameters,
        inliers,
        bestThreshold,
        problemType_,
        maximumThreshold_,
        minimumInlierNumber_);

    NDArray<bool, 1> inliers_ = MakeNDArray<bool, 1>({NUM_TENTS});

    // nb::buffer_info bufInliers = inliers_.data();
    bool* ptrInliers = inliers_.data();
    for (size_t i = 0; i < NUM_TENTS; i++)
        ptrInliers[i] = inliers[i];

    return nb::make_tuple(inliers_, inlierNumber, bestThreshold);

}

nb::tuple findRigidTransformation(
	NDArray<double, 2> correspondences_,
	NDArray<double, 2> probabilities_,
	int sampler,
    bool use_magsac_plus_plus,
    double sigma_th,
    double conf,
    int min_iters,
    int max_iters,
    int partition_num)
{
	size_t NUM_TENTS = correspondences_.shape(0);
	size_t DIM = correspondences_.shape(1);

	if (DIM != 6) {
		throw std::invalid_argument("correspondences should be an array with dims [n,6], n>=3");
	}
    if (NUM_TENTS < 3) {
        throw std::invalid_argument("correspondences should be an array with dims [n,6], n>=3");
    }

    double* ptr1 = correspondences_.data();
    std::vector<double> correspondences;
    correspondences.assign(ptr1, ptr1 + correspondences_.size());

    std::vector<double> T(16);
    std::vector<bool> inliers(NUM_TENTS);

    std::vector<double> probabilities;
    if (sampler == 3 || sampler == 4)
    {
        double* ptr_prob = probabilities_.data();
        probabilities.assign(ptr_prob, ptr_prob + probabilities_.size());
    }

    int num_inl = findRigidTransformation_(
        correspondences,
        inliers,
        T,
        probabilities,
        sampler,
        use_magsac_plus_plus,
        sigma_th,
        conf,
        min_iters,
        max_iters,
        partition_num);

    NDArray<bool, 1> inliers_ = MakeNDArray<bool, 1>({NUM_TENTS});
    bool* ptr3 = inliers_.data();
    for (size_t i = 0; i < NUM_TENTS; i++)
        ptr3[i] = inliers[i];
    if (num_inl == 0) {
        return nb::make_tuple(nb::none(), inliers_);
    }

    NDArray<double, 2> T_ = MakeNDArray<double, 2>({4, 4});
    double* ptr2 = T_.data();
    for (size_t i = 0; i < 16; i++)
        ptr2[i] = T[i];
    return nb::make_tuple(T_, inliers_);
}

nb::tuple findFundamentalMatrix(
    NDArray<double, 2> correspondences_,
    double w1,
    double h1,
    double w2,
    double h2,
    NDArray<double, 2> probabilities_,
	int sampler,
    bool use_magsac_plus_plus,
    double sigma_th,
    double conf,
    int min_iters,
    int max_iters,
    int partition_num)
{
	size_t NUM_TENTS = correspondences_.shape(0);
	size_t DIM = correspondences_.shape(1);
	if (DIM != 4) {
		throw std::invalid_argument("x1y1 should be an array with dims [n,4], n>=7");
	}
    if (NUM_TENTS < 7) {
        throw std::invalid_argument("x1y1 should be an array with dims [n,4], n>=7");
    }

    double* ptr1 = correspondences_.data();
    std::vector<double> correspondences;
    correspondences.assign(ptr1, ptr1 + correspondences_.size());

    std::vector<double> F(9);
    std::vector<bool> inliers(NUM_TENTS);

    std::vector<double> probabilities;
    if (sampler == 3 || sampler == 4)
    {
        double* ptr_prob = probabilities_.data();
        probabilities.assign(ptr_prob, ptr_prob + probabilities_.size());
    }

    int num_inl = findFundamentalMatrix_(
        correspondences,
        inliers,
        F,
        probabilities,
        w1,
        h1,
        w2,
        h2,
        sampler,
        use_magsac_plus_plus,
        sigma_th,
        conf,
        min_iters,
        max_iters,
        partition_num);

    NDArray<bool, 1> inliers_ = MakeNDArray<bool, 1>({NUM_TENTS});
    bool* ptr3 = inliers_.data();
    for (size_t i = 0; i < NUM_TENTS; i++)
        ptr3[i] = inliers[i];
    if (num_inl == 0) {
        return nb::make_tuple(nb::none(), inliers_);
    }

    NDArray<double, 2> F_ = MakeNDArray<double, 2>({3, 3});
    double* ptr2 = F_.data();
    for (size_t i = 0; i < 9; i++)
        ptr2[i] = F[i];
    return nb::make_tuple(F_, inliers_);
}

nb::tuple findEssentialMatrix(
	NDArray<double, 2> correspondences_,
    NDArray<double, 2>  K1_,
    NDArray<double, 2>  K2_,
    double w1,
    double h1,
    double w2,
    double h2,
	NDArray<double, 2>  probabilities_,
	int sampler,
    bool use_magsac_plus_plus,
    double sigma_th,
    double conf,
    int min_iters,
    int max_iters,
    int partition_num)
{
	size_t NUM_TENTS = correspondences_.shape(0);
	size_t DIM = correspondences_.shape(1);
	if (DIM != 4) {
		throw std::invalid_argument("x1y1 should be an array with dims [n,4], n>=5");
	}
    if (NUM_TENTS < 5) {
        throw std::invalid_argument("x1y1 should be an array with dims [n,4], n>=5");
    }

    double* ptr1 = correspondences_.data();
    std::vector<double> correspondences;
    correspondences.assign(ptr1, ptr1 + correspondences_.size());

    size_t three_a = K1_.shape(0);
    size_t three_b = K1_.shape(1);
    if ((three_a != 3) || (three_b != 3)) {
        throw std::invalid_argument("K1 shape should be [3x3]");
    }
    double* ptr1_k = K1_.data();
    std::vector<double> K1;
    K1.assign(ptr1_k, ptr1_k + K1_.size());

    three_a = K2_.shape(0);
    three_b = K2_.shape(1);
    if ((three_a != 3) || (three_b != 3)) {
        throw std::invalid_argument("K2 shape should be [3x3]");
    }
    double* ptr2_k = K2_.data();
    std::vector<double> K2;
    K2.assign(ptr2_k, ptr2_k + K2_.size());

    std::vector<double> E(9);
    std::vector<bool> inliers(NUM_TENTS);

    std::vector<double> probabilities;
    if (sampler == 3 || sampler == 4)
    {
        double* ptr_prob = probabilities_.data();
        probabilities.assign(ptr_prob, ptr_prob + probabilities_.size());
    }

    int num_inl = findEssentialMatrix_(
        correspondences,
        inliers,
        E,
        K1,
        K2,
        probabilities,
        w1,
        h1,
        w2,
        h2,
        sampler,
        use_magsac_plus_plus,
        sigma_th,
        conf,
        min_iters,
        max_iters,
        partition_num);

    NDArray<bool, 1> inliers_ = MakeNDArray<bool, 1>({NUM_TENTS});
    bool* ptr3 = inliers_.data();
    for (size_t i = 0; i < NUM_TENTS; i++)
        ptr3[i] = inliers[i];

    if (num_inl == 0) {
        return nb::make_tuple(nb::none(), inliers_);
    }

    NDArray<double, 2> E_ = MakeNDArray<double, 2>({3, 3});
    double* ptr2 = E_.data();
    for (size_t i = 0; i < 9; i++)
        ptr2[i] = E[i];
    return nb::make_tuple(E_, inliers_);
}

nb::tuple findLine2D(
    NDArray<double, 2> points_,
    double w1,
    double h1,
    NDArray<double, 2> probabilities_,
    int sampler,
    bool use_magsac_plus_plus,
    double sigma_th,
    double conf,
    int min_iters,
    int max_iters,
    int partition_num)
{
	size_t NUM_TENTS = points_.shape(0);
	size_t DIM = points_.shape(1);

	if (DIM != 2)
		throw std::invalid_argument("The points should be an array with dims [n,2], n>=2");
    if (NUM_TENTS < 2)
        throw std::invalid_argument("The points should be an array with dims [n,2], n>=2");

    double* ptr1 = points_.data();
    std::vector<double> points;
    points.assign(ptr1, ptr1 + points_.size());

    std::vector<double> line2d(3);
    std::vector<bool> inliers(NUM_TENTS);

    std::vector<double> probabilities;
    if (sampler == 3 || sampler == 4)
    {
        double* ptr_prob = probabilities_.data();
        probabilities.assign(ptr_prob, ptr_prob + probabilities_.size());
    }

    int num_inl = findLine2D_(
        points,
        inliers,
        line2d,
        probabilities,
        w1,
        h1,
        sampler,
        use_magsac_plus_plus,
        sigma_th,
        conf,
        min_iters,
        max_iters,
        partition_num);

    NDArray<bool, 1> inliers_ = MakeNDArray<bool, 1>({NUM_TENTS});    
    bool *ptr3 = inliers_.data();
    for (size_t i = 0; i < NUM_TENTS; i++)
        ptr3[i] = inliers[i];

    if (num_inl == 0){
        return nb::make_tuple(nb::none(),inliers_);
    }
    NDArray<double, 1> line_ = MakeNDArray<double, 1>({3});        
    double *ptr2 = line_.data();
    for (size_t i = 0; i < 3; i++)
        ptr2[i] = line2d[i];

    return nb::make_tuple(line_, inliers_);
}

nb::tuple findHomography(
    NDArray<double, 2> correspondences_,
    double w1,
    double h1,
    double w2,
    double h2,
    NDArray<double, 2> probabilities_,
    int sampler,
    bool use_magsac_plus_plus,
    double sigma_th,
    double conf,
    int min_iters,
    int max_iters,
    int partition_num)
{
	size_t NUM_TENTS = correspondences_.shape(0);
	size_t DIM = correspondences_.shape(1);
	if (DIM != 4) {
		throw std::invalid_argument("x1y1 should be an array with dims [n,4], n>=4");
	}
    if (NUM_TENTS < 4) {
        throw std::invalid_argument("x1y1 should be an array with dims [n,4], n>=4");
    }

    double* ptr1 = correspondences_.data();
    std::vector<double> correspondences;
    correspondences.assign(ptr1, ptr1 + correspondences_.size());

    std::vector<double> H(9);
    std::vector<bool> inliers(NUM_TENTS);

    std::vector<double> probabilities;
    if (sampler == 3 || sampler == 4)
    {
        double* ptr_prob = probabilities_.data();
        probabilities.assign(ptr_prob, ptr_prob + probabilities_.size());
    }

    int num_inl = findHomography_(
                    correspondences,
                    inliers,
                    H,
                    probabilities,
                    w1,
                    h1,
                    w2,
                    h2,
                    sampler,
					use_magsac_plus_plus,
                    sigma_th,
                    conf,
                    min_iters,
                    max_iters,
                    partition_num);

    NDArray<bool, 1> inliers_ = MakeNDArray<bool, 1>({NUM_TENTS});
    bool *ptr3 = inliers_.data();
    for (size_t i = 0; i < NUM_TENTS; i++)
        ptr3[i] = inliers[i];

    if (num_inl  == 0){
        return nb::make_tuple(nb::none(),inliers_);
    }
    NDArray<double, 2> H_ = MakeNDArray<double, 2>({3, 3});
    double *ptr2 = H_.data();
    for (size_t i = 0; i < 9; i++)
        ptr2[i] = H[i];

    return nb::make_tuple(H_,inliers_);
}

NB_MODULE(pymagsac, m) {
    m.doc() = R"doc(
        Python module
        -----------------------
        .. currentmodule:: pymagsac
        .. autosummary::
           :toctree: _generate

           findEssentialMatrix,
           findFundamentalMatrix,
           findHomography,
           adaptiveInlierSelection

    )doc";

    m.def("adaptiveInlierSelection", &adaptiveInlierSelection, R"doc(some doc)doc",
        nb::arg("x1y1"),
        nb::arg("x2y2"),
        nb::arg("modelParameters"),
        nb::arg("maximumThreshold"),
        nb::arg("problemType"),
        nb::arg("minimumInlierNumber") = 20);

    m.def("findEssentialMatrix", &findEssentialMatrix, R"doc(some doc)doc",
        nb::arg("correspondences"),
        nb::arg("K1"),
        nb::arg("K2"),
        nb::arg("w1"),
        nb::arg("h1"),
        nb::arg("w2"),
        nb::arg("h2"),
        nb::arg("probabilities"),
		nb::arg("sampler") = 4,
        nb::arg("use_magsac_plus_plus") = true,
        nb::arg("sigma_th") = 1.0,
        nb::arg("conf") = 0.99,
        nb::arg("min_iters") = 50,
        nb::arg("max_iters") = 1000,
        nb::arg("partition_num") = 5);

    m.def("findFundamentalMatrix", &findFundamentalMatrix, R"doc(some doc)doc",
        nb::arg("correspondences"),
        nb::arg("w1"),
        nb::arg("h1"),
        nb::arg("w2"),
        nb::arg("h2"),
        nb::arg("probabilities"),
		nb::arg("sampler") = 4,
        nb::arg("use_magsac_plus_plus") = true,
        nb::arg("sigma_th") = 1.0,
        nb::arg("conf") = 0.99,
        nb::arg("min_iters") = 50,
        nb::arg("max_iters") = 1000,
        nb::arg("partition_num") = 5);

    m.def("findRigidTransformation", &findRigidTransformation, R"doc(some doc)doc",
        nb::arg("correspondences"),
        nb::arg("probabilities")= MakeNDArray<double, 2>({0, 0}),
		nb::arg("sampler") = 4,
        nb::arg("use_magsac_plus_plus") = true,
        nb::arg("sigma_th") = 1.0,
        nb::arg("conf") = 0.99,
        nb::arg("min_iters") = 50,
        nb::arg("max_iters") = 1000,
        nb::arg("partition_num") = 5);

  m.def("findHomography", &findHomography, R"doc(some doc)doc",
        nb::arg("correspondences"),
        nb::arg("w1"),
        nb::arg("h1"),
        nb::arg("w2"),
        nb::arg("h2"),
        nb::arg("probabilities"),
		nb::arg("sampler") = 4,
        nb::arg("use_magsac_plus_plus") = true,
        nb::arg("sigma_th") = 1.0,
        nb::arg("conf") = 0.99,
        nb::arg("min_iters") = 50,
        nb::arg("max_iters") = 1000,
        nb::arg("partition_num") = 5);

  m.def("findLine2D", &findLine2D, R"doc(some doc)doc",
        nb::arg("points"),
        nb::arg("w1"),
        nb::arg("h1"),
        nb::arg("probabilities"),
		nb::arg("sampler") = 0,
        nb::arg("use_magsac_plus_plus") = true,
        nb::arg("sigma_th") = 1.0,
        nb::arg("conf") = 0.99,
        nb::arg("min_iters") = 50,
        nb::arg("max_iters") = 1000,
        nb::arg("partition_num") = 5);

}
