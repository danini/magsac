#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include "magsac_python.hpp"
#include "bindings_utils.hpp"


namespace py = pybind11;

py::tuple adaptiveInlierSelection(
    py::array_t<double>  x1y1_,
    py::array_t<double>  x2y2_,
    py::array_t<double>  modelParameters_,
    double maximumThreshold_,
    int problemType_,
    int minimumInlierNumber_) 
{
    if (problemType_ < 0 || problemType_ > 2)
        throw std::invalid_argument("Variable 'problemType' should be in interval [0,2]");

    py::buffer_info buf1 = x1y1_.request();
    size_t NUM_TENTS = buf1.shape[0];
    size_t DIM = buf1.shape[1];

    if (DIM != 2) {
        throw std::invalid_argument("x1y1 should be an array with dims [n,2]");
    }

    py::buffer_info buf1a = x2y2_.request();
    size_t NUM_TENTSa = buf1a.shape[0];
    size_t DIMa = buf1a.shape[1];

    if (DIMa != 2) {
        throw std::invalid_argument("x2y2 should be an array with dims [n,2]");
    }

    if (NUM_TENTSa != NUM_TENTS) {
        throw std::invalid_argument("x1y1 and x2y2 should be the same size");
    }

    py::buffer_info bufModel = modelParameters_.request();
    size_t DIMModelX = bufModel.shape[0];
    size_t DIMModelY = bufModel.shape[1];

    if (DIMModelX != 3 || DIMModelY != 3)
        throw std::invalid_argument("The model should be a 3*3 matrix.");

    double* ptr1 = (double*)buf1.ptr;
    std::vector<double> x1y1;
    x1y1.assign(ptr1, ptr1 + buf1.size);

    double* ptr1a = (double*)buf1a.ptr;
    std::vector<double> x2y2;
    x2y2.assign(ptr1a, ptr1a + buf1a.size);

    double* ptrModel = (double*)bufModel.ptr;
    std::vector<double> modelParameters;
    modelParameters.assign(ptrModel, ptrModel + bufModel.size);

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

    py::array_t<bool> inliers_ = py::array_t<bool>(NUM_TENTS);
    py::buffer_info bufInliers = inliers_.request();
    bool* ptrInliers = (bool*)bufInliers.ptr;
    for (size_t i = 0; i < NUM_TENTS; i++)
        ptrInliers[i] = inliers[i];

    return py::make_tuple(inliers_, inlierNumber, bestThreshold);

}

py::tuple findRigidTransformation(
	py::array_t<double>  correspondences_,
	py::array_t<double>  probabilities_,
	int sampler,
    bool use_magsac_plus_plus,
    double sigma_th,
    double conf,
    int min_iters,
    int max_iters,
    int partition_num)
{
	py::buffer_info buf1 = correspondences_.request();
	size_t NUM_TENTS = buf1.shape[0];
	size_t DIM = buf1.shape[1];

	if (DIM != 6) {
		throw std::invalid_argument("correspondences should be an array with dims [n,6], n>=3");
	}
    if (NUM_TENTS < 3) {
        throw std::invalid_argument("correspondences should be an array with dims [n,6], n>=3");
    }

    double* ptr1 = (double*)buf1.ptr;
    std::vector<double> correspondences;
    correspondences.assign(ptr1, ptr1 + buf1.size);

    std::vector<double> T(16);
    std::vector<bool> inliers(NUM_TENTS);

    std::vector<double> probabilities;
    if (sampler == 3 || sampler == 4)
    {
        py::buffer_info buf_prob = probabilities_.request();
        double* ptr_prob = (double*)buf_prob.ptr;
        probabilities.assign(ptr_prob, ptr_prob + buf_prob.size);        
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

    py::array_t<bool> inliers_ = py::array_t<bool>(NUM_TENTS);
    py::buffer_info buf3 = inliers_.request();
    bool* ptr3 = (bool*)buf3.ptr;
    for (size_t i = 0; i < NUM_TENTS; i++)
        ptr3[i] = inliers[i];
    if (num_inl == 0) {
        return py::make_tuple(pybind11::cast<pybind11::none>(Py_None), inliers_);
    }
    py::array_t<double> T_ = py::array_t<double>({ 4,4 });
    py::buffer_info buf2 = T_.request();
    double* ptr2 = (double*)buf2.ptr;
    for (size_t i = 0; i < 16; i++)
        ptr2[i] = T[i];
    return py::make_tuple(T_, inliers_);
}

py::tuple findFundamentalMatrix(
	py::array_t<double>  correspondences_,
    double w1, 
    double h1,
    double w2,
    double h2,
	py::array_t<double>  probabilities_,
	int sampler,
    bool use_magsac_plus_plus,
    double sigma_th,
    double conf,
    int min_iters,
    int max_iters,
    int partition_num)
{
	py::buffer_info buf1 = correspondences_.request();
	size_t NUM_TENTS = buf1.shape[0];
	size_t DIM = buf1.shape[1];

	if (DIM != 4) {
		throw std::invalid_argument("x1y1 should be an array with dims [n,4], n>=7");
	}
    if (NUM_TENTS < 7) {
        throw std::invalid_argument("x1y1 should be an array with dims [n,4], n>=7");
    }

    double* ptr1 = (double*)buf1.ptr;
    std::vector<double> correspondences;
    correspondences.assign(ptr1, ptr1 + buf1.size);

    std::vector<double> F(9);
    std::vector<bool> inliers(NUM_TENTS);

    std::vector<double> probabilities;
    if (sampler == 3 || sampler == 4)
    {
        py::buffer_info buf_prob = probabilities_.request();
        double* ptr_prob = (double*)buf_prob.ptr;
        probabilities.assign(ptr_prob, ptr_prob + buf_prob.size);        
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

    py::array_t<bool> inliers_ = py::array_t<bool>(NUM_TENTS);
    py::buffer_info buf3 = inliers_.request();
    bool* ptr3 = (bool*)buf3.ptr;
    for (size_t i = 0; i < NUM_TENTS; i++)
        ptr3[i] = inliers[i];
    if (num_inl == 0) {
        return py::make_tuple(pybind11::cast<pybind11::none>(Py_None), inliers_);
    }
    py::array_t<double> F_ = py::array_t<double>({ 3,3 });
    py::buffer_info buf2 = F_.request();
    double* ptr2 = (double*)buf2.ptr;
    for (size_t i = 0; i < 9; i++)
        ptr2[i] = F[i];
    return py::make_tuple(F_, inliers_);
}

py::tuple findEssentialMatrix(
	py::array_t<double>  correspondences_,
    py::array_t<double>  K1_,
    py::array_t<double>  K2_,
    double w1, 
    double h1,
    double w2,
    double h2,
	py::array_t<double>  probabilities_,
	int sampler,
    bool use_magsac_plus_plus,
    double sigma_th,
    double conf,
    int min_iters,
    int max_iters,
    int partition_num) 
{
	py::buffer_info buf1 = correspondences_.request();
	size_t NUM_TENTS = buf1.shape[0];
	size_t DIM = buf1.shape[1];

	if (DIM != 4) {
		throw std::invalid_argument("x1y1 should be an array with dims [n,4], n>=5");
	}
    if (NUM_TENTS < 5) {
        throw std::invalid_argument("x1y1 should be an array with dims [n,4], n>=5");
    }

    double* ptr1 = (double*)buf1.ptr;
    std::vector<double> correspondences;
    correspondences.assign(ptr1, ptr1 + buf1.size);

    py::buffer_info K1_buf = K1_.request();
    size_t three_a = K1_buf.shape[0];
    size_t three_b = K1_buf.shape[1];

    if ((three_a != 3) || (three_b != 3)) {
        throw std::invalid_argument("K1 shape should be [3x3]");
    }
    double* ptr1_k = (double*)K1_buf.ptr;
    std::vector<double> K1;
    K1.assign(ptr1_k, ptr1_k + K1_buf.size);

    py::buffer_info K2_buf = K2_.request();
    three_a = K2_buf.shape[0];
    three_b = K2_buf.shape[1];

    if ((three_a != 3) || (three_b != 3)) {
        throw std::invalid_argument("K2 shape should be [3x3]");
    }
    double* ptr2_k = (double*)K2_buf.ptr;
    std::vector<double> K2;
    K2.assign(ptr2_k, ptr2_k + K2_buf.size);

    std::vector<double> E(9);
    std::vector<bool> inliers(NUM_TENTS);

    std::vector<double> probabilities;
    if (sampler == 3 || sampler == 4)
    {
        py::buffer_info buf_prob = probabilities_.request();
        double* ptr_prob = (double*)buf_prob.ptr;
        probabilities.assign(ptr_prob, ptr_prob + buf_prob.size);        
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

    py::array_t<bool> inliers_ = py::array_t<bool>(NUM_TENTS);
    py::buffer_info buf3 = inliers_.request();
    bool* ptr3 = (bool*)buf3.ptr;
    for (size_t i = 0; i < NUM_TENTS; i++)
        ptr3[i] = inliers[i];

    if (num_inl == 0) {
        return py::make_tuple(pybind11::cast<pybind11::none>(Py_None), inliers_);
    }

    py::array_t<double> E_ = py::array_t<double>({ 3,3 });
    py::buffer_info buf2 = E_.request();
    double* ptr2 = (double*)buf2.ptr;
    for (size_t i = 0; i < 9; i++)
        ptr2[i] = E[i];
    return py::make_tuple(E_, inliers_);
}

py::tuple findLine2D(
    py::array_t<double> points_,
    double w1, 
    double h1,
    py::array_t<double> probabilities_,
    int sampler,
    bool use_magsac_plus_plus,
    double sigma_th,
    double conf,
    int min_iters,
    int max_iters,
    int partition_num) 
{
    py::buffer_info buf1 = points_.request();
	size_t NUM_TENTS = buf1.shape[0];
	size_t DIM = buf1.shape[1];

	if (DIM != 2) 
		throw std::invalid_argument("The points should be an array with dims [n,2], n>=2");
    if (NUM_TENTS < 2) 
        throw std::invalid_argument("The points should be an array with dims [n,2], n>=2");

    double* ptr1 = (double*)buf1.ptr;
    std::vector<double> points;
    points.assign(ptr1, ptr1 + buf1.size);
    
    std::vector<double> line2d(3);
    std::vector<bool> inliers(NUM_TENTS);

    std::vector<double> probabilities;
    if (sampler == 3 || sampler == 4)
    {
        py::buffer_info buf_prob = probabilities_.request();
        double* ptr_prob = (double*)buf_prob.ptr;
        probabilities.assign(ptr_prob, ptr_prob + buf_prob.size);        
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
    
    py::array_t<bool> inliers_ = py::array_t<bool>(NUM_TENTS);
    py::buffer_info buf3 = inliers_.request();
    bool *ptr3 = (bool *)buf3.ptr;
    for (size_t i = 0; i < NUM_TENTS; i++)
        ptr3[i] = inliers[i];   
    
    if (num_inl  == 0){
        return py::make_tuple(pybind11::cast<pybind11::none>(Py_None),inliers_);
    }
    py::array_t<double> line_ = py::array_t<double>({3});
    py::buffer_info buf2 = line_.request();
    double *ptr2 = (double *)buf2.ptr;
    for (size_t i = 0; i < 3; i++)
        ptr2[i] = line2d[i];
    
    return py::make_tuple(line_, inliers_);
}

py::tuple findHomography(
    py::array_t<double>  correspondences_,
    double w1, 
    double h1,
    double w2,
    double h2,
    py::array_t<double>  probabilities_,
    int sampler,
    bool use_magsac_plus_plus,
    double sigma_th,
    double conf,
    int min_iters,
    int max_iters,
    int partition_num) 
{
	py::buffer_info buf1 = correspondences_.request();
	size_t NUM_TENTS = buf1.shape[0];
	size_t DIM = buf1.shape[1];

	if (DIM != 4) {
		throw std::invalid_argument("x1y1 should be an array with dims [n,4], n>=4");
	}
    if (NUM_TENTS < 4) {
        throw std::invalid_argument("x1y1 should be an array with dims [n,4], n>=4");
    }

    double* ptr1 = (double*)buf1.ptr;
    std::vector<double> correspondences;
    correspondences.assign(ptr1, ptr1 + buf1.size);
    
    std::vector<double> H(9);
    std::vector<bool> inliers(NUM_TENTS);

    std::vector<double> probabilities;
    if (sampler == 3 || sampler == 4)
    {
        py::buffer_info buf_prob = probabilities_.request();
        double* ptr_prob = (double*)buf_prob.ptr;
        probabilities.assign(ptr_prob, ptr_prob + buf_prob.size);        
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
    
    py::array_t<bool> inliers_ = py::array_t<bool>(NUM_TENTS);
    py::buffer_info buf3 = inliers_.request();
    bool *ptr3 = (bool *)buf3.ptr;
    for (size_t i = 0; i < NUM_TENTS; i++)
        ptr3[i] = inliers[i];   
    
    if (num_inl  == 0){
        return py::make_tuple(pybind11::cast<pybind11::none>(Py_None),inliers_);
    }
    py::array_t<double> H_ = py::array_t<double>({3,3});
    py::buffer_info buf2 = H_.request();
    double *ptr2 = (double *)buf2.ptr;
    for (size_t i = 0; i < 9; i++)
        ptr2[i] = H[i];
    
    return py::make_tuple(H_,inliers_);
                         }
namespace
{
    // Copies a 3x3 matrix to a row-major vector.
    std::vector<double> readMatrix3(const DoubleArray& array, const char* what)
    {
        if (array.ndim() != 2 || array.shape(0) != 3 || array.shape(1) != 3)
            throw std::invalid_argument(std::string(what) + " should be an array with dims [3,3]");
        return std::vector<double>(array.data(), array.data() + 9);
    }
}

py::tuple findPnP(DoubleArray correspondences_, DoubleArray probabilities_, int sampler, bool use_magsac_plus_plus,
    double sigma_th, double conf, int min_iters, int max_iters, int partition_num)
{
    std::vector<double> correspondences = readRows(correspondences_, 5, 3, "correspondences");
    std::vector<double> probabilities = readProbabilities(sampler, probabilities_, correspondences_.shape(0));
    std::vector<bool> inliers;
    std::vector<double> pose;
    const int num_inliers = findPnP_(correspondences, inliers, pose, probabilities, sampler, use_magsac_plus_plus,
        sigma_th, conf, min_iters, max_iters, partition_num);
    return py::make_tuple(toArray(num_inliers, pose, { 3, 4 }), toMask(inliers));
}

py::tuple findRadialHomography(DoubleArray correspondences_, DoubleArray probabilities_, int sampler, bool use_magsac_plus_plus,
    double sigma_th, double conf, int min_iters, int max_iters, int partition_num)
{
    std::vector<double> correspondences = readRows(correspondences_, 4, 5, "correspondences");
    std::vector<double> probabilities = readProbabilities(sampler, probabilities_, correspondences_.shape(0));
    std::vector<bool> inliers;
    std::vector<double> H, distortion;
    const int num_inliers = findRadialHomography_(correspondences, inliers, H, distortion, probabilities, sampler,
        use_magsac_plus_plus, sigma_th, conf, min_iters, max_iters, partition_num);
    return py::make_tuple(toArray(num_inliers, H, { 3, 3 }), toArray(num_inliers, distortion, { 2 }), toMask(inliers));
}

py::tuple findHomographyAffine(DoubleArray correspondences_, DoubleArray probabilities_, int sampler, bool use_magsac_plus_plus,
    double sigma_th, double conf, int min_iters, int max_iters, int partition_num)
{
    std::vector<double> correspondences = readRows(correspondences_, 8, 2, "correspondences");
    std::vector<double> probabilities = readProbabilities(sampler, probabilities_, correspondences_.shape(0));
    std::vector<bool> inliers;
    std::vector<double> H;
    const int num_inliers = findHomographyAffine_(correspondences, inliers, H, probabilities, sampler,
        use_magsac_plus_plus, sigma_th, conf, min_iters, max_iters, partition_num);
    return py::make_tuple(toArray(num_inliers, H, { 3, 3 }), toMask(inliers));
}

py::tuple findEssentialMatrixPlanar(DoubleArray correspondences_, DoubleArray K1_, DoubleArray K2_, DoubleArray probabilities_,
    int sampler, bool use_magsac_plus_plus, double sigma_th, double conf, int min_iters, int max_iters, int partition_num)
{
    std::vector<double> correspondences = readRows(correspondences_, 4, 2, "correspondences");
    std::vector<double> K1 = readMatrix3(K1_, "K1"), K2 = readMatrix3(K2_, "K2");
    std::vector<double> probabilities = readProbabilities(sampler, probabilities_, correspondences_.shape(0));
    std::vector<bool> inliers;
    std::vector<double> E;
    const int num_inliers = findEssentialMatrixPlanar_(correspondences, inliers, E, K1, K2, probabilities, sampler,
        use_magsac_plus_plus, sigma_th, conf, min_iters, max_iters, partition_num);
    return py::make_tuple(toArray(num_inliers, E, { 3, 3 }), toMask(inliers));
}

py::tuple findEssentialMatrixGravity(DoubleArray correspondences_, DoubleArray K1_, DoubleArray K2_,
    DoubleArray gravity_src_, DoubleArray gravity_dst_, DoubleArray probabilities_,
    int sampler, bool use_magsac_plus_plus, double sigma_th, double conf, int min_iters, int max_iters, int partition_num)
{
    std::vector<double> correspondences = readRows(correspondences_, 4, 3, "correspondences");
    std::vector<double> K1 = readMatrix3(K1_, "K1"), K2 = readMatrix3(K2_, "K2");
    std::vector<double> gravity_src = readMatrix3(gravity_src_, "gravity_src"), gravity_dst = readMatrix3(gravity_dst_, "gravity_dst");
    std::vector<double> probabilities = readProbabilities(sampler, probabilities_, correspondences_.shape(0));
    std::vector<bool> inliers;
    std::vector<double> E;
    const int num_inliers = findEssentialMatrixGravity_(correspondences, inliers, E, K1, K2, gravity_src, gravity_dst,
        probabilities, sampler, use_magsac_plus_plus, sigma_th, conf, min_iters, max_iters, partition_num);
    return py::make_tuple(toArray(num_inliers, E, { 3, 3 }), toMask(inliers));
}

PYBIND11_PLUGIN(pymagsac) {
                                                                             
    py::module m("pymagsac", R"doc(
        Python module
        -----------------------
        .. currentmodule:: pymagsac
        .. autosummary::
           :toctree: _generate
           
           findEssentialMatrix,
           findFundamentalMatrix,
           findHomography,
           adaptiveInlierSelection
    
    )doc");

  m.def("findPnP", &findPnP, R"doc(Fits a camera pose [R | t] (3x4), x_camera = R x_world + t, to correspondences [n,5]: (u, v, x, y, z),
where (u, v) are normalized image coordinates (K^-1 applied) and (x, y, z) the world point. Returns (pose, mask).
The residual is the reprojection error in normalized coordinates, and sigma_th is in these units: a threshold of
t pixels is t / f for a focal length of f pixels (the default, 0.01, is 10 px at f = 1000 px).
Samplers: 0 (uniform), 1 (PROSAC, rows sorted by decreasing quality), 3 (NG-RANSAC) and 4 (AR-Sampler); 3 and 4 need
probabilities, one per row (for 4, at least one positive). The mask marks the rows whose residual is at most sigma_th.)doc",
        py::arg("correspondences"),
        py::arg("probabilities"),
        py::arg("sampler") = 0,
        py::arg("use_magsac_plus_plus") = true,
        py::arg("sigma_th") = 0.01,
        py::arg("conf") = 0.99,
        py::arg("min_iters") = 50,
        py::arg("max_iters") = 1000,
        py::arg("partition_num") = 5);

  m.def("findRadialHomography", &findRadialHomography, R"doc(Fits a homography H and the radial distortion (lambda_1, lambda_2) of the division model
in each image to correspondences [n,4]: (x1, y1, x2, y2), with coordinates relative to the distortion centres (for example,
the image centres, divided by the image size). An undistorted point is (x, y, 1 + lambda * (x^2 + y^2)).
Returns (H, distortion, mask). The residual is the mean of the forward and backward transfer errors between the undistorted
points, in the units of the coordinates, as sigma_th: with coordinates divided by s pixels, a threshold of t pixels
is t / s (the default, 0.01, is 10 px for s = 1000 px).
Samplers: 0 (uniform), 1 (PROSAC, rows sorted by decreasing quality), 3 (NG-RANSAC) and 4 (AR-Sampler); 3 and 4 need
probabilities, one per row (for 4, at least one positive). The mask marks the rows whose residual is at most sigma_th.)doc",
        py::arg("correspondences"),
        py::arg("probabilities"),
        py::arg("sampler") = 0,
        py::arg("use_magsac_plus_plus") = true,
        py::arg("sigma_th") = 0.01,
        py::arg("conf") = 0.99,
        py::arg("min_iters") = 50,
        py::arg("max_iters") = 1000,
        py::arg("partition_num") = 5);

  m.def("findHomographyAffine", &findHomographyAffine, R"doc(Fits a homography to affine correspondences [n,8]: (x1, y1, x2, y2, a11, a12, a21, a22),
where A = [[a11, a12], [a21, a22]] maps a neighbourhood of (x1, y1) to one of (x2, y2). Returns (H, mask).
Only the minimal solver (two affine correspondences) uses A: the residual is the transfer error of (x1, y1) in pixels,
as sigma_th, and the final model is fitted to the inliers' points (four-point solver).
Samplers: 0 (uniform), 1 (PROSAC, rows sorted by decreasing quality), 3 (NG-RANSAC) and 4 (AR-Sampler); 3 and 4 need
probabilities, one per row (for 4, at least one positive). The mask marks the rows whose residual is at most sigma_th.)doc",
        py::arg("correspondences"),
        py::arg("probabilities"),
        py::arg("sampler") = 0,
        py::arg("use_magsac_plus_plus") = true,
        py::arg("sigma_th") = 1.0,
        py::arg("conf") = 0.99,
        py::arg("min_iters") = 50,
        py::arg("max_iters") = 1000,
        py::arg("partition_num") = 5);

  m.def("findEssentialMatrixPlanar", &findEssentialMatrixPlanar, R"doc(Fits an essential matrix assuming a planar motion (rotation about the camera's y axis,
translation in its xz plane) to correspondences [n,4] in pixels, given the intrinsics K1 and K2.
Returns (E, mask). Only the minimal solver (two points) assumes the planar motion: the final model is a general essential
matrix fitted to the inliers (five-point solver, then bundle adjustment), so it satisfies the constraint only approximately.
The residual is the Sampson distance, in pixels (converted by the mean focal length), as sigma_th.
Samplers: 0 (uniform), 1 (PROSAC, rows sorted by decreasing quality), 3 (NG-RANSAC) and 4 (AR-Sampler); 3 and 4 need
probabilities, one per row (for 4, at least one positive). The mask marks the rows whose residual is at most sigma_th.)doc",
        py::arg("correspondences"),
        py::arg("K1"),
        py::arg("K2"),
        py::arg("probabilities"),
        py::arg("sampler") = 0,
        py::arg("use_magsac_plus_plus") = true,
        py::arg("sigma_th") = 1.0,
        py::arg("conf") = 0.99,
        py::arg("min_iters") = 50,
        py::arg("max_iters") = 1000,
        py::arg("partition_num") = 5);

  m.def("findEssentialMatrixGravity", &findEssentialMatrixGravity, R"doc(Fits an essential matrix to correspondences [n,4] in pixels, given the intrinsics K1 and K2
and the rotations gravity_src and gravity_dst that align each camera's y axis with gravity (so the remaining rotation is about
the vertical axis). Returns (E, mask). Only the minimal solver (three points) uses the gravity directions: the final model
is a general essential matrix fitted to the inliers (five-point solver, then bundle adjustment), so it satisfies the
constraint only approximately. The residual is the Sampson distance, in pixels (converted by the mean focal length), as sigma_th.
Samplers: 0 (uniform), 1 (PROSAC, rows sorted by decreasing quality), 3 (NG-RANSAC) and 4 (AR-Sampler); 3 and 4 need
probabilities, one per row (for 4, at least one positive). The mask marks the rows whose residual is at most sigma_th.)doc",
        py::arg("correspondences"),
        py::arg("K1"),
        py::arg("K2"),
        py::arg("gravity_src"),
        py::arg("gravity_dst"),
        py::arg("probabilities"),
        py::arg("sampler") = 0,
        py::arg("use_magsac_plus_plus") = true,
        py::arg("sigma_th") = 1.0,
        py::arg("conf") = 0.99,
        py::arg("min_iters") = 50,
        py::arg("max_iters") = 1000,
        py::arg("partition_num") = 5);

    m.def("adaptiveInlierSelection", &adaptiveInlierSelection, R"doc(some doc)doc",
        py::arg("x1y1"),
        py::arg("x2y2"),
        py::arg("modelParameters"),
        py::arg("maximumThreshold"),
        py::arg("problemType"),
        py::arg("minimumInlierNumber") = 20);

    m.def("findEssentialMatrix", &findEssentialMatrix, R"doc(some doc)doc",
        py::arg("correspondences"),
        py::arg("K1"),
        py::arg("K2"),
        py::arg("w1"),
        py::arg("h1"),
        py::arg("w2"),
        py::arg("h2"),
        py::arg("probabilities"),
		py::arg("sampler") = 4,
        py::arg("use_magsac_plus_plus") = true,
        py::arg("sigma_th") = 1.0,
        py::arg("conf") = 0.99,
        py::arg("min_iters") = 50,
        py::arg("max_iters") = 1000,
        py::arg("partition_num") = 5);

    m.def("findFundamentalMatrix", &findFundamentalMatrix, R"doc(some doc)doc",
        py::arg("correspondences"),
        py::arg("w1"),
        py::arg("h1"),
        py::arg("w2"),
        py::arg("h2"),
        py::arg("probabilities"),
		py::arg("sampler") = 4,
        py::arg("use_magsac_plus_plus") = true,
        py::arg("sigma_th") = 1.0,
        py::arg("conf") = 0.99,
        py::arg("min_iters") = 50,
        py::arg("max_iters") = 1000,
        py::arg("partition_num") = 5);

    m.def("findRigidTransformation", &findRigidTransformation, R"doc(some doc)doc",
        py::arg("correspondences"),
        py::arg("probabilities"),
		py::arg("sampler") = 4,
        py::arg("use_magsac_plus_plus") = true,
        py::arg("sigma_th") = 1.0,
        py::arg("conf") = 0.99,
        py::arg("min_iters") = 50,
        py::arg("max_iters") = 1000,
        py::arg("partition_num") = 5);

  m.def("findHomography", &findHomography, R"doc(some doc)doc",
        py::arg("correspondences"),
        py::arg("w1"),
        py::arg("h1"),
        py::arg("w2"),
        py::arg("h2"),
        py::arg("probabilities"),
		py::arg("sampler") = 4,
        py::arg("use_magsac_plus_plus") = true,
        py::arg("sigma_th") = 1.0,
        py::arg("conf") = 0.99,
        py::arg("min_iters") = 50,
        py::arg("max_iters") = 1000,
        py::arg("partition_num") = 5); 

  m.def("findLine2D", &findLine2D, R"doc(some doc)doc",
        py::arg("points"),
        py::arg("w1"),
        py::arg("h1"),
        py::arg("probabilities"),
		py::arg("sampler") = 0,
        py::arg("use_magsac_plus_plus") = true,
        py::arg("sigma_th") = 1.0,
        py::arg("conf") = 0.99,
        py::arg("min_iters") = 50,
        py::arg("max_iters") = 1000,
        py::arg("partition_num") = 5); 


  return m.ptr();
}
