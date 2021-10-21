#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include "magsac_python.hpp"


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

py::tuple findFundamentalMatrix(py::array_t<double>  x1y1_,
    py::array_t<double>  x2y2_,
    double w1, 
    double h1,
    double w2,
    double h2,
    bool use_magsac_plus_plus,
    double sigma_th,
    double conf,
    int max_iters,
    int partition_num,
    bool save_samples) {
    py::buffer_info buf1 = x1y1_.request();
    size_t NUM_TENTS = buf1.shape[0];
    size_t DIM = buf1.shape[1];

    if (DIM != 2) {
        throw std::invalid_argument("x1y1 should be an array with dims [n,2], n>=7");
    }
    if (NUM_TENTS < 7) {
        throw std::invalid_argument("x1y1 should be an array with dims [n,2], n>=7");
    }
    py::buffer_info buf1a = x2y2_.request();
    size_t NUM_TENTSa = buf1a.shape[0];
    size_t DIMa = buf1a.shape[1];

    if (DIMa != 2) {
        throw std::invalid_argument("x2y2 should be an array with dims [n,2], n>=7");
    }
    if (NUM_TENTSa != NUM_TENTS) {
        throw std::invalid_argument("x1y1 and x2y2 should be the same size");
    }

    double* ptr1 = (double*)buf1.ptr;
    std::vector<double> x1y1;
    x1y1.assign(ptr1, ptr1 + buf1.size);

    double* ptr1a = (double*)buf1a.ptr;
    std::vector<double> x2y2;
    x2y2.assign(ptr1a, ptr1a + buf1a.size);
    std::vector<double> F(9);
    std::vector<bool> inliers(NUM_TENTS);
    std::vector<size_t> minimal_samples;

    int num_inl = findFundamentalMatrix_(x1y1,
        x2y2,
        inliers,
        F,
        minimal_samples,
        w1,
        h1,
        w2,
        h2,
        use_magsac_plus_plus,
        sigma_th,
        conf,
        max_iters,
        partition_num,
        save_samples);

    py::array_t<bool> inliers_ = py::array_t<bool>(NUM_TENTS);
    py::buffer_info buf3 = inliers_.request();
    bool* ptr3 = (bool*)buf3.ptr;
    for (size_t i = 0; i < NUM_TENTS; i++)
        ptr3[i] = inliers[i];
    if (num_inl == 0) {
        return py::make_tuple(pybind11::cast<pybind11::none>(Py_None), inliers_, pybind11::cast<pybind11::none>(Py_None));
    }
    py::array_t<double> F_ = py::array_t<double>({ 3,3 });
    py::buffer_info buf2 = F_.request();
    double* ptr2 = (double*)buf2.ptr;
    for (size_t i = 0; i < 9; i++)
        ptr2[i] = F[i];
            
    if (save_samples)
    {
        const size_t sample_number = minimal_samples.size() / 5;

        py::array_t<int> minimal_samples_ = py::array_t<int>({ sample_number, 5 });
        py::buffer_info buffer_samples = minimal_samples_.request();
        int* ptr_samples = (int*)buffer_samples.ptr;
        for (size_t i = 0; i < minimal_samples.size(); i++)
            ptr_samples[i] = minimal_samples[i];
            
        return py::make_tuple(F_, inliers_, minimal_samples_);
    }

    return py::make_tuple(F_, inliers_, pybind11::cast<pybind11::none>(Py_None));
}

py::tuple findEssentialMatrix(py::array_t<double>  x1y1_,
    py::array_t<double>  x2y2_,
    py::array_t<double>  K1_,
    py::array_t<double>  K2_,
    double w1, 
    double h1,
    double w2,
    double h2,
    bool use_magsac_plus_plus,
    double sigma_th,
    double conf,
    int max_iters,
    int partition_num,
    bool save_samples) 
{
    py::buffer_info buf1 = x1y1_.request();
    size_t NUM_TENTS = buf1.shape[0];
    size_t DIM = buf1.shape[1];

    if (DIM != 2) {
        throw std::invalid_argument("x1y1 should be an array with dims [n,2], n>=7");
    }
    if (NUM_TENTS < 5) {
        throw std::invalid_argument("x1y1 should be an array with dims [n,2], n>=7");
    }
    py::buffer_info buf1a = x2y2_.request();
    size_t NUM_TENTSa = buf1a.shape[0];
    size_t DIMa = buf1a.shape[1];

    if (DIMa != 2) {
        throw std::invalid_argument("x2y2 should be an array with dims [n,2], n>=7");
    }
    if (NUM_TENTSa != NUM_TENTS) {
        throw std::invalid_argument("x1y1 and x2y2 should be the same size");
    }

    double* ptr1 = (double*)buf1.ptr;
    std::vector<double> x1y1;
    x1y1.assign(ptr1, ptr1 + buf1.size);

    double* ptr1a = (double*)buf1a.ptr;
    std::vector<double> x2y2;
    x2y2.assign(ptr1a, ptr1a + buf1a.size);

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
    std::vector<size_t> minimal_samples;

    int num_inl = findEssentialMatrix_(x1y1,
        x2y2,
        inliers,
        E, 
        K1, 
        K2,
        minimal_samples,
        w1,
        h1,
        w2,
        h2,
        use_magsac_plus_plus,
        sigma_th,
        conf,
        max_iters,
        partition_num,
        save_samples);

    py::array_t<bool> inliers_ = py::array_t<bool>(NUM_TENTS);
    py::buffer_info buf3 = inliers_.request();
    bool* ptr3 = (bool*)buf3.ptr;
    for (size_t i = 0; i < NUM_TENTS; i++)
        ptr3[i] = inliers[i];

    if (num_inl == 0) {
        return py::make_tuple(pybind11::cast<pybind11::none>(Py_None), inliers_, pybind11::cast<pybind11::none>(Py_None));
    }

    py::array_t<double> E_ = py::array_t<double>({ 3,3 });
    py::buffer_info buf2 = E_.request();
    double* ptr2 = (double*)buf2.ptr;
    for (size_t i = 0; i < 9; i++)
        ptr2[i] = E[i];
    
    if (save_samples)
    {
        const size_t sample_number = minimal_samples.size() / 5;

        py::array_t<int> minimal_samples_ = py::array_t<int>({ sample_number, 5 });
        py::buffer_info buffer_samples = minimal_samples_.request();
        int* ptr_samples = (int*)buffer_samples.ptr;
        for (size_t i = 0; i < minimal_samples.size(); i++)
            ptr_samples[i] = minimal_samples[i];
            
        return py::make_tuple(E_, inliers_, minimal_samples_);
    }
    
    return py::make_tuple(E_, inliers_, pybind11::cast<pybind11::none>(Py_None));
}
                                
py::tuple findHomography(py::array_t<double>  x1y1_,
                         py::array_t<double>  x2y2_,
                         double w1, 
                         double h1,
                         double w2,
                         double h2,
						 bool use_magsac_plus_plus,
                         double sigma_th,
                         double conf,
                         int max_iters,
                         int partition_num) {
    py::buffer_info buf1 = x1y1_.request();
    size_t NUM_TENTS = buf1.shape[0];
    size_t DIM = buf1.shape[1];
    
    if (DIM != 2) {
        throw std::invalid_argument( "x1y1 should be an array with dims [n,2], n>=4" );
    }
    if (NUM_TENTS < 4) {
        throw std::invalid_argument( "x1y1 should be an array with dims [n,2], n>=4");
    }
    py::buffer_info buf1a = x2y2_.request();
    size_t NUM_TENTSa = buf1a.shape[0];
    size_t DIMa = buf1a.shape[1];
    
    if (DIMa != 2) {
        throw std::invalid_argument( "x2y2 should be an array with dims [n,2], n>=4" );
    }
    if (NUM_TENTSa != NUM_TENTS) {
        throw std::invalid_argument( "x1y1 and x2y2 should be the same size");
    }
    
    double *ptr1 = (double *) buf1.ptr;
    std::vector<double> x1y1;
    x1y1.assign(ptr1, ptr1 + buf1.size);
    
    double *ptr1a = (double *) buf1a.ptr;
    std::vector<double> x2y2;
    x2y2.assign(ptr1a, ptr1a + buf1a.size);
    std::vector<double> H(9);
    std::vector<bool> inliers(NUM_TENTS);
    
    int num_inl = findHomography_(x1y1,
                    x2y2,
                    inliers,
                    H,
                    w1,
                    h1,
                    w2,
                    h2,
					use_magsac_plus_plus,
                    sigma_th,
                    conf,
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

    m.def("adaptiveInlierSelection", &adaptiveInlierSelection, R"doc(some doc)doc",
        py::arg("x1y1"),
        py::arg("x2y2"),
        py::arg("modelParameters"),
        py::arg("maximumThreshold"),
        py::arg("problemType"),
        py::arg("minimumInlierNumber") = 20);

    m.def("findEssentialMatrix", &findEssentialMatrix, R"doc(some doc)doc",
        py::arg("x1y1"),
        py::arg("x2y2"),
        py::arg("K1"),
        py::arg("K2"),
        py::arg("w1"),
        py::arg("h1"),
        py::arg("w2"),
        py::arg("h2"),
        py::arg("use_magsac_plus_plus") = true,
        py::arg("sigma_th") = 1.0,
        py::arg("conf") = 0.99,
        py::arg("max_iters") = 1000,
        py::arg("partition_num") = 5,
        py::arg("save_samples") = false);

    m.def("findFundamentalMatrix", &findFundamentalMatrix, R"doc(some doc)doc",
        py::arg("x1y1"),
        py::arg("x2y2"),
        py::arg("w1"),
        py::arg("h1"),
        py::arg("w2"),
        py::arg("h2"),
        py::arg("use_magsac_plus_plus") = true,
        py::arg("sigma_th") = 1.0,
        py::arg("conf") = 0.99,
        py::arg("max_iters") = 1000,
        py::arg("partition_num") = 5,
        py::arg("save_samples") = false);
    

  m.def("findHomography", &findHomography, R"doc(some doc)doc",
        py::arg("x1y1"),
        py::arg("x2y2"),
        py::arg("w1"),
        py::arg("h1"),
        py::arg("w2"),
        py::arg("h2"),
        py::arg("use_magsac_plus_plus") = true,
        py::arg("sigma_th") = 1.0,
        py::arg("conf") = 0.99,
        py::arg("max_iters") = 1000,
        py::arg("partition_num") = 5); 


  return m.ptr();
}
