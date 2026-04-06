#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> //unneeded?
#include <pybind11/stl_bind.h> //unneeded?
#include <torch/torch.h> //unneeded?
#include <chrono>
#include <algorithm>
#include <unordered_set>
#include <cmath>
#include <csignal>

#include "cpp_sampler.hpp"

#define FP double
PYBIND11_MAKE_OPAQUE(std::vector<FP>)
PYBIND11_MAKE_OPAQUE(std::vector<unsigned short>)
namespace py = pybind11;

#if WRAPPER_BUILD && !INTERNAL_TIMING//disable printing timings when called from python
void tic(std::string,int mode){}
void toc(std::string){}
#endif

py::array_t<FP> pyforward(cpp_sampler* smplr, const py::list& theory, size_t number_of_events)
{
    tic("pybind");
    // *TEMPORARY* ensure that all my hand-rolled pytorch tensor operations cut the mustard
    //smplr->test_functions();

    /* Get the information from theory: We are expecting 6 items:
        i) bins and crossections in x and Q^2 coordinates = 4 variables
        ii) cross section weights and phase space acceptance = 2 variables
       x_bins, xsec_x, Q2_bins, xsec_Q2, weights, acceptance = theory_out[:6] */
    const py::array_t<FP>& py_x_bins     = theory[0].cast<py::array_t<FP>>();
    const py::array_t<FP>& py_x_sec_x    = theory[1].cast<py::array_t<FP>>();
    const py::array_t<FP>& py_q2_bins    = theory[2].cast<py::array_t<FP>>();
    const py::array_t<FP>& py_x_sec_q2   = theory[3].cast<py::array_t<FP>>();
    const py::array_t<FP>& py_weights    = theory[4].cast<py::array_t<FP>>();
    const py::array_t<unsigned short>& py_acceptance = theory[5].cast<py::array_t<unsigned short>>();

    /*
    FP* ptr_x_bins     = reinterpret_cast<FP*>(py_x_bins.request().ptr);
    FP* ptr_x_sec_x    = reinterpret_cast<FP*>(py_x_sec_x.request().ptr);
    FP* ptr_q2_bins    = reinterpret_cast<FP*>(py_q2_bins.request().ptr);
    FP* ptr_x_sec_q2   = reinterpret_cast<FP*>(py_x_sec_q2.request().ptr);
    FP* ptr_weights    = reinterpret_cast<FP*>(py_weights.request().ptr);
    int* ptr_acceptance   = reinterpret_cast<int*>(py_acceptance.request().ptr);

    std::vector<FP> x_bins     {ptr_x_bins,ptr_x_bins+py_x_bins.size()};
    std::vector<FP> x_sec_x    {ptr_x_sec_x,ptr_x_sec_x+py_x_sec_x.size()};
    std::vector<FP> q2_bins    {ptr_q2_bins,ptr_q2_bins+py_q2_bins.size()};
    std::vector<FP> x_sec_q2   {ptr_x_sec_q2,ptr_x_sec_q2+py_x_sec_q2.size()};
    std::vector<FP> weights    {ptr_weights,ptr_weights+py_weights.size()};
    std::vector<int> acceptance {ptr_acceptance,ptr_acceptance+py_acceptance.size()};
    */
    py::buffer_info info_x_bins     = py_x_bins.request();
    py::buffer_info info_x_sec_x    = py_x_sec_x.request();
    py::buffer_info info_q2_bins    = py_q2_bins.request();
    py::buffer_info info_x_sec_q2   = py_x_sec_q2.request();
    py::buffer_info info_weights    = py_weights.request();
    py::buffer_info info_acceptance = py_acceptance.request();
    //*NOTE*: I think shape[0] corresponds to the batch id
    matrix<FP> x_bins     (reinterpret_cast<FP*>(info_x_bins.ptr),    info_x_bins.shape[1],    info_x_bins.shape[2]);
    matrix<FP> x_sec_x    (reinterpret_cast<FP*>(info_x_sec_x.ptr),   info_x_sec_x.shape[1],   info_x_sec_x.shape[2], info_x_sec_x.shape[3]);
    matrix<FP> q2_bins    (reinterpret_cast<FP*>(info_q2_bins.ptr),   info_q2_bins.shape[1],   info_q2_bins.shape[2]);
    matrix<FP> x_sec_q2   (reinterpret_cast<FP*>(info_x_sec_q2.ptr),  info_x_sec_q2.shape[1],  info_x_sec_q2.shape[2], info_x_sec_q2.shape[3]);
    matrix<FP> weights    (reinterpret_cast<FP*>(info_weights.ptr),   info_weights.shape[1],   info_weights.shape[2]);
    matrix<unsigned short>   acceptance (reinterpret_cast<unsigned short*>(info_acceptance.ptr),info_acceptance.shape[1],info_acceptance.shape[2]);
    toc("pybind");

    /* Compute flat grid-indices which help to avoid another for-loop over the grid itself:
       based off: npy_grid_idx = self.calc_grid_indices(weights[0].size()[0],weights[0].size()[1]) */
    //printf("comparing times of calc_grid_indices...\n");
    //auto t0 = std::chrono::high_resolution_clock::now();
    matrix<size_t> grid_idx = smplr->calc_grid_indices(py_weights.shape(1),py_weights.shape(2));
    //auto t1 = std::chrono::high_resolution_clock::now();
    //auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
    //printf("C++ version took: %zu us\n", us.count());

    //std::vector<FP> fltvec(grid_idx.begin(), grid_idx.end());
    //py::array_t<FP> result = py::array_t<FP>(fltvec.size(), fltvec.data());
    py::array_t<FP> result;
    /* Get batch size from theory outputs: */
    //size_t batch_size = x_bins.shape(0); // equivalent of: batch_size = x_bins.size()[0]
    size_t batch_size = py_x_bins.shape(0); // equivalent of: batch_size = x_bins.size()[0]
    // batch_size is effectively the number of pages.
    // this is easy to implement if we extract a matrix at a given page
#ifndef WRAPPER_BUILD //disable this check for use in the pybind wrapper
    if (batch_size > 1) raise(SIGINT);
#endif
    //assert(batch_size == 1 && "We currently only support a batch-size of 1!");

    /* Run everything sequentially and generate n_events for each prediction, where #events_generated = batch_size * n_events
    +++++++++++++++++++++++++++++++*/
    tic("total");
    for (size_t i = 0; i < batch_size; i++){
      /*detached_weights = weights[i].detach().cpu()
        n = torch.abs(detached_weights*n_events).to(torch.int).reshape(-1,1)
        max_n = torch.max(n) */
      size_t offset_start_of_batch = i*(py_weights.shape(1)*py_weights.shape(2));
      size_t offset_end_of_batch = (i+1)*(py_weights.shape(1)*py_weights.shape(2));
      std::vector<FP> n_as_FP(py_weights.data()+offset_start_of_batch, py_weights.data()+offset_end_of_batch);
      std::vector<size_t> n(n_as_FP.size());
      std::transform(n_as_FP.begin(), n_as_FP.end(), n.begin(),  [&number_of_events](FP element){return static_cast<size_t>(std::floor(element*number_of_events));});
      auto max_n_ptr = std::max_element(n.begin(),n.end());
      size_t max_n = *max_n_ptr;
      /*weight_tensor_cpu = self.calc_weight_tensor(n)
        weight_tensor_gpu = torch.as_tensor(weight_tensor_cpu,device=self.devices,dtype=torch.FP32)*/
      matrix<FP> weight_mat = smplr->calc_weight_matrix(n,max_n);
      weight_mat.assert_shape(361,13057,1);
      /*
      data_list.append(self.forward_single_sample(x_bins[i], xsec_x[i], Q2_bins[i], xsec_Q2[i],acceptance[i],weight_tensor_gpu,torch_grid_idx,max_n))
      */
      tic("forward_single_sample");
      auto res = smplr->forward_single_sample(x_bins,x_sec_x,q2_bins,x_sec_q2,acceptance,weight_mat,grid_idx,max_n);
      toc("forward_single_sample");
      tic("pybind_wrapping_result");
      result = py::array_t<FP>(py::buffer_info(res.ptr(),  //ptr
                                               sizeof(FP), //itemsize
                                               py::format_descriptor<FP>::format(),
                                               2, // ndim
                                               std::vector<size_t> {res.rows(), res.cols() }, // shape
                                               std::vector<size_t> {res.cols() * sizeof(FP), sizeof(FP)})); // strides
      toc("pybind_wrapping_result");
    }
    //+++++++++++++++++++++++++++++++
    // return torch.cat(data_list,dim=0)
    //auto t1 = std::chrono::high_resolution_clock::now();
    //auto us = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    //printf("internal C++ sampler took: %f ms\n", ((float)us.count()/1000.0));
    toc("total");
    return(result);
}

void sampler_delay_recording(cpp_sampler* smplr){
  smplr->delay_recording();
}

void sampler_start_recording(cpp_sampler* smplr){
  smplr->start_recording();
}

PYBIND11_MODULE(cpp_wrapper, m) {
    m.doc() = "C++ sampler";
    m.def("forward", &pyforward);
    pybind11::class_<cpp_sampler>(m, "cpp_sampler")
        .def(py::init())
        .def(py::init<std::string>())
        .def("forward", &pyforward)
        .def("delay_recording", &sampler_delay_recording)
        .def("start_recording", &sampler_start_recording);
}
