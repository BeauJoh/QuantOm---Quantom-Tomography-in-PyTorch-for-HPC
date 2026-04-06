#include "omp_sampler.hpp"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>
#include <cassert>
#include <sys/stat.h>
#include <random>

#if INTERNAL_TIMING
    const bool internal_timing_enabled = true;
#else
    const bool internal_timing_enabled = false;
#endif

//private instance variables---rather than expose the SYCL interface in the header file
struct observation_settings{
  bool transpose;
  int x;
  int y;
  int z;

//C++ equivalent of self.gen_settings = {
//  'x': [True,1,0,0],
//  'q2':[False,0,1,1]
//}

  observation_settings(int is_q2){
    switch(is_q2){
      case 0:
        transpose = true;
        x=1;
        y=0;
        z=0;
        break;
      case 1:
        transpose = false;
        x=0;
        y=1;
        z=1;
        break;
    }
  }
};

void omp_sampler::delay_recording(){
  disable_recording();
}

void omp_sampler::start_recording(){
  enable_recording();
}

//sampler functions:
omp_sampler::omp_sampler(std::string internal_timing_filename){
  if (internal_timing_filename.length() != 0) {
    assign_filename_for_recording(internal_timing_filename);
    //the shortlist of regions to log and plot
    mark_for_recording("linear_interpolation");
    mark_for_recording("calc_rho");
    mark_for_recording("calc_cdf");
    mark_for_recording("calc_grid_indices");
    mark_for_recording("calc_weight_matrix");
    mark_for_recording("uobs_fill_and_flattern");
    mark_for_recording("bin_obs_flattern");
    mark_for_recording("reshape");
    mark_for_recording("filter");
    mark_for_recording("pybind");
    mark_for_recording("total");
    mark_for_recording("forward_single_sample");
    mark_for_recording("pybind_wrapping_result");
  }
  ref_sampler = cpp_sampler();
  printf("OpenMP is using %i threads\n",omp_get_max_threads());
}

// C++ OMP
// Compute grid indices so that we are able to vectorize the sampler
// ***************************
matrix<size_t> omp_sampler::calc_grid_indices(size_t size_dim_0,size_t size_dim_1){
  //if constexpr (internal_timing_enabled) tic("calc_grid_indices");
  auto res = ref_sampler.calc_grid_indices(size_dim_0,size_dim_1);
  //if constexpr (internal_timing_enabled) toc("calc_grid_indices");
  return res;
}
//***************************


matrix<FP> omp_sampler::calc_weight_matrix(const std::vector<size_t>& n, size_t max_n){
#if SHOW_OLD
  //if constexpr (internal_timing_enabled) tic("omp_sampler:calc_weight_matrix old");
  auto ref = ref_sampler.calc_weight_matrix(n,max_n);
  //if constexpr (internal_timing_enabled) toc("omp_sampler:calc_weight_matrix old");
#endif

  if constexpr (internal_timing_enabled) tic("calc_weight_matrix");
  matrix<FP> weight_mat(n.size(),max_n);
  FP* _m = weight_mat.ptr();
  const size_t p = 0, _cols = weight_mat.cols(), _page = weight_mat.pages();
  #pragma omp parallel for shared(_m) schedule(static,16)
  for (size_t i = 0; i < n.size(); i++)
    for (size_t j = 0; j < n[i]; j++)
       _m[i*_cols*_page + p*_cols + j] = 1.0f;
       //weight_mat(i,j) = 1.0f;
  if constexpr (internal_timing_enabled) toc("calc_weight_matrix");
#if SHOW_OLD
  weight_mat.force_assert_equal(ref);
#endif
  /*
  //checking
  for (size_t i = 0; i < n.size(); i++)
    for (size_t j = 0; j < n[i]; j++)
      if (res(i,j)!= weight_mat(i,j)) {
        printf("error in implementation, i = %zu, j = %zu, res[] = %f, weight_mat = %f\n",i,j,res(i,j),weight_mat(i,j));
      }
  */
  return weight_mat;
}

matrix<FP> omp_sampler::trapez(const matrix<FP>& my, FP dx, const matrix<FP>& mx){
  if constexpr (internal_timing_enabled) tic("trapez");
  //mx.rows x my.rows
  //@ first index:
  //my=[0.0038, 0.0100, 0.0134, 0.0159, 0.0180, 0.0197, 0.0211, 0.0224, 0.0235, 0.0244]
  //mx=[0.0010, 0.0068, 0.0127, 0.0185, 0.0243, 0.0302, 0.0360, 0.0419, 0.0477, 0.0535]
  assert(mx.rows() == my.rows() && mx.cols() == my.cols() && mx.pages() == my.pages());
  matrix<FP> result(my.rows(),1,1);
  //#pragma omp parallel for
  for (size_t r = 0; r < my.rows(); r++){
    FP sum = 0;
    //#pragma omp parallel for reduction(+: sum)
    for (size_t c = 1; c < my.cols(); c++){
      sum += ((mx(r,c) - mx(r,c-1))/2) * (my(r,c)+my(r,c-1));
    }
    result(r) = sum;
  }
  if constexpr (internal_timing_enabled) toc("trapez");
  return(result);
}

matrix<FP> omp_sampler::cumulative_trapez(const matrix<FP>& my, const matrix<FP>& mx){
  if constexpr (internal_timing_enabled) tic("omp_sampler:cumulative_trapez");
  matrix<FP> my_inter; my_inter.explicit_copy(my);
  matrix<FP> mx_inter; mx_inter.explicit_copy(mx);
  //assert(mx.rows() == my.rows() && mx.cols() == my.cols() && mx.pages() == my.pages());
  matrix<FP> result(my.rows(),my.cols(),my.pages());
  //#pragma omp parallel for
  for (size_t r = 0; r < my.rows(); r++){
    FP sum = 0;
    for (size_t c = 1; c < my.cols(); c++){
      sum += ((mx(r,c) - mx(r,c-1))/2) * (my(r,c)+my(r,c-1));
      result(r,c) = sum;
    }
  }
  if constexpr (internal_timing_enabled) toc("omp_sampler:cumulative_trapez");
  return(result);
}

matrix<FP> omp_sampler::trapezoid(const matrix<FP>& my, float dx, const matrix<FP>& mx, size_t dim){
  if constexpr (internal_timing_enabled) tic("omp_sampler:trapezoid");
  bool mx_provided = mx.size() != 0;
  switch(dim){
    //by_row
    case BY_ROW: {
      matrix<FP> result(my.rows(),1,1);
      if (mx_provided){
        //assert(my.rows() == mx.rows());
        bool is_vector = (mx.cols() == 1 && mx.pages() == 1);
        //#pragma omp parallel for
        for (size_t c = 0; c < my.cols(); c++){
          FP sum = 0;
          for (size_t p = 0; p < my.pages(); p++){
          for (size_t r = 1; r < my.rows(); r++){
              size_t cc = c;
              size_t pp = p;
              if (is_vector) {cc = 0; pp = 0;}
              sum += ((mx(r,cc,pp)-mx(r-1,cc,pp))/2.0f) * (my(r,c,p)+my(r-1,c,p));
            }
          }
          result(c) = dx*sum;
        }
      } else {
        //#pragma omp parallel for
        for (size_t c = 0; c < my.cols(); c++){
          for (size_t p = 0; p < my.pages(); p++){
            FP sum = 0;
            for (size_t r = 1; r < my.rows(); r++){
              sum += (dx/2) * (my(r,c,p)+my(r-1,c,p));
            }
            result(c,p) = sum;
          }
        }
      }
      if constexpr (internal_timing_enabled) toc("omp_sampler:trapezoid");
      return(result);
      break;
    }
    //by column
    case BY_COL: {
      matrix<FP> result(my.cols(),my.pages(),1);
      if (mx_provided){
        //#pragma omp parallel for
        for (size_t r = 0; r < my.rows(); r++){
          for (size_t p = 0; p < my.pages(); p++){
            FP sum = 0;
            for (size_t c = 1; c < my.cols(); c++){
              sum += ((mx(r,c,p)-mx(r,c-1,p))/2) * (my(r,c,p)+my(r,c-1,p));
            }
            result(r,p) = sum;
          }
        }
      } else {
        //#pragma omp parallel for
        for (size_t r = 0; r < my.rows(); r++){
          for (size_t p = 0; p < my.pages(); p++){
            FP sum = 0;
            for (size_t c = 1; c < my.cols(); c++){
              sum += (dx/2) * (my(r,c,p)+my(r,c-1,p));
            }
            result(r,p) = sum;
          }
        }
      }
      if constexpr (internal_timing_enabled) toc("omp_sampler:trapezoid");
      return(result);
      break;
    }
    //by page
    case BY_PAGE: {
      matrix<FP> result(my.rows(),1);
      if (mx_provided){
        bool is_vector = (mx.rows() == 1 && mx.cols() == 1);
        //#pragma omp parallel for
        for (size_t r = 0; r < my.rows(); r++){
          FP sum = 0;
          for (size_t c = 0; c < my.cols(); c++){
            for (size_t p = 1; p < my.pages(); p++){
              size_t cc = c;
              size_t rr = r;
              if (is_vector) {cc = 0; rr = 0;}

              sum += ((mx(rr,cc,p)-mx(rr,cc,p-1))/2.0) * (my(rr,cc,p)+my(rr,cc,p-1));
            }
          }
          result(r) = sum;
        }
      } else {
        //#pragma omp parallel for
        for (size_t p = 0; p < my.pages(); p++){
          for (size_t c = 0; c < my.cols(); c++){
            FP sum = 0;
            for (size_t r = 1; r < my.rows(); r++){
              sum += (dx/2) * (my(r,c,p)+my(r-1,c,p));
            }
            result(p,c) = sum;
          }
        }
      }
      if constexpr (internal_timing_enabled) toc("omp_sampler:trapezoid");
      return(result);
      break;
    }
    default: {
      assert(false && "not implemented for 3 or more dimensions");
      if constexpr (internal_timing_enabled) toc("omp_sampler:trapezoid");
      return(matrix<FP>());
      break;
    }
  }
}


//# Components to calculate the CDF in x and Q2 coordinates:
// #***************************
//    # Calculate the norm: Integral(xsec(x) dx), with: x = bins
//    # which is approximated by the trapezoid rule:
//    def calc_norm(self,bins,xsec):
//        return torch.trapezoid(xsec,bins)

//# Determine the density: rho = xsec / norm
matrix<FP> omp_sampler::calc_rho(const matrix<FP>& bins,
                                 const matrix<FP>& xsec){
  //if constexpr (internal_timing_enabled) tic("calc_rho");
  auto res = ref_sampler.calc_rho(bins,xsec);
  //if constexpr (internal_timing_enabled) toc("calc_rho");
  return res;
}
//***************************

// Calculate the inverse from the given CDF, by performing a linear interpolation from the
// binned CDF from theory to the corresponding bin (in either x or Q2)
matrix<FP> omp_sampler::linear_interpolation(const matrix<FP>& u, const matrix<FP>& cdf, const matrix<FP>& bin, const matrix<FP>& weights){
  //use the most optimized version by default
  //return noindx_linear_interpolation(u,cdf,bin,weights);
  if constexpr (internal_timing_enabled) tic("linear_interpolation");
  auto res = noindx_linear_interpolation(u,cdf,bin,weights);
  if constexpr (internal_timing_enabled) toc("linear_interpolation");
  return res;

}

matrix<FP> omp_sampler::master_linear_interpolation(const matrix<FP>& u, const matrix<FP>& cdf, const matrix<FP>& bin, const matrix<FP>& weights){
    if constexpr (internal_timing_enabled) tic("omp_sampler:master_linear_interpolation");
    //This function is directly taken from Nobuo Satos original code
    //The core function: bin(u) = m*u + b

    //# Determine m = (bin[i+1] - bin[i]) / (cdf[i+1] - cdf[i])
    //m = (bin[1:] - bin[:-1]) / (cdf[1:] - cdf[:-1] + 1e-5)
    matrix<FP> res(u.rows(),u.cols());

    #pragma omp parallel shared(res)
    {
      matrix<FP> m(cdf.rows(),cdf.cols()-1);
      #pragma omp for
      for(size_t r = 0; r < cdf.rows(); r++){
        for(size_t c = 0; c < cdf.cols()-1; c++){
            m(r,c) = (bin(r,c+1)-bin(r,c)) / (cdf(r,c+1)-cdf(r,c) + 1e-5);
        }
      }
      //# b = -m*cdf[i] + bin[i]
      //b = bin[:-1] - (m * cdf[:-1])
      matrix<FP> b(cdf.rows(),cdf.cols()-1);
      #pragma omp for
      for(size_t r = 0; r < cdf.rows(); r++){
        for(size_t c = 0; c < cdf.cols()-1; c++){
            b(r,c) = bin(r,c) - (m(r,c)*cdf(r,c));
        }
      }
      //# Make sure that we get the proper indices that obey: u >= cdf
      //indicies = torch.sum(torch.ge(u[:, None], cdf[None, :]), 1) - 1
      matrix<unsigned short> indices(u.rows(),u.cols());
      #pragma omp for
      for(size_t r = 0; r < indices.rows(); r++){
        for(size_t c = 0; c < indices.cols(); c++){
          FP this_u = u(r,c);
          for(size_t cdf_c = 0; cdf_c < cdf.cols(); cdf_c++){
            FP this_cdf = cdf(r,cdf_c);
            if (this_u >= this_cdf) indices(r,c)+=1;
          }
          indices(r,c)--;
        }
      }
      //# They should lie between 0 and n_points -1
      //indicies = torch.clamp(indicies, 0,m.size()[0] - 1)
      FP clamp_min=0;
      FP clamp_max=m.cols()-1;
      #pragma omp for
      for (size_t i = 0; i < indices.size(); i++){
        if (indices[i] < clamp_min) indices[i] = clamp_min;
        if (indices[i] > clamp_max) indices[i] = clamp_max;
      }
      //# And return the interpolation:
      //return m[indicies] * u + b[indicies]
      //# with:
      //obs_gen = obs_gen*weights
      //matrix<FP> res(indices.rows(),indices.cols());
      #pragma omp for
      for(size_t r = 0; r < indices.rows(); r++){
        for(size_t c = 0; c < indices.cols(); c++){
          res(r,c) = (m(r,indices(r,c)) * u(r,c) + b(r,indices(r,c))) * weights(r,c);
        }
      }
    }//end of parallel region
    if constexpr (internal_timing_enabled) toc("omp_sampler:master_linear_interpolation");
    return res;
}
//#***************************


matrix<FP> omp_sampler::noindx_linear_interpolation(const matrix<FP>& u, const matrix<FP>& cdf, const matrix<FP>& bin, const matrix<FP>& weights){
    if constexpr (internal_timing_enabled) tic("omp_sampler:noindx_linear_interpolation");
    matrix<FP> res(u.rows(),u.cols());
    FP* _res = res.ptr();
    #pragma omp parallel shared(_res)
    {
      size_t _rows = cdf.rows(), _cols = cdf.cols(), _page = cdf.pages();
      matrix<FP> m(_rows,_cols-1);
      matrix<FP> b(cdf.rows(),cdf.cols()-1);
      FP* _m = m.ptr();
      FP* _b = b.ptr();
      const FP* _u = u.ptr();
      const FP* _bin = bin.ptr();
      const FP* _cdf = cdf.ptr();
      short clamp_min=0;
      short clamp_max=m.cols()-1;
      const FP* _weights = weights.ptr();
      size_t p = 0, _res_cols = _cols-1;

      #pragma omp for schedule(static,16)
      for(size_t r = 0; r < _rows; r++){
        for(size_t c = 0; c < _cols-1; c++){
          size_t this_id = r*_cols*_page + p*_cols + c; 
          size_t next_id = r*_cols*_page + p*_cols + c+1;
          size_t res_id =  r*_res_cols*_page + p*_res_cols + c;
          //The core function: bin(u) = m*u + b
          //# Determine m = (bin[i+1] - bin[i]) / (cdf[i+1] - cdf[i])
          //m = (bin[1:] - bin[:-1]) / (cdf[1:] - cdf[:-1] + 1e-5)
          _m[res_id] = (_bin[next_id] - _bin[this_id]) /
                       (_cdf[next_id] - _cdf[this_id] + 1e-5);
          //# b = -m*cdf[i] + bin[i]
          //b = bin[:-1] - (m * cdf[:-1])
          _b[res_id] = _bin[this_id] - (m[res_id]*_cdf[this_id]);
        }
      }
      //# Make sure that we get the proper indices that obey: u >= cdf
      //indicies = torch.sum(torch.ge(u[:, None], cdf[None, :]), 1) - 1
      _rows = u.rows(); _cols = u.cols(); _page = u.pages();
      size_t _cdf_rows = cdf.rows(), _cdf_cols = cdf.cols(), _cdf_page = cdf.pages(), _b_cols = b.cols(), _b_page = b.pages(), _m_cols = m.cols(), _m_page = m.pages();
      #pragma omp for schedule(static,16)
      for(size_t r = 0; r < _rows; r++){
        for(size_t c = 0; c < _cols; c++){
          short _index = -1;
          size_t this_id = r*_cols*_page + p*_cols + c; 
          //FP this_u = u(r,c);
          FP this_u = _u[this_id];
          for(size_t cdf_c = 0; cdf_c < _cdf_cols; cdf_c++){
            size_t cdf_id = r*_cdf_cols*_cdf_page+ p*_cdf_cols + cdf_c;
            FP this_cdf = _cdf[cdf_id];
            if (this_u >= this_cdf) _index++;
          }
          //# They should lie between 0 and n_points -1
          //# ref: indicies = torch.clamp(indicies, 0,m.size()[0] - 1)
          if (_index < clamp_min) _index = clamp_min;
          if (_index > clamp_max) _index = clamp_max;
          //# And return the interpolation:
          // ref: return m[indicies] * u + b[indicies]
          //# with:
          // ref: obs_gen = obs_gen*weights
          size_t m_lookup_id = r*_m_cols*_m_page + p*_m_cols + _index;
          size_t b_lookup_id = r*_b_cols*_b_page + p*_b_cols + _index;
          _res[this_id] = (_m[m_lookup_id] * _u[this_id] + _b[b_lookup_id]) * _weights[this_id];
        }
      }
    }//end of parallel region
  if constexpr(internal_timing_enabled) toc("omp_sampler:noindx_linear_interpolation");
  return res;
}
//#***************************


//# Now compute the CDF itself:
matrix<FP> omp_sampler::calc_cdf(const matrix<FP>& bins,
                                 const matrix<FP>& xsec,
                                 const matrix<unsigned short>& acceptance,
                                 bool transpose){
  //if constexpr(internal_timing_enabled) tic("calc_cdf");
  auto res = ref_sampler.calc_cdf(bins,xsec,acceptance,transpose);
  //if constexpr (internal_timing_enabled) toc("calc_cdf");
  return res;
}

// Generate events, based on the provided x / Q2 grid, the acceptance matrix, weight tensors and grid index tensors: 
//***************************
/*
# Define a helper function that generates one variable: (e.g. x or Q2)
def generate_single_observable(self,obs_bins,obs_xsec,acceptance_matrix,weight_tensor,grid_index_tensor,n_max,obs_type):
    settings = self.gen_settings[obs_type]

    # Compute the CDF:
    cdf_obs = self.calc_cdf(obs_bins,obs_xsec,acceptance_matrix,settings[0])

    # First generate u:
    u_obs = torch.rand((grid_index_tensor.size()[0],n_max),device=self.devices,dtype=torch.float32)

    # Use the cdf in obs that we just computed and combine them with the index tensor,
    # i.e. assign the proper values to the grid position:
    cdf_obs_flat = torch.flip(cdf_obs[grid_index_tensor[:,settings[1]],grid_index_tensor[:,settings[2]]],dims=(1,))
    
    # Do the same for the bins in x:
    bin_obs_flat = torch.flip(obs_bins[grid_index_tensor[:,settings[3]]],dims=(1,))

    # Now utilize the linear interpolation function and compute x:
    # Also, we can now leverage that the indices are flat and run a vectorization:
    obs_gen = torch.vmap(self.linear_interpolation,in_dims=0,randomness=self.vmap_randomness)(u_obs,cdf_obs_flat,bin_obs_flat) * weight_tensor 
    return obs_gen.flatten()[:,None]
*/
matrix<FP> omp_sampler::generate_single_observable(const matrix<FP>& obs_bins,
                                                   const matrix<FP>& obs_xsec,
                                                   const matrix<unsigned short>& acceptance,
                                                   const matrix<FP>& weights,
                                                   const matrix<size_t>& grid_index,
                                                   size_t      n_max,
                                                   int         obs_type){

  observation_settings settings(obs_type);
  //# Compute the CDF:
  matrix<FP> cdf_obs = this->calc_cdf(obs_bins,obs_xsec,acceptance,settings.transpose);
  if constexpr (internal_timing_enabled) tic("uobs_fill_and_flattern");
  //# First generate u:
  //u_obs = torch.rand(
  //    (grid_index_tensor.size()[0], n_max),
  //    device=self.devices,
  //    dtype=torch.float32,
  //)
  matrix<FP> u_obs(grid_index.rows(),n_max);
#if TEST_FILL
  //fill with 0.5 for debug:
  FP val = 0.5;
  FP* _data = u_obs.ptr();
  #pragma omp parallel for
  for (size_t i = 0; i < _num_elements; i++){
    _data[i] = val;
  }
#else
  //random fill
  std::random_device rd;
  std::vector<std::mt19937> generators;
  for (int i = 0, N = omp_get_max_threads(); i < N; ++i) {
    generators.emplace_back(rd());
  }
  std::uniform_real_distribution<> dist(0.0, 1.0);
  size_t _num_elements = u_obs.size();
  FP* _data = u_obs.ptr();
  #pragma omp parallel for shared(dist) schedule(static, 16)
  for (size_t i = 0; i < _num_elements; i++){
    std::mt19937& e2 = generators[omp_get_thread_num()];
    _data[i] = dist(e2);
  }
#endif
  matrix<FP> cdf_obs_flat(grid_index.rows(),cdf_obs.pages());
  {
  FP* wo = cdf_obs_flat.ptr(); size_t wo_cols = cdf_obs_flat.cols(), wo_page = cdf_obs_flat.pages();
  const long unsigned int * gi = grid_index.ptr(); size_t gi_rows = grid_index.rows(), gi_cols = grid_index.cols(), gi_page = grid_index.pages();
  const FP * co = cdf_obs.ptr(); size_t co_cols = cdf_obs.cols(), co_page = cdf_obs.pages();
  #pragma omp parallel for
  for (size_t r = 0; r < gi_rows; r++){
    //size_t x = grid_index(r,settings.x);
    size_t x = gi[r*gi_cols*gi_page + settings.x];
    //size_t y = grid_index(r,settings.y);
    size_t y = gi[r*gi_cols*gi_page + settings.y];
    for (size_t p = 0; p < co_page; p++){
      //cdf_obs_flat_idx(r,p) = cdf_obs(x,y,p);
      wo[r*wo_cols*wo_page + p] = co[x*co_cols*co_page + p*co_cols + y];
    }
  }
  }
  if constexpr (internal_timing_enabled) toc("uobs_fill_and_flattern");
  //# Do the same for the bins in x:
  //bin_obs_flat = obs_bins[grid_index_tensor[:, settings[3]]]
  if constexpr (internal_timing_enabled) tic("bin_obs_flattern");
  matrix<FP> bin_obs_flat(grid_index.rows(),obs_bins.cols());
  {
  FP* bof = bin_obs_flat.ptr();
  const FP* ob = obs_bins.ptr();
  const long unsigned int * gi = grid_index.ptr();
  size_t gi_rows = grid_index.rows(), gi_cols = grid_index.cols(), gi_page = grid_index.pages(), ob_col = obs_bins.cols(), bof_col = bin_obs_flat.cols();
  #pragma omp parallel for
  for (size_t r = 0; r < gi_rows; r++){
    size_t z = gi[r*gi_cols*gi_page + settings.z];
    for (size_t c = 0; c < ob_col; c++){
      bof[r * bof_col+c] = ob[z * ob_col+c];
    }
  }
  }
  if constexpr (internal_timing_enabled) toc("bin_obs_flattern");
  //# Now utilize the linear interpolation function and compute x:
  //# Also, we can now leverage that the indices are flat and run a vectorization:
  //obs_gen = (
  //    torch.vmap(
  //        self.linear_interpolation, in_dims=0, randomness=self.vmap_randomness
  //    )(u_obs, cdf_obs_flat, bin_obs_flat)
  //    * weight_tensor
  //)
  matrix<FP> obs_gen = linear_interpolation(u_obs, cdf_obs_flat, bin_obs_flat, weights);

  //return obs_gen.flatten()[:, None]
#if SHOW_OLD
  if constexpr (internal_timing) tic("omp_sampler:flattern res old");
  matrix<FP> flat_obs_gen_old(obs_gen.size());
  {
  size_t i = 0;
  for (size_t r = 0; r < obs_gen.rows(); r++){
    for (size_t c = 0; c < obs_gen.cols(); c++){
      flat_obs_gen_old[i] = obs_gen(r,c);
      i++;
    }
  }
  }
  if constexpr (internal_timing) toc("omp_sampler:flattern res old");
#endif
#if SHOW_OLD
  if constexpr (internal_timing) tic("omp_sampler:flattern res");
  matrix<FP> flat_obs_gen(obs_gen.size());
  const size_t og_rows = obs_gen.rows(), og_cols = obs_gen.cols();
  #pragma omp parallel for
  for (size_t r = 0; r < og_rows; r++){
    for (size_t c = 0; c < og_cols; c++){
      const size_t i = r * og_cols+c;
      flat_obs_gen[i] = obs_gen[i];
    }
  }
  if constexpr (internal_timing) toc("omp_sampler:flattern res");
#endif
  if constexpr (internal_timing_enabled) tic("reshape");
  obs_gen.reshape(obs_gen.size(),1,1);
  if constexpr (internal_timing_enabled) toc("reshape");
#if SHOW_OLD
  flat_obs_gen.force_assert_equal(flat_obs_gen_old);
  flat_obs_gen.force_assert_equal(obs_gen);
#endif
  return(obs_gen);
}

//# Generate events, based on the provided x / Q2 grid, the acceptance matrix, weight tensors and grid index tensors:
//***************************
matrix<FP> omp_sampler::gen_events(const matrix<FP>& x_bins,
                                   const matrix<FP>& xsec_x,
                                   const matrix<FP>& q2_bins,
                                   const matrix<FP>& xsec_q2,
                                   const matrix<unsigned short>& acceptance,
                                   const matrix<FP>& weight_vector,
                                   const matrix<size_t>& grid_index,
                                   const size_t max_n){
  //x_gen = self.generate_single_observable(x_bins,xsec_x,acceptance_matrix,weight_tensor,grid_index_tensor,n_max,'x')
  //Q2_gen = self.generate_single_observable(Q2_bins,xsec_Q2,acceptance_matrix,weight_tensor,grid_index_tensor,n_max,'q2')
  matrix<FP> x_gen = this->generate_single_observable(x_bins,xsec_x,acceptance,weight_vector,grid_index,max_n,0);
  matrix<FP> q2_gen = this->generate_single_observable(q2_bins,xsec_q2,acceptance,weight_vector,grid_index,max_n,1);
  assert(x_gen.rows() > 0);
  assert(q2_gen.rows() > 0);

  //return torch.cat([x_gen,Q2_gen],dim=1)
  //**PLUS** filter out interesting (non-zeros) pairs:
  //cond = (evts != 0.0)        
  //events = evts*cond
#if SHOW_OLD
  if constexpr (internal_timing_enabled) tic("cpp_sampler:filter");
  std::vector<FP> temp;
  size_t total = 0;
  for(size_t r = 0; r < x_gen.rows(); r++){
      if(x_gen(r) > 0.0 && q2_gen(r) > 0.0){
      //res(r,0) = x_gen(r);
      //res(r,1) = q2_gen(r);
      temp.push_back(x_gen(r));
      temp.push_back(q2_gen(r));
      total++;
    }
  }
  matrix<FP> res(temp, total, 2);
  if constexpr (internal_timing_enabled) toc("cpp_sampler:filter");
#else
  if constexpr (internal_timing_enabled) tic("filter");
  size_t n = x_gen.rows();
  std::vector<size_t> mask(n);
  std::vector<size_t> prefix(n);

  //for (size_t i = 0; i < n; i++) if (q2_gen.ptr()[i] != 0.0) std::cout << i << " : " << q2_gen.ptr()[i] << " " << std::endl;
  //raise(SIGINT);

  // Step 1: Build mask in parallel
  #pragma omp parallel for
  for (size_t i = 0; i < n; ++i)
    mask[i] = (x_gen(i) > 0.0 && q2_gen(i) > 0.0);

  //for (size_t i = 0; i < n; i++) if (mask[i] != 0) std::cout << i << " : " << mask[i] << " " << std::endl;
  //raise(SIGINT);

  // Step 2: Parallel prefix sum (exclusive scan) of mask
  ///size_t num_threads;
  //std::vector<size_t> thread_sums;
  size_t num_threads = omp_get_max_threads();
  std::vector<size_t> thread_sums(num_threads, 0.0);

  #pragma omp parallel
  {
    size_t tid = omp_get_thread_num();
    size_t local_sum = 0;
    size_t start = (n * tid) / num_threads;
    size_t end   = (n * (tid + 1)) / num_threads;

    for (size_t i = start; i < end; ++i) {
      prefix[i] = local_sum;
      local_sum += mask[i];
    }

    thread_sums[tid] = local_sum;
    #pragma omp barrier
    size_t offset = 0;
    for (size_t t = 0; t < tid; ++t)
      offset += thread_sums[t];

    for (size_t i = start; i < end; ++i)
      prefix[i] += offset;
  }
  size_t new_size = prefix[n - 1] + mask[n - 1];
  std::vector<FP> output(new_size*2);

  // Step 3: Parallel scatter to output
  #pragma omp parallel for
  for (size_t i = 0; i < n; ++i) {
    if (mask[i]) {
      output[prefix[i]*2] = x_gen[i];
      output[prefix[i]*2+1] = q2_gen[i];
    }
  }
  matrix<FP> res(output, new_size, 2);
  if constexpr (internal_timing_enabled) toc("filter");
#endif
  //print_total_measured_time();
  //reset_total_measured_time();
  return (res);
}
//***************************


//# Define a forward pass:
//***************************
// Forward pass for a single parameter sample:
matrix<FP> omp_sampler::forward_single_sample(const matrix<FP>& x_bins,
                                              const matrix<FP>& x_sec_x,
                                              const matrix<FP>& q2_bins,
                                              const matrix<FP>& xsec_q2,
                                              const matrix<unsigned short>& acceptance,
                                              const matrix<FP>& weight_vector,
                                              const matrix<size_t>& grid_idx,
                                              size_t max_n){

  /*
  # Generate events:
  evts = self.gen_events(x_bins,xsec_x,q2_bins,xsec_q2,acceptance,weight_vector,grid_idx,max_n)

  # Make sure that everything is well defined:
  cond = (evts[:,0] > 0.0) & (evts[:,1] > 0.0)
  events = evts[cond]
            
  # Detach everything that we do not need --> Free the GPU memory:
  del cond
  del evts
  # Free cache on the device we are using 
  self.empty_cache()
               
  return events

  */

  x_bins.assert_shape(19,10,1);
  //x_bins increases by 0.0058
  x_bins.assert_leading_values(std::vector<FP>{0.0010, 0.0068, 0.0127, 0.0185, 0.0243,
      0.0302, 0.0360, 0.0419, 0.0477, 0.0535});
  x_bins.assert_ending_values(std::vector<FP>{0.9465, 0.9523, 0.9581, 0.9640, 0.9698,
      0.9756, 0.9815, 0.9873, 0.9932, 0.9990});
  matrix<FP> evts = this->gen_events(x_bins,x_sec_x,q2_bins,xsec_q2,acceptance,weight_vector,grid_idx,max_n);
  return (evts);
}
//***************************

void omp_sampler::test_matrix_implementation(){
  //an even simpler test!
  print_magenta("Test #0 populate a 2x3 tensor 1..6.\n");
  print_blue("Reference Implementation:\n\
  > tmp = torch.tensor([[1,2,3],[4,5,6]])\n\
  >>> tensor([[1, 2, 3],\n\
              [4, 5, 6]])\n\
  > torch.vmap(lambda s: sum(s),in_dims=0)(tmp)\n\
  >>> tensor([ 6, 15])\n\
  >>> ipdb> torch.vmap(lambda s: sum(s),in_dims=1)(tmp)\n\
  >>> tensor([5, 7, 9])");
  matrix<int> tmp(std::vector<int>({1,2,3,4,5,6}),2,3);
  print_magenta("\ntmp = \n"); tmp.print();
  matrix<int> row_sum = tmp.sum(BY_ROW);
  assert(row_sum[0] == 6 && row_sum[1] == 15);
  print_green("\nPassed.\n");
  print_cyan("row sum = \n"); row_sum.print();
  matrix<int> col_sum = tmp.sum(BY_COL);
  print_cyan("col sum = \n"); col_sum.print();
  assert(col_sum[0] == 5 && col_sum[1] == 7 && col_sum[2] == 9);
  print_green("\nPassed.\n");
  
  //Test on just a 3x9 matrix
  print_magenta("Test #1 populate a 3x9 tensor 1..27.\n");
  matrix<int> test_matrix(3,9);
  for(int i = 1; i < 28; i++){
    test_matrix[i-1] = i;
  }
  print_cyan("Test Matrix = \n");
  test_matrix.print();
  print_magenta("Test #1.1 sum along dim=0 (row-wise)\n");
  print_blue("Reference Implementation:\n");
  print_blue("\t> torch.vmap(lambda s: sum(s),in_dims=0)(tmp)\n\
  >>>   tensor([ 45, 126, 207])\n");
  matrix<int> row_summed = test_matrix.sum(0);
  row_summed.print();
  assert(row_summed.as_vector() == std::vector<int>({45,126,207}));
  print_magenta("Test #1.2 sum along dim=1 (column-wise)\n");
  print_blue("\t> torch.vmap(lambda s: sum(s),in_dims=1)(tmp)\n\
  >>>   tensor([30, 33, 36, 39, 42, 45, 48, 51, 54])\n");
  matrix<int> col_summed = test_matrix.sum(1);
  col_summed.print();
  assert(col_summed.as_vector() == std::vector<int>({30, 33, 36, 39, 42, 45, 48, 51, 54}));
  print_green("\nPassed.\n");

  print_magenta("Test #2 populate a 3x3x3 tensor 1..27.\n");
  /*
  //would like to populate the matrix with a single (regular) memory access pattern but currently when pages are specified results in a fragmented view
  for(int i = 1; i < 28; i++){
    full_matrix[i-1] = i;
  }
  */
  size_t rows = 3, cols = 3, pages = 3;
  matrix<int> full_matrix(3,3,3);
  int tmp_value = 1;
  for(size_t r = 0; r < rows; r++){
    for(size_t c = 0; c < cols; c++){
      for(size_t p = 0; p < pages; p++){
        full_matrix(r,c,p) = tmp_value;
        tmp_value++;
      }
    }
  }
  print_blue("Reference Implementation:\n\
  >         data = [[[1, 2, 3], [4, 5, 6], [7, 8, 9]],\n\
                   [[10, 11, 12], [13, 14, 15], [16, 17, 18]],\n\
                   [[19, 20, 21], [22, 23, 24], [25, 26, 27]]]\n\
        tensor = torch.tensor(data)\n\
  >>> tensor = tensor([[[ 1,  2,  3],\n\
                        [ 4,  5,  6],\n\
                        [ 7,  8,  9]],\n\
                       [[10, 11, 12],\n\
                        [13, 14, 15],\n\
                        [16, 17, 18]],\n\
                       [[19, 20, 21],\n\
                        [22, 23, 24],\n\
                        [25, 26, 27]]])\n");
  print_cyan("Test Matrix = \n");
  full_matrix.print();
  print_magenta("Test #2.1 sum along dim=0 (row-wise)\n");
  print_blue("\t> torch.vmap(lambda s: sum(s),in_dims=0)(tensor)\n\
  >>>    tensor([[12, 15, 18],\n\
  >>>           [39, 42, 45],\n\
  >>>           [66, 69, 72]])\n");
  row_sum = full_matrix.sum(0);
  row_sum.print();
  assert(row_sum.rows() == 3 && row_sum.cols() == 3 && row_sum.pages() == 1);
  assert(row_sum.as_vector() == std::vector<int>({12,15,18,39,42,45,66,69,72}));
  print_magenta("Test #2.2 sum along dim=1 (col-wise)\n");
  print_blue("\t> torch.vmap(lambda s: sum(s),in_dims=1)(tensor)\n\
  >>>   tensor([[30, 33, 36],\n\
  >>>           [39, 42, 45],\n\
  >>>           [48, 51, 54]])\n");
  col_sum = full_matrix.sum(1);
  col_sum.print();
  assert(row_sum.rows() == 3 && row_sum.cols() == 3 && row_sum.pages() == 1);
  assert(col_sum.as_vector() == std::vector<int>({30, 33, 36, 39, 42, 45, 48, 51, 54}));

  print_magenta("Test #2.3 sum along dim=2 (page-wise)\n");
  print_blue("\t> torch.vmap(lambda s: sum(s),in_dims=2)(tensor)\n\
  >>>   tensor([[30, 39, 48],\n\
  >>>           [33, 42, 51],\n\
  >>>           [36, 45, 54]])\n");
  col_sum = full_matrix.sum(2);
  col_sum.print();
  assert(row_sum.rows() == 3 && row_sum.cols() == 3 && row_sum.pages() == 1);
  assert(col_sum.as_vector() == std::vector<int>({30, 39, 48, 33, 42, 51, 36, 45, 54}));
  
  print_green("\nPassed.\n\n");

  print_magenta("Test #3 populate a 3x4x2 tensor 1..24.\n");
  rows = 3, cols = 4, pages = 2;
  matrix<int> harder_shaped_matrix(rows,cols,pages);

  tmp_value = 1;
  for(size_t r = 0; r < rows; r++){
    for(size_t c = 0; c < cols; c++){
      for(size_t p = 0; p < pages; p++){
        harder_shaped_matrix(r,c,p) = tmp_value;
        tmp_value++;
      }
    }
  }

  print_blue("Reference Implementation:\n\
  >      data = torch.tensor([[[1, 2], [3, 4], [5, 6], [7, 8]],\n\
                              [[9, 10], [11, 12], [13, 14], [15, 16]],\n\
                              [[17, 18], [19, 20], [21, 22], [23, 24]]])\n\
  >      print(\"tensor = {}, shape = {}\".format(data,data.shape))\n\
  >>>    tensor = tensor([[[ 1,  2],\n\
  >>>                      [ 3,  4],\n\
  >>>                      [ 5,  6],\n\
  >>>                      [ 7,  8]],\n\
  >>>                     [[ 9, 10],\n\
  >>>                      [11, 12],\n\
  >>>                      [13, 14],\n\
  >>>                      [15, 16]],\n\
  >>>                     [[17, 18],\n\
  >>>                      [19, 20],\n\
  >>>                      [21, 22],\n\
  >>>                      [23, 24]]]), shape = torch.Size([3, 4, 2])\n");

  print_cyan("Test Matrix = ");
  harder_shaped_matrix.print();
  harder_shaped_matrix.assert_values(std::vector<int>({1, 3, 5, 7, 2, 4, 6, 8,
        9, 11, 13, 15, 10, 12, 14, 16, 17, 19, 21, 23, 18, 20, 22, 24}));

  print_cyan("The values represented as a vector:\n");
  harder_shaped_matrix.print_as_vector();
  print_magenta("Test #3.1 sum along dim=0 (row-wise)\n");
  print_blue("\
  >      sum0 = torch.vmap(lambda s: sum(s),in_dims=0)(data)\n\
  >      print(\"sum along dim 0 = {}, shape = {}\".format(sum0,sum0.shape))\n\
  >>>    sum along dim 0 = tensor([[16, 20],\n\
  >>>                              [48, 52],\n\
  >>>                              [80, 84]]), shape = torch.Size([3, 2])\n");
  row_sum = harder_shaped_matrix.sum(0);
  row_sum.print();
  assert(row_sum.as_vector()==std::vector<int>({16, 20, 48, 52, 80, 84}));

  print_magenta("Test #3.2 sum along dim=1 (column-wise)\n");
  print_blue("\
  >      sum1 = torch.vmap(lambda s: sum(s),in_dims=1)(data)\n\
  >      print(\"sum along dim 1 = {}, shape = {}\".format(sum1,sum1.shape))\n\
  >>>    sum along dim 1 = tensor([[27, 30],\n\
  >>>                              [33, 36],\n\
  >>>                              [39, 42],\n\
  >>>                              [45, 48]]), shape = torch.Size([4, 2])\n");
  col_sum = harder_shaped_matrix.sum(1);
  col_sum.print();
  assert(col_sum.as_vector()==std::vector<int>({27, 30, 33, 36, 39, 42, 45, 48}));

  print_magenta("Test #3.3 sum along dim=2 (page-wise)\n");
  print_blue("\
  >      sum2 = torch.vmap(lambda s: sum(s),in_dims=2)(data)\n\
  >      print(\"sum along dim 2 = {}, shape = {}\".format(sum2,sum2.shape))\n\
  >>>    sum along dim 2 = tensor([[27, 33, 39, 45],\n\
  >>>                              [30, 36, 42, 48]]), shape = torch.Size([2, 4])\n");
  col_sum = harder_shaped_matrix.sum(2);
  col_sum.print();
  assert(col_sum.as_vector() == std::vector<int>({27, 33, 39, 45, 30, 36, 42, 48}));


  print_magenta("Test #3.4 sum along all dims\n");
  print_blue("Reference Implementation:\n");
  print_blue("\
  \t> test_matrix = torch.tensor([[[0,1,2,3],[4,5,6,7],[8,9,10,11]],\n\
        [[12,13,14,15],[16,17,18,19],[20,21,22,23]]])\n\
      test_matrix.shape\n\
  >>> torch.Size([2,3,4])\n\
  >   torch.vmap(lambda s: sum(s),in_dims=0)(test_matrix)\n\
  >>> tensor([[12, 15, 18, 21],\n\
              [48, 51, 54, 57]])\n\
  >   torch.vmap(lambda s: sum(s),in_dims=1)(test_matrix)\n\
  >>> tensor([[12, 14, 16, 18],\n\
              [20, 22, 24, 26],\n\
              [28, 30, 32, 34]])\n\
  >   torch.vmap(lambda s: sum(s),in_dims=2)(test_matrix)\n\
  >>> tensor([[12, 20, 28],\n\
              [14, 22, 30],\n\
              [16, 24, 32],\n\
              [18, 26, 34]])\
      ");
  rows = 2, cols = 3, pages = 4;
  matrix<int> tm(rows,cols,pages);

  tmp_value = 0;
  for(size_t r = 0; r < rows; r++){
    for(size_t c = 0; c < cols; c++){
      for(size_t p = 0; p < pages; p++){
        tm(r,c,p) = tmp_value;
        tmp_value++;
      }
    }
  }

  tm.sum(BY_ROW).assert_values(std::vector<int>({12,15,18,21,48,51,54,57}));
  tm.sum(BY_COL).assert_values(std::vector<int>({12,14,16,18,20,22,24,26,28,30,32,34}));
  tm.sum(BY_PAGE).assert_values(std::vector<int>({12,20,28,14,22,30,16,24,32,18,26,34}));

  print_magenta("Test #4 slice and replace along all dims\n");
  print_magenta("Test #4.1 slice row-wise\n");
  print_blue("Reference Implementation:\n");
  print_blue("\
  >   print test_matrix[0]\n\
  >>> tensor([[ 0,  1,  2,  3],\n\
              [ 4,  5,  6,  7],\n\
              [ 8,  9, 10, 11]])\n\
  ");
  matrix<int> row_slice = tm.slice(BY_ROW,0);
  row_slice.print();
  row_slice.assert_values(std::vector<int>({0,1,2,3,4,5,6,7,8,9,10,11}));

  print_magenta("Test #4.2 replace row-wise\n");
  matrix<int> replaced_matrix; replaced_matrix.explicit_copy(tm);
  print_blue("Reference Implementation:\n");
  print_blue("\
  >   replaced_matrix = test_matrix\n\
  >   replaced_matrix[0] = torch.tensor([[-1,-2,-3,-4],[-5,-6,-7,-8],[-9,-10,-11,-12]])\n\
  >   print replaced_matrix\n\
  >>> tensor([[[ -1,  -2,  -3,  -4],\n\
               [ -5,  -6,  -7,  -8],\n\
               [ -9, -10, -11, -12]],\n\
               \n\
              [[ 12,  13,  14,  15],\n\
               [ 16,  17,  18,  19],\n\
               [ 20,  21,  22,  23]]])\n\
  ");
  replaced_matrix.replace(BY_ROW,0,std::vector<int>({-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12}));
  replaced_matrix.print();
  replaced_matrix.assert_values(std::vector<int>({-1, -5, -9, -2, -6, -10, -3, -7, -11, -4, -8,
          -12, 12, 16, 20, 13, 17, 21, 14, 18, 22, 15, 19, 23}));

  print_magenta("Test #4.3 slice column-wise\n");
  print_blue("Reference Implementation:\n");
  print_blue("\
  >    print test_matrix[:,0]\n\
  >>>  tensor([[ 0,  1,  2,  3],\n\
               [12, 13, 14, 15]])\n\
  ");
  matrix<int> col_slice = tm.slice(BY_COL,0);
  col_slice.print();
  col_slice.assert_values(std::vector<int>({0,1,2,3,12,13,14,15}));

  print_magenta("Test #4.4 replace column-wise\n");
  print_blue("Reference Implementation:\n");
  print_blue("\
  >    replaced_matrix[:,2] = torch.tensor([[100,101,102,103],[104,105,106,107]])\n\
  >    print replaced_matrix\n\
  >>>  tensor([[[ -1,  -2,  -3,  -4],\n\
                [ -5,  -6,  -7,  -8],\n\
                [100, 101, 102, 103]],\n\
                \n\
               [[ 12,  13,  14,  15],\n\
                [ 16,  17,  18,  19],\n\
                [104, 105, 106, 107]]])\n\
  ");
  replaced_matrix.replace(BY_COL,2,std::vector<int>({100,101,102,103,104,105,106,107}));
  replaced_matrix.print();
  replaced_matrix.assert_values(std::vector<int>({-1, -5, 100, -2, -6, 101, -3, -7, 102, -4, -8,
          103, 12, 16, 104, 13, 17, 105, 14, 18, 106, 15, 19, 107}));

  print_magenta("Test #4.5 slice page-wise\n");
  print_blue("Reference Implementation:\n");
  print_blue("\
  > print test_matrix[:,:,0]\n\
  >>>    tensor([[ 0,  4,  8],\n\
               [12, 16, 20]])\n\
  ");
  matrix<int> page_slice = tm.slice(BY_PAGE,0);
  page_slice.print();
  page_slice.assert_values(std::vector<int>({0,4,8,12,16,20}));

  print_magenta("Test #4.6 replace page-wise\n");
  print_blue("Reference Implementation:\n");
  print_blue("\
  >    replaced_matrix[:,:,1] = torch.tensor([[0,0,0],[0,0,0]])\n\
  >    print replaced_matrix\n\
  >>>  tensor([[[ -1,   0,  -3,  -4],\n\
                [ -5,   0,  -7,  -8],\n\
                [100,   0, 102, 103]],\n\
                \n\
               [[ 12,   0,  14,  15],\n\
                [ 16,   0,  18,  19],\n\
                [104,   0, 106, 107]]])\n\
  ");
  replaced_matrix.replace(BY_PAGE,1,std::vector<int>({0,0,0,0,0,0}));
  replaced_matrix.print();
  replaced_matrix.assert_values(std::vector<int>({-1, -5, 100, 0, 0, 0, -3, -7, 102, -4, -8, 103,
          12, 16, 104, 0, 0, 0, 14, 18, 106, 15, 19, 107}));

  print_green("\nPassed.\n\n");
}

void omp_sampler::test_trapezoid_implementation(){
  std::vector<FP> y{1.0f, 5.0f, 10.0f};
  matrix<FP> my(y.data(),3,1);
  my.print();
  matrix<FP> res=this->trapezoid(my);
  print_blue("Reference Implementation:\n");
  printf(">>> # Computes the trapezoidal rule in 1D, spacing is implicitly 1\n");
  printf(">>> y = torch.tensor([1, 5, 10])\n>>> torch.trapezoid(y)\n>>> tensor(10.5)\n");
  printf("C++ implementation answer = %f\n",res[0]);
  assert(res[0] == 10.5f);
 
  printf(">>> # Computes the trapezoidal rule in 1D with constant spacing of 2\n");
  printf(">>> # NOTE: the result is the same as before, but multiplied by 2\n");
  printf(">>> torch.trapezoid(y, dx=2)\n>>> 21.0\n");
  res = this->trapezoid(my,2.0f);
  printf("C++ implementation answer = %f\n",res[0]);
  assert(res[0] == 21.f);

  printf(">>> # Computes the trapezoidal rule in 1D with arbitrary spacing\n");
  printf(">>> x = torch.tensor([1, 3, 6])\n");
  printf(">>> torch.trapezoid(y, x)\n");
  printf(">>> 28.5\n");
  std::vector<FP> x{1.0f, 3.0f, 6.0f};
  matrix<FP> mx(x.data(),3,1);
  res = this->trapezoid(my,1.0f,mx);
  printf("C++ implementation answer = %f\n",res[0]);
  assert(res[0] == 28.5f);

  printf(">>> # Computes the trapezoidal rule for each row of a 3x3 matrix\n");
  printf(">>> y = torch.arange(9).reshape(3, 3)\n");
  printf("tensor([[0, 1, 2],\n");
  printf("        [3, 4, 5],\n");
  printf("        [6, 7, 8]])\n");
  printf(">>> torch.trapezoid(y)\n");
  printf("tensor([ 2., 8., 14.])\n");
  std::vector<FP> z{0.,1.,2.,3.,4.,5.,6.,7.,8.};
  matrix<FP> mz(z,3,3);
  res = this->trapezoid(mz,1.f,matrix<FP>(),BY_COL);
  std::cout << "C++ implementation answer = " << res << std::endl;
  auto ref = std::vector<FP> {2.0f,8.0f,14.0f};
  assert (res.as_vector() == ref);

  printf(">>> # Computes the trapezoidal rule for each column of the matrix\n");
  printf(">>> torch.trapezoid(y, dim=0)\n");
  printf("tensor([ 6., 8., 10.])\n");
  res = this->trapezoid(mz,1.f,matrix<FP>(),BY_ROW);
  std::cout<< "C++ implementation answer = " << res << std::endl;
  assert((res.as_vector() == std::vector<FP> {6.0f,8.0f,10.0f}));

  printf(">>> # Computes the trapezoidal rule for each row of a 3x3 ones matrix\n");
  printf(">>> #   with the same arbitrary spacing\n");
  printf(">>> y = torch.ones(3, 3)\n");
  printf(">>> x = torch.tensor([1, 3, 6])\n");
  printf(">>> torch.trapezoid(y, x)\n");
  printf("array([5., 5., 5.])\n");
  matrix<FP> ma(3,3);
  ma.fill(1.0f);
  matrix<FP> mb(std::vector<FP>({1.f,3.f,6.f}),3,1);
  res = this->trapezoid(ma,1.f,mb);

  std::cout << "C++ implementation answer = " << res << std::endl;
  assert((res.as_vector() == std::vector<FP> {5.0f,5.0f,5.0f}));

  printf(">>> # Computes the trapezoidal rule for each row of a 3x3 ones matrix\n");
  printf(">>> #   with different arbitrary spacing per row\n");
  printf(">>> y = torch.ones(3, 3)\n");
  printf(">>> x = torch.tensor([[1, 2, 3], [1, 3, 5], [1, 4, 7]])\n");
  printf(">>> torch.trapezoid(y, x)\n");
  printf("array([2., 4., 6.])\n");
  matrix<FP> mc(std::vector<FP>({1.f,2.f,3.f,1.f,3.f,5.f,1.f,4.f,7.f}),3,3);
  res = this->trapezoid(ma,1.f,mc,BY_COL);
  std::cout<< "C++ implementation answer = " << res << std::endl;
  assert((res.as_vector() == std::vector<FP> {2.0f,4.0f,6.0f}));

  printf(">>> # Computes the trapezoidal rule for each column of a 3x3 ones matrix\n");
  printf(">>> #   with the same arbitrary spacing\n");
  printf(">>> y = torch.ones(3, 3)\n");
  printf(">>> x = torch.tensor([1, 3, 6])\n");
  printf(">>> torch.trapezoid(y, x, dim=0)\n");
  printf("array([5., 5., 5.])\n");
  matrix<FP> md(3,3);
  md.fill(1.0f);
  matrix<FP> me(std::vector<FP>({1.f,3.f,6.f}),3,1);
  res = this->trapezoid(md,1.f,me,BY_ROW);
  std::cout << "C++ implementation answer = " << res << std::endl;
  assert((res.as_vector() == std::vector<FP> {5.0f,5.0f,5.0f}));

  printf(">>> # Computes the trapezoidal rule for each column of a 3x3 ones matrix\n");
  printf(">>> #   with different arbitrary spacing per row\n");
  printf(">>> y = torch.ones(3, 3)\n");
  printf(">>> x = torch.tensor([[1, 2, 3], [1, 3, 5], [1, 4, 7]])\n");
  printf(">>> torch.trapezoid(y, x, dim=0)\n");
  printf("array([0., 2., 4.])\n");
  matrix<FP> mf(std::vector<FP>({1.f,2.f,3.f,1.f,3.f,5.f,1.f,4.f,7.f}),3,3);
  res = this->trapezoid(md,1.f,mf,BY_ROW);
  std::cout<< "C++ implementation answer = " << res << std::endl;
  assert((res.as_vector() == std::vector<FP> {0.0f,2.0f,4.0f}));

  printf("Congratulations trapezoidal function passes all tests!\n");
}

void omp_sampler::test_functions(){
  this->test_matrix_implementation();
  this->test_trapezoid_implementation();
}
