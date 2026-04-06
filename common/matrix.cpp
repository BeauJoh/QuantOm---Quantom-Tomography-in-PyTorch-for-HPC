#include "matrix.hpp"
#include "utils.hpp"

#include <iostream>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <ranges>
#include <vector>
#include <random>

#ifndef WRAPPER_BUILD
#include <csignal>
#endif

#if SYCL_BUILD
#include <sycl/sycl.hpp>
#endif

using namespace std;

template <class T>
struct matrix<T>::impl {
#if SYCL_BUILD
    std::shared_ptr<sycl::queue> _queue; // shared ownership
    sycl::context _ctx;
    sycl::device  _dev;
    T* _device_data = nullptr;
    std::size_t _num_elements;

    void create_memory_on_device(void* queue, size_t num_elements){
      if (queue != nullptr){
        auto q = *static_cast<std::shared_ptr<sycl::queue>*>(queue);
        _queue = q;                 // share ownership
        _ctx = _queue->get_context();
        _dev = _queue->get_device();
        _num_elements = num_elements;
        _device_data = sycl::malloc_device<T>(_num_elements, _dev, _ctx);
        _queue->memset(_device_data, 0, sizeof(T) * _num_elements);
        _queue->wait();
      }
    }
    void move(matrix<T>::impl other){
      if (!other._queue) return;
      _queue = std::move(other._queue);
      _ctx = std::move(other._ctx);
      _dev = std::move(other._dev);
      _device_data = other._device_data;
      other._device_data = nullptr;
    }
    void deep_copy(matrix<T>::impl other){
      if (!other._queue) return;
      _queue = other._queue;
      _ctx = other._ctx;
      _dev = other._dev;
      // Allocate new USM memory
      if (other._device_data) {
        _device_data = sycl::malloc_device<T>(_num_elements, _dev, _ctx);
        _queue->wait();
        // Copy device data
        std::copy(other._device_data,
            other._device_data + _num_elements,
            _device_data);
        _queue->wait();
      }
    }
    void release() {// Release currently owned USM memory
      if (_device_data) {
        sycl::free(_device_data, _ctx);
        _device_data = nullptr;
      }
    }
    T* dev_ptr() { return _device_data; }
    const T* dev_ptr() const { return _device_data; }
    void to_dev(T* host_data, size_t nbytes){
      _queue->memcpy(_device_data, host_data, nbytes);
    };
    void to_host(T* host_data, size_t nbytes){
      _queue->memcpy(host_data, _device_data, nbytes);
    };
#else
  void create_memory_on_device(void* queue, size_t num_elements){}
  void move(matrix<T>::impl other){}
  void deep_copy(matrix<T>::impl other){}
  void release() {}
  T* dev_ptr() { return nullptr; };
  const T* dev_ptr() const { return nullptr; };
  void to_dev(T* host_ptr, size_t nbytes){};
  void to_host(T* host_ptr, size_t nbytes){};
#endif
};

template <class T>
matrix<T>::matrix(){
  _num_elements = 0;
  _data = vector<T>();
  _cols = 0;
  _rows = 0;
  _page = 0;
  _nbytes = 0;
}
template <class T>
matrix<T>::matrix(size_t rows, size_t cols, size_t page, void* queue){
  _num_elements = cols*rows*page;
  _cols = cols;
  _rows = rows;
  _page = page;
  _nbytes = sizeof(T)*cols*rows*page;
  _data = vector<T>(_num_elements);
  p_ = std::make_unique<impl>();
  p_->create_memory_on_device(queue, _num_elements);
}

template <class T>
T* matrix<T>::dev_ptr(){
  return(p_->dev_ptr());
}
template <class T>
const T* matrix<T>::dev_ptr() const{
  return(p_->dev_ptr());
}
template <class T>
void matrix<T>::to_dev(){
  return(p_->to_dev(_data.data(),_nbytes));
}
template <class T>
void matrix<T>::to_host(){
  return(p_->to_host(_data.data(),_nbytes));
}
template <class T>
matrix<T>::matrix(T* ptr, size_t rows, size_t cols, size_t page){
  _num_elements = cols*rows*page;
  _data = vector<T>(ptr,ptr+(_num_elements));
  _cols = cols;
  _rows = rows;
  _page = page;
  _nbytes = sizeof(T)*cols*rows*page;
  p_ = std::make_unique<impl>();
}

template <class T>
matrix<T>::~matrix<T>(){
  _data.clear();
  _num_elements = 0;
  _cols = 0;
  _rows = 0;
  _page = 0;
  _nbytes = 0;
  if (p_) p_->release();
}

template <class T>
matrix<T>::matrix(matrix&& other) noexcept
{
  _num_elements = other._num_elements;
  _cols = other._cols;
  _rows = other._rows;
  _page = other._page;
  _nbytes = other._nbytes;
  _data = std::move(other._data);
  if (other.p_) p_->move(*other.p_);
}

template <class T>
matrix<T>& matrix<T>::operator=(matrix&& other) noexcept
{
  if (this == &other)
    return *this;
  _num_elements = other._num_elements;
  _cols = other._cols;
  _rows = other._rows;
  _page = other._page;
  _nbytes = other._nbytes;
  _data = std::move(other._data);
  if (p_) p_->release();
  if (other.p_) p_->move(*other.p_);

  return *this;
}

template <class T>
void matrix<T>::explicit_copy(const matrix &other, void* queue){
  _num_elements = other._num_elements;
  _cols = other._cols;
  _rows = other._rows;
  _page = other._page;
  _nbytes = other._nbytes;
  _data = other._data;
  if(queue){
    p_ = std::make_unique<impl>();
    p_->create_memory_on_device(queue, _num_elements);
  }
  if(other.p_) p_->deep_copy(*other.p_);
}

template <class T>
void matrix<T>::fill(T val){
  for (size_t i = 0; i < _num_elements; i++){
    _data[i] = val;
  }
}

template <class T>
void matrix<T>::fill_rand(){
  std::random_device rd;
  std::mt19937 e2(rd());
  std::uniform_real_distribution<> dist(0.0, 1.0);
  for (size_t i = 0; i < _num_elements; i++){
    _data[i] = dist(e2);
  }
}

template <class T>
matrix<T>::matrix(vector<T> vec, size_t rows, size_t cols, size_t page){
  _num_elements = cols*rows*page;
  _data = vec;
  _cols = cols;
  _rows = rows;
  _page = page;
  _nbytes = sizeof(T)*cols*rows*page;
}


template <class T>
T& matrix<T>::operator[] (size_t index) {
  return _data[index];
}

template <class T>
const T matrix<T>::operator[] (size_t index) const{
  return _data[index];
}


template <class T>
T matrix<T>::operator() (size_t r, size_t c, size_t p) const{
  return _data[r*_cols*_page + p*_cols + c];
  //return _data[(r*_cols + c)+(p*_cols*_rows)];
}

template <class T>
T& matrix<T>::operator() (size_t r, size_t c, size_t p){
  return _data[r*_cols*_page + p*_cols + c];
  //return _data[(r*_cols + c)+(p*_cols*_rows)];
}

template <class T>
void matrix<T>::print(){
  //globally set the precision to 5 decimal places
  std::cout.unsetf(std::ios::floatfield); std::cout << std::setprecision(3);// << std::cout.width(4);
  size_t max_digits_to_pad = std::to_string(*max_element(_data.begin(),_data.end())).length();
  if(*min_element(_data.begin(),_data.end()) < 0) max_digits_to_pad++; // negative values need an extra padding white-space

  cout << endl;
  if (_rows > 1 && _cols > 1 && _page > 1){//3-D
    cout << "[";
    for (size_t r = 0; r < _rows; r++){
      cout << "[";
      for (size_t c = 0; c < _cols; c++){
        cout << "[";
        for (size_t p = 0; p < _page; p++){
          if (p != 0) cout << " ";
          auto val = this->operator()(r,c,p);//_data[(p*_cols*_rows)+(r*_cols+c)];
          size_t val_len = std::to_string(val).length();
          while(val_len < max_digits_to_pad){
            cout << " "; val_len++;
          }
          cout << val;
          if (p != _page-1) cout << ",";
        }
        cout << "]";
        if (c != _cols-1) cout << "," << endl << "  ";
      }
      cout << "]";
      if (r != _rows-1) cout << "," << endl << " ";
    }
    cout << "]" << endl;
  }
  else if (_rows > 1 && _cols > 1) {//2-D
    cout << "[";
    for (size_t r = 0; r < _rows; r++){
      cout << "[";
      for (size_t c = 0; c < _cols; c++){
          if (c != 0) cout << " ";
          auto val = this->operator()(r,c,0);
          size_t val_len = std::to_string(val).length();
          while(val_len < max_digits_to_pad){
            cout << " "; val_len++;
          }
          cout << val;
          if (c != _cols-1) cout << ",";
        }
        cout << "]";
        if (r != _rows-1) cout << "," << endl << " ";
      }
    cout << "]" << endl;
  }
  else{//1-D
    cout << "[";
    for (size_t r = 0; r < _rows; r++){
      if (r != 0) cout << " ";
      auto val = this->operator()(r,0,0);
      size_t val_len = std::to_string(val).length();
      while(val_len < max_digits_to_pad){
        cout << " "; val_len++;
      }
      cout << val;
      if (r != _rows-1) cout << ",";
    }
    cout << "]" << endl;
  }
  return;
}

template <class T>
std::vector<std::size_t> matrix<T>::shape(){
  return std::vector<std::size_t>({_rows,_cols,_page});
}

template <class T>
void matrix<T>::print_shape() const {
  std::cout << "[" << _rows << ", " << _cols << ", " << _page << "]" << std::endl;
  return;
}

template <class T>
void matrix<T>::print_as_vector() const {
  for (size_t i = 0; i < _num_elements; i++){
    std::cout << _data[i] << " ";
  }
  std::cout << std::endl;

}

template <class T>
matrix<T> matrix<T>::sum(std::size_t dimension){
  if (dimension == BY_ROW){
    matrix<T> result(_rows,_page,1);
    for (size_t r = 0; r < _rows; r++){
      for (size_t p = 0; p < _page; p++){
        T sum = 0;
        for (size_t c = 0; c < _cols; c++){
          sum += this->operator()(r,c,p);
        }
        result(r,p) += sum;
      }
    }
    return(result);
  }
  else if (dimension == BY_COL){
    matrix<T> result(_cols,_page,1);
    for (size_t c = 0; c < _cols; c++){
      for (size_t p = 0; p < _page; p++){
        T sum = 0;
        for (size_t r = 0; r < _rows; r++){
          sum += this->operator()(r,c,p);
        }
        result(c,p) = sum;
      }
    }
    return(result);
  }
  else if (dimension == BY_PAGE){
    matrix<T> result(_page,_cols,1);
    for (size_t p = 0; p < _page; p++){
      for (size_t c = 0; c < _cols; c++){
        T sum = 0;
        for (size_t r = 0; r < _rows; r++){
          sum += this->operator()(r,c,p);
        }
        result(p,c) = sum;
      }
    }
    return(result);
  }
  else{
    assert(false && "not implemented for 3 or more dimensions");
    return(matrix<T>());
  }
}

template <class T>
matrix<T> matrix<T>::transpose(){
  matrix<T> tmat(this->cols(), this->rows(), this->pages());
  for(size_t p = 0; p < this->pages(); p++){
    for(size_t c = 0; c < this->cols(); c++){
      for(size_t r = 0; r < this->rows(); r++){
        tmat(c,r,p) = this->operator()(r,c,p);
      }
    }
  }
  return(tmat);
}


template <class T>
matrix<T> matrix<T>::slice(std::size_t dimension, std::size_t index) const {
  //row-wise
  if (dimension == BY_ROW){
    matrix<T> result(_cols,_page,1);
    size_t r = index;
    for (size_t p = 0; p < _page; p++){
      for (size_t c = 0; c < _cols; c++){
        result(c,p) = this->operator()(r,c,p);
      }
    }
    return(result);
  }
  //column-wise
  else if (dimension == BY_COL) {
    matrix<T> result(_rows,_page,1);
    size_t c = index;
    for (size_t p = 0; p < _page; p++){
      for (size_t r = 0; r < _rows; r++){
        result(r,p) = this->operator()(r,c,p);
      }
    }
    return(result);
  }
  //page-wise
  else if (dimension == BY_PAGE) {
    matrix<T> result(_rows,_cols,1);
    size_t p = index;
    for (size_t c = 0; c < _cols; c++){
      for (size_t r = 0; r < _rows; r++){
        result(r,c) = this->operator()(r,c,p);
      }
    }
    return(result);
  }
  else if (dimension == BY_ROW_TRANSPOSE){
    matrix<T> result(_cols,_page,1);
    size_t r = index;
    for (size_t p = 0; p < _page; p++){
      for (size_t c = 0; c < _cols; c++){
          size_t rindex = r*_cols*_page + p*_cols + c;
          size_t windex = p*_cols + c;
          result[windex] = _data[rindex];
      }
     }
     return(result);
  }

  else{
    assert(false && "not implemented for 3 or more dimensions");
    return(matrix<T>());
  }
}

template <class T>
void matrix<T>::replace(std::size_t dimension, std::size_t index, std::vector<T> values){
  switch(dimension){
    case BY_ROW: {
      assert(values.size() == _cols*_page);
      size_t r = index;
      size_t ctr = 0;
      for (size_t c = 0; c < _cols; c++){
        for (size_t p = 0; p < _page; p++){
          this->operator()(r,c,p) = values[ctr];
          ctr++;
        }
       }
      return;
      break;
    }
    case BY_COL: {
      assert(values.size() == _rows*_page);
      size_t c = index;
      size_t ctr = 0;
      for (size_t r = 0; r < _rows; r++){
        for (size_t p = 0; p < _page; p++){
          this->operator()(r,c,p) = values[ctr];
          ctr++;
        }
       }
      return;
      break;
    }
    case BY_PAGE: {
      assert(values.size() == _rows*_cols);
      size_t p = index;
      size_t ctr  = 0;
      for (size_t c = 0; c < _cols; c++){
        for (size_t r = 0; r < _rows; r++){
            this->operator()(r,c,p) = values[ctr];
            ctr++;
        }
      }
      return;
      break;
    }
    default: {
      assert(false && "Only 3-D tensors have been implemented");
      return;
      break;
    }
  }
}

template <class T>
void matrix<T>::assert_shape(std::vector<size_t> shape) const{
#ifdef WRAPPER_BUILD
  return;
#endif
  assert(shape.size() == 3);
  this->assert_shape(shape[0], shape[1], shape[2]);
}

template <class T>
void matrix<T>::assert_shape(size_t row, size_t col, size_t page) const{
#ifdef WRAPPER_BUILD
  return;
#endif
  assert(_rows == row);
  assert(_cols == col);
  assert(_page == page);
}

template <class T>
void matrix<T>::assert_values(matrix<T> mat) const {
#ifdef WRAPPER_BUILD
  return;
#endif
  this->assert_shape(mat.shape());
  this->assert_values(mat.as_vector());
}

template <class T>
void matrix<T>::assert_values(std::vector<T> values) const {
#ifdef WRAPPER_BUILD
  return;
#endif
  assert(values.size() == this->size());
  for (size_t i = 0; i < values.size(); i++) {
    assert(are_equal(values[i],_data[i]));
  }
}

template <class T>
void matrix<T>::assert_leading_values(std::vector<T> values) const{
#ifdef WRAPPER_BUILD
  return;
#endif
  for (size_t i = 0; i < values.size(); i++) {
    assert(are_equal(values[i],_data[i]));
  }
}

template <class T>
void matrix<T>::assert_ending_values(std::vector<T> values) const {
#ifdef WRAPPER_BUILD
  return;
#endif
  size_t offset = _data.size()-values.size();
  for (size_t i = 0; i < values.size(); i++) {
    assert(are_equal(values[i],_data[offset+i]));
  }
}

template <class T>
void matrix<T>::force_assert_equal(matrix<T>& test, bool print){
  assert(this->size() == test.size());
  for (size_t i = 0; i < test.size(); i++) {
    if (print) std::cout << "i = " << i << " ref = " << this->operator[](i) << " test = " << test[i] << std::endl;
    assert(are_equal(this->operator[](i),test[i]));
  }
}

template <class T>
void matrix<T>::reshape(size_t rows, size_t cols, size_t pages){
  _rows = rows;
  _cols = cols;
  _page = pages;
  _num_elements = rows*cols*pages;
  _nbytes = sizeof(T)*_num_elements;
}

//and instantiate all versions of the matrix class template
template class matrix<unsigned long>;
template class matrix<unsigned short>;
template class matrix<float>;
template class matrix<double>;
template class matrix<int>;
