#ifndef MATRIX 
#define MATRIX

#include <vector>
#include <iostream>
#include <memory>

#define BY_ROW  0
#define BY_COL  1
#define BY_PAGE 2
#define BY_ROW_TRANSPOSE 3

template <class T>
class matrix {
  private:
    std::size_t _rows;
    std::size_t _cols;
    std::size_t _page;
    std::size_t _nbytes; //the size of the matrix in bytes
    std::size_t _num_elements; //the number of elements in the matrix
    std::vector<T> _data;
    struct impl;                    // incomplete
    std::unique_ptr<impl> p_;        // fixed-size ABI

  public:
    T* dev_ptr();
    const T* dev_ptr() const;
    void to_dev();
    void to_host();
    matrix();
    matrix(std::size_t rows, std::size_t cols = 1, std::size_t pages = 1, void* queue = nullptr);
    matrix(T* ptr, std::size_t rows, std::size_t cols, std::size_t pages = 1);
    matrix(std::vector<T> vec, size_t rows, size_t cols, size_t pages = 1); //expensive (uses copy)
    ~matrix();

    matrix(const matrix&) = delete;
    matrix& operator=(const matrix&) = delete;

    matrix(matrix&& other) noexcept;
    matrix& operator=(matrix&& other) noexcept;

    void explicit_copy(const matrix &mat, void* queue = nullptr);

    void fill_rand();
    void fill(T val);
    std::vector<std::size_t> shape();
    void print_shape() const;
    void print_as_vector() const;
    
    T& operator[] (unsigned long index);
    const T operator[] (size_t index) const;
    T operator() (size_t row, size_t col=0, size_t page=0) const;
    T& operator() (size_t row, size_t col=0, size_t page=0);
    T* ptr() {return _data.data();}
    const T* ptr() const {return _data.data();}

    void assert_shape(std::vector<size_t> shape) const;
    void assert_shape(size_t row, size_t col, size_t page) const;
    void assert_values(matrix<T> values) const;
    void assert_values(std::vector<T> values) const;
    void assert_leading_values(std::vector<T> values) const;
    void assert_ending_values(std::vector<T> values) const;
    void force_assert_equal(matrix<T>& test, bool print = false);

    std::vector<T> as_vector() const {return _data;}
    std::vector<T> as_vector() {return _data;}
    void print();
    matrix<T> sum(size_t dimension=0);
    matrix<T> transpose();
    //matrix<T> flip();

    friend std::ostream& operator<<(std::ostream& os, const matrix<T>& m){
      os << "[";
      for (size_t h = 0; h < m.rows(); h++){
        os << "[ ";
        for (size_t w = 0; w < m.cols(); w++){
          os << m[h*m.cols()+w] << " ";
        }
        os << "]";
      }
      os << "]";
      return os;
    }
    matrix<T> slice(std::size_t dimension, std::size_t index) const;
    void replace(std::size_t dimension, std::size_t index, std::vector<T> values);
    void reshape(size_t rows, size_t cols, size_t pages);
    const std::size_t& cols() const {return _cols;}
    const std::size_t& rows() const {return _rows;}
    const std::size_t& pages() const {return _page;}
    const std::size_t& bytes() const {return _nbytes;}
    const std::size_t& size() const {return _num_elements;}
    typename std::vector<T>::iterator begin() {return _data.begin();}
    typename std::vector<T>::iterator end() {return _data.end();}
};
#endif
