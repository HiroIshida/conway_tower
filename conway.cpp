#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Core>
#include <iostream>
#include <optional>

namespace py = pybind11;

py::array_t<bool> evolve_conway(py::array_t<bool> input, size_t T, std::optional<Eigen::MatrixX2i> perturbs) {
    py::buffer_info buf = input.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Number of dimensions must be two");
    }
    size_t nx = buf.shape[0];
    size_t ny = buf.shape[1];
    size_t nt = T;
    std::vector<size_t> shape = {nt, nx, ny};
    std::vector<size_t> strides = {nx * ny * sizeof(bool), ny * sizeof(bool), sizeof(bool)};
    py::array_t<bool> output(shape, strides);

    bool *input_ptr = (bool *) input.request().ptr;
    bool *output_ptr = (bool *) output.request().ptr;

    py::array_t<bool> tmp({nx, ny});
    auto tmp_ptr = (bool *) tmp.request().ptr;
    auto tmp_accessor = tmp.mutable_unchecked<2>();

    memcpy(output_ptr, input_ptr, nx * ny * sizeof(bool)); // copy initial state
    auto voxels = output.mutable_unchecked<3>();
    for(size_t t = 0; t < nt - 1; ++t) { // evolve t times
      if(perturbs) {
        std::fill(tmp_ptr, tmp_ptr + nx * ny, false);
        auto delta_x = perturbs->coeff(t, 0);
        auto delta_y = perturbs->coeff(t, 1);
        for(size_t i = 0; i < nx; ++i) {
          for(size_t j = 0; j < ny; ++j) {
            // TODO: can we do this in place?
            tmp_accessor(i, j) = voxels(t, (i - delta_x) % nx, (j - delta_y) % ny);
          }
        }
        size_t offset = t * nx * ny;
        memcpy(output_ptr + offset, tmp_ptr, nx * ny * sizeof(bool));
      }
      for(size_t i = 0; i < nx; ++i) {
        for(size_t j = 0; j < ny; ++j) {
          size_t nearest_alive_count = 0;
          voxels(t, (i-1) % nx, (j-1) % ny) ? nearest_alive_count++ : 0;
          voxels(t, (i-1) % nx, j) ? nearest_alive_count++ : 0;
          voxels(t, (i-1) % nx, (j+1) % ny) ? nearest_alive_count++ : 0;
          voxels(t, i, (j-1) % ny) ? nearest_alive_count++ : 0;
          voxels(t, i, (j+1) % ny) ? nearest_alive_count++ : 0;
          voxels(t, (i+1) % nx, (j-1) % ny) ? nearest_alive_count++ : 0;
          voxels(t, (i+1) % nx, j) ? nearest_alive_count++ : 0;
          voxels(t, (i+1) % nx, (j+1) % ny) ? nearest_alive_count++ : 0;
          if(voxels(t, i, j)) {
            voxels(t+1, i, j) = nearest_alive_count == 2 || nearest_alive_count == 3;
          } else {
            voxels(t+1, i, j) = nearest_alive_count == 3;
          }
        }
      }
    }
    return output;
}

PYBIND11_MODULE(_conway_tower, m) {
  m.def("evolve_conway", &evolve_conway, "Evolve Conway's game of life", py::arg("input"), py::arg("T"), py::arg("perturbs") = std::nullopt);
}
