// Copyright (C) 2017-2020 Chris Richardson, Garth N. Wells and
// Francesco Ballarin
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "array.h"
#include "caster_mpi.h"
#include "caster_petsc.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/la/PETScMatrix.h>
#include <dolfinx/la/PETScVector.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/la/Vector.h>
#include <dolfinx/la/VectorSpaceBasis.h>
#include <dolfinx/la/utils.h>
#include <memory>
#include <petsc4py/petsc4py.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <xtl/xspan.hpp>

namespace py = pybind11;

namespace
{
  template <class T>
  std::array<T, 2> convert_vector_to_array(const std::vector<T>& input)
  {
    // TODO remove this when pybind11#2123 is fixed.
    assert(input.size() == 2);
    std::array<T, 2> output {{input[0], input[1]}};
    return output;
  }
}

namespace dolfinx_wrappers
{

void la(py::module& m)
{
  // dolfinx::la::SparsityPattern
  py::class_<dolfinx::la::SparsityPattern,
             std::shared_ptr<dolfinx::la::SparsityPattern>>(m,
                                                            "SparsityPattern")
      .def(py::init(
          [](const MPICommWrapper comm,
             const std::array<std::shared_ptr<const dolfinx::common::IndexMap>,
                              2>& maps,
             const std::array<int, 2>& bs) {
            return dolfinx::la::SparsityPattern(comm.get(), maps, bs);
          }))
      .def(py::init(
          [](const MPICommWrapper comm,
             const std::vector<std::vector<const dolfinx::la::SparsityPattern*>>
                 patterns,
             const std::array<
                 std::vector<std::pair<
                     std::reference_wrapper<const dolfinx::common::IndexMap>,
                     int>>,
                 2>& maps,
             const std::array<std::vector<int>, 2>& bs) {
            return dolfinx::la::SparsityPattern(comm.get(), patterns, maps, bs);
          }))
      .def("index_map", &dolfinx::la::SparsityPattern::index_map)
      .def("assemble", &dolfinx::la::SparsityPattern::assemble)
      .def("num_nonzeros", &dolfinx::la::SparsityPattern::num_nonzeros)
      .def("insert", &dolfinx::la::SparsityPattern::insert)
      .def("insert_diagonal", &dolfinx::la::SparsityPattern::insert_diagonal)
      .def_property_readonly("diagonal_pattern",
                             &dolfinx::la::SparsityPattern::diagonal_pattern,
                             py::return_value_policy::reference_internal)
      .def_property_readonly(
          "off_diagonal_pattern",
          &dolfinx::la::SparsityPattern::off_diagonal_pattern,
          py::return_value_policy::reference_internal);

  // dolfinx::la::VectorSpaceBasis
  py::class_<dolfinx::la::VectorSpaceBasis,
             std::shared_ptr<dolfinx::la::VectorSpaceBasis>>(m,
                                                             "VectorSpaceBasis")
      .def(py::init([](const std::vector<Vec> x) {
        std::vector<std::shared_ptr<dolfinx::la::PETScVector>> _x;
        for (std::size_t i = 0; i < x.size(); ++i)
        {
          assert(x[i]);
          _x.push_back(std::make_shared<dolfinx::la::PETScVector>(x[i], true));
        }
        return dolfinx::la::VectorSpaceBasis(_x);
      }))
      .def("is_orthonormal", &dolfinx::la::VectorSpaceBasis::is_orthonormal,
           py::arg("tol") = 1.0e-10)
      .def("is_orthogonal", &dolfinx::la::VectorSpaceBasis::is_orthogonal,
           py::arg("tol") = 1.0e-10)
      .def("in_nullspace", &dolfinx::la::VectorSpaceBasis::in_nullspace,
           py::arg("A"), py::arg("tol") = 1.0e-10)
      .def("orthogonalize", &dolfinx::la::VectorSpaceBasis::orthogonalize)
      .def("orthonormalize", &dolfinx::la::VectorSpaceBasis::orthonormalize,
           py::arg("tol") = 1.0e-10)
      .def("dim", &dolfinx::la::VectorSpaceBasis::dim)
      .def("__getitem__", [](const dolfinx::la::VectorSpaceBasis& self, int i) {
        return self[i]->vec();
      });

  // dolfinx::la::Vector
  py::class_<dolfinx::la::Vector<PetscScalar>,
             std::shared_ptr<dolfinx::la::Vector<PetscScalar>>>(m, "Vector")
      .def_property_readonly(
          "array",
          [](dolfinx::la::Vector<PetscScalar>& self) {
            std::vector<PetscScalar>& array = self.mutable_array();
            return py::array(array.size(), array.data(), py::cast(self));
          })
      .def("scatter_forward", &dolfinx::la::Vector<PetscScalar>::scatter_fwd)
      .def("scatter_reverse", &dolfinx::la::Vector<PetscScalar>::scatter_rev);

  // utils
  py::enum_<dolfinx::la::GhostBlockLayout>(m, "GhostBlockLayout")
      .value("intertwined", dolfinx::la::GhostBlockLayout::intertwined)
      .value("trailing", dolfinx::la::GhostBlockLayout::trailing);

  m.def("create_vector",
        py::overload_cast<const dolfinx::common::IndexMap&, int>(
            &dolfinx::la::create_petsc_vector),
        py::return_value_policy::take_ownership,
        "Create a ghosted PETSc Vec for index map.");
  m.def(
      "create_matrix",
      [](const MPICommWrapper comm, const dolfinx::la::SparsityPattern& p,
         const std::string& type) {
        return dolfinx::la::create_petsc_matrix(comm.get(), p, type);
      },
      py::return_value_policy::take_ownership, py::arg("comm"), py::arg("p"),
      py::arg("type") = std::string(),
      "Create a PETSc Mat from sparsity pattern.");
  // TODO: check reference counting for index sets
  m.def("create_petsc_index_sets", &dolfinx::la::create_petsc_index_sets,
        py::arg("maps"), py::arg("is_bs"), py::arg("ghosted") = true,
        py::arg("ghost_block_layout") = dolfinx::la::GhostBlockLayout::intertwined,
        py::return_value_policy::take_ownership);

  // NOTE: Enabling the below requires adding a C API for MatNullSpace to
  // petsc4py
  //   m.def("create_nullspace",
  //         [](const MPICommWrapper comm, MPI_Comm comm,
  //            const dolfinx::la::VectorSpaceBasis& nullspace) {
  //           return dolfinx::la::create_petsc_nullspace(comm.get(),
  //           nullspace);
  //         },
  //         py::return_value_policy::take_ownership,
  //         "Create a PETSc MatNullSpace.");

  py::class_<dolfinx::la::MatSubMatrixWrapper,
             std::shared_ptr<dolfinx::la::MatSubMatrixWrapper>>(m,
                                                            "MatSubMatrixWrapper")
      .def(py::init(
          [](Mat A,
             std::vector<IS> index_sets_) {
            // Due to pybind11#2123, the argument index_sets is of type
            //   std::vector<IS>
            // rather than
            //   std::array<IS, 2>
            // as in the C++ backend. Convert here the std::vector to a std::array.
            // TODO remove this when pybind11#2123 is fixed.
            auto index_sets = convert_vector_to_array(index_sets_);
            return std::make_unique<dolfinx::la::MatSubMatrixWrapper>(A, index_sets);
          }))
      .def(py::init(
          [](Mat A,
             std::vector<IS> unrestricted_index_sets_,
             std::vector<IS> restricted_index_sets_,
             std::array<std::map<std::int32_t, std::int32_t>, 2> unrestricted_to_restricted,
             std::array<int, 2> unrestricted_to_restricted_bs) {
            // Due to pybind11#2123, the arguments {restricted, unrestricted}_index_sets are of type
            //   std::vector<IS>
            // rather than
            //   std::array<IS, 2>
            // as in the C++ backend. Convert here the std::vector to a std::array.
            // TODO remove this when pybind11#2123 is fixed.
            auto unrestricted_index_sets = convert_vector_to_array(unrestricted_index_sets_);
            auto restricted_index_sets = convert_vector_to_array(restricted_index_sets_);
            return std::make_unique<dolfinx::la::MatSubMatrixWrapper>(A, unrestricted_index_sets,
                                                                      restricted_index_sets,
                                                                      unrestricted_to_restricted,
                                                                      unrestricted_to_restricted_bs);
          }))
      .def("restore", &dolfinx::la::MatSubMatrixWrapper::restore)
      .def("mat", &dolfinx::la::MatSubMatrixWrapper::mat);


  py::class_<dolfinx::la::VecSubVectorReadWrapper,
             std::shared_ptr<dolfinx::la::VecSubVectorReadWrapper>>(m,
                                                                    "VecSubVectorReadWrapper")
      .def(py::init<Vec, IS, bool>(),
           py::arg("x"), py::arg("index_set"), py::arg("ghosted") = true)
      .def(py::init<Vec, IS, IS, const std::map<std::int32_t, std::int32_t>&, int, bool>(),
           py::arg("x"), py::arg("unrestricted_index_set"), py::arg("restricted_index_set"),
           py::arg("unrestricted_to_restricted"), py::arg("unrestricted_to_restricted_bs"),
           py::arg("ghosted") = true)
      .def_property_readonly(
          "content",
          [](dolfinx::la::VecSubVectorReadWrapper& self) {
            std::vector<PetscScalar>& array = self.mutable_content();
            return py::array(array.size(), array.data(), py::none());
          },
          py::return_value_policy::reference_internal);

  py::class_<dolfinx::la::VecSubVectorWrapper, dolfinx::la::VecSubVectorReadWrapper,
             std::shared_ptr<dolfinx::la::VecSubVectorWrapper>>(m,
                                                                "VecSubVectorWrapper")
      .def(py::init<Vec, IS, bool>(),
           py::arg("x"), py::arg("index_set"), py::arg("ghosted") = true)
      .def(py::init<Vec, IS, IS, const std::map<std::int32_t, std::int32_t>&, int, bool>(),
           py::arg("x"), py::arg("unrestricted_index_set"), py::arg("restricted_index_set"),
           py::arg("unrestricted_to_restricted"), py::arg("unrestricted_to_restricted_bs"),
           py::arg("ghosted") = true)
      .def("restore", &dolfinx::la::VecSubVectorWrapper::restore);
}
} // namespace dolfinx_wrappers
