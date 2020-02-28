// Copyright (C) 2004-2020 Johan Hoffman, Johan Jansson, Anders Logg, Garth
// N. Wells and Francesco Ballarin
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "PETScOperator.h"
#include "utils.h"
#include <functional>
#include <map>
#include <petscmat.h>
#include <string>

namespace dolfinx::la
{
class SparsityPattern;
class VectorSpaceBasis;

/// Create a PETSc Mat. Caller is responsible for destroying the
/// returned object.
Mat create_petsc_matrix(MPI_Comm comm, const SparsityPattern& sparsity_pattern,
                        const std::string& type = std::string());

/// Create PETSc MatNullSpace. Caller is responsible for destruction
/// returned object.
MatNullSpace create_petsc_nullspace(MPI_Comm comm,
                                    const VectorSpaceBasis& nullspace);

/// It is a simple wrapper for a PETSc matrix pointer (Mat). Its main
/// purpose is to assist memory management of PETSc Mat objects.
///
/// For advanced usage, access the PETSc Mat pointer using the function
/// mat() and use the standard PETSc interface.

class PETScMatrix : public PETScOperator
{
public:
  /// Return a function with an interface for adding or inserting values
  /// into the matrix A (calls MatSetValuesLocal)
  /// @param[in] A The matrix to set values in
  /// @param[in] mode The PETSc insert mode (ADD_VALUES, INSERT_VALUES, ...)
  static std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                           const std::int32_t*, const PetscScalar*)>
  set_fn(Mat A, InsertMode mode);

  /// Return a function with an interface for adding or inserting values
  /// into the matrix A using blocked indices
  /// (calls MatSetValuesBlockedLocal)
  /// @param[in] A The matrix to set values in
  /// @param[in] mode The PETSc insert mode (ADD_VALUES, INSERT_VALUES, ...)
  static std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                           const std::int32_t*, const PetscScalar*)>
  set_block_fn(Mat A, InsertMode mode);

  /// Return a function with an interface for adding or inserting blocked
  /// values to the matrix A using non-blocked insertion (calls
  /// MatSetValuesLocal). Internally it expands the blocked indices into
  /// non-blocked arrays.
  /// @param[in] A The matrix to set values in
  /// @param[in] bs0 Block size for the matrix rows
  /// @param[in] bs1 Block size for the matrix columns
  /// @param[in] mode The PETSc insert mode (ADD_VALUES, INSERT_VALUES, ...)
  static std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                           const std::int32_t*, const PetscScalar*)>
  set_block_expand_fn(Mat A, int bs0, int bs1, InsertMode mode);

  /// Create holder for a PETSc Mat object from a sparsity pattern
  PETScMatrix(MPI_Comm comm, const SparsityPattern& sparsity_pattern,
              const std::string& type = std::string());

  /// Create holder of a PETSc Mat object/pointer. The Mat A object
  /// should already be created. If inc_ref_count is true, the reference
  /// counter of the Mat will be increased. The Mat reference count will
  /// always be decreased upon destruction of the the PETScMatrix.
  PETScMatrix(Mat A, bool inc_ref_count);

  // Copy constructor (deleted)
  PETScMatrix(const PETScMatrix& A) = delete;

  /// Move constructor (falls through to base class move constructor)
  PETScMatrix(PETScMatrix&& A) = default;

  /// Destructor
  ~PETScMatrix() = default;

  /// Assignment operator (deleted)
  PETScMatrix& operator=(const PETScMatrix& A) = delete;

  /// Move assignment operator
  PETScMatrix& operator=(PETScMatrix&& A) = default;

  /// Assembly type
  ///   FINAL - corresponds to PETSc MAT_FINAL_ASSEMBLY
  ///   FLUSH - corresponds to PETSc MAT_FLUSH_ASSEMBLY
  enum class AssemblyType : std::int32_t
  {
    FINAL,
    FLUSH
  };

  /// Finalize assembly of tensor. The following values are recognized
  /// for the mode parameter:
  /// @param type
  ///   FINAL    - corresponds to PETSc MatAssemblyBegin+End(MAT_FINAL_ASSEMBLY)
  ///   FLUSH  - corresponds to PETSc MatAssemblyBegin+End(MAT_FLUSH_ASSEMBLY)
  void apply(AssemblyType type);

  /// Return norm of matrix
  double norm(la::Norm norm_type) const;

  //--- Special PETSc Functions ---

  /// Sets the prefix used by PETSc when searching the options
  /// database
  void set_options_prefix(std::string options_prefix);

  /// Returns the prefix used by PETSc when searching the options
  /// database
  std::string get_options_prefix() const;

  /// Call PETSc function MatSetFromOptions on the PETSc Mat object
  void set_from_options();

  /// Attach nullspace to matrix (typically used by Krylov solvers
  /// when solving singular systems)
  void set_nullspace(const la::VectorSpaceBasis& nullspace);

  /// Attach 'near' nullspace to matrix (used by preconditioners,
  /// such as smoothed aggregation algerbraic multigrid)
  void set_near_nullspace(const la::VectorSpaceBasis& nullspace);
};

/// Wrapper around a local submatrix of a Mat object, used in combination with DofMapRestriction
class MatSubMatrixWrapper
{
public:
  /// Constructor (for cases without restriction)
  MatSubMatrixWrapper(Mat A,
                      std::array<IS, 2> index_sets),

  /// Constructor (for cases with restriction)
  MatSubMatrixWrapper(Mat A,
                      std::array<IS, 2> unrestricted_index_sets,
                      std::array<IS, 2> restricted_index_sets,
                      std::array<std::map<std::int32_t, std::int32_t>, 2> unrestricted_to_restricted,
                      std::array<int, 2> unrestricted_to_restricted_bs);

  /// Destructor
  ~MatSubMatrixWrapper();

  /// Copy constructor (deleted)
  MatSubMatrixWrapper(const MatSubMatrixWrapper& A) = delete;

  /// Move constructor (deleted)
  MatSubMatrixWrapper(MatSubMatrixWrapper&& A) = delete;

  /// Assignment operator (deleted)
  MatSubMatrixWrapper& operator=(const MatSubMatrixWrapper& A) = delete;

  /// Move assignment operator (deleted)
  MatSubMatrixWrapper& operator=(MatSubMatrixWrapper&& A) = delete;

  /// Restore PETSc Mat object
  void restore();

  /// Pointer to submatrix
  Mat mat() const;
private:
  Mat _global_matrix;
  Mat _sub_matrix;
  std::array<IS, 2> _is;
};
} // namespace dolfinx::la
