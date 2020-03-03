// Copyright (C) 2004-2018 Johan Hoffman, Johan Jansson, Anders Logg and
// Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_PETSC

#include "petsc.h"
#include "SparsityPattern.h"
#include "Vector.h"
#include "utils.h"
#include <algorithm>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <iostream>
#include <sstream>

using namespace dolfinx;
using namespace dolfinx::la;

//-----------------------------------------------------------------------------
#define CHECK_ERROR(NAME)                                                      \
  do                                                                           \
  {                                                                            \
    if (ierr != 0)                                                             \
      petsc::error(ierr, __FILE__, NAME);                                      \
  } while (0)

//-----------------------------------------------------------------------------
void la::petsc::error(int error_code, std::string filename,
                      std::string petsc_function)
{
  // Fetch PETSc error description
  const char* desc;
  PetscErrorMessage(error_code, &desc, nullptr);

  // Log detailed error info
  spdlog::info("PETSc error in '{}', '{}'", filename.c_str(),
               petsc_function.c_str());
  spdlog::info("PETSc error code '{}' '{}'", error_code, desc);
  throw std::runtime_error("Failed to successfully call PETSc function '"
                           + petsc_function + "'. PETSc error code is: "
                           + std ::to_string(error_code) + ", "
                           + std::string(desc));
}
//-----------------------------------------------------------------------------
std::vector<Vec>
la::petsc::create_vectors(MPI_Comm comm,
                          const std::vector<std::span<const PetscScalar>>& x)
{
  std::vector<Vec> v(x.size());
  for (std::size_t i = 0; i < v.size(); ++i)
  {
    VecCreateMPI(comm, x[i].size(), PETSC_DETERMINE, &v[i]);
    PetscScalar* data;
    VecGetArray(v[i], &data);
    std::ranges::copy(x[i], data);
    VecRestoreArray(v[i], &data);
  }

  return v;
}
//-----------------------------------------------------------------------------
Vec la::petsc::create_vector(const common::IndexMap& map, int bs)
{
  return la::petsc::create_vector(map.comm(), map.local_range(), map.ghosts(),
                                  bs);
}
//-----------------------------------------------------------------------------
Vec la::petsc::create_vector(MPI_Comm comm, std::array<std::int64_t, 2> range,
                             std::span<const std::int64_t> ghosts, int bs)
{
  PetscErrorCode ierr;

  // Get local size
  assert(range[1] >= range[0]);
  std::int32_t local_size = range[1] - range[0];

  Vec x = nullptr;
  std::vector<PetscInt> _ghosts(ghosts.begin(), ghosts.end());
  if (bs == 1)
  {
    ierr = VecCreateGhost(comm, local_size, PETSC_DETERMINE, _ghosts.size(),
                          _ghosts.data(), &x);
    CHECK_ERROR("VecCreateGhost");
  }
  else
  {
    ierr = VecCreateGhostBlock(comm, bs, bs * local_size, PETSC_DETERMINE,
                               _ghosts.size(), _ghosts.data(), &x);
    CHECK_ERROR("VecCreateGhostBlock");
  }

  assert(x);
  return x;
}
//-----------------------------------------------------------------------------
Vec la::petsc::create_vector_wrap(const common::IndexMap& map, int bs,
                                  std::span<const PetscScalar> x)
{
  const std::int32_t size_local = bs * map.size_local();
  const std::int64_t size_global = bs * map.size_global();
  const std::vector<PetscInt> ghosts(map.ghosts().begin(), map.ghosts().end());
  Vec vec;
  PetscErrorCode ierr;
  if (bs == 1)
  {
    ierr
        = VecCreateGhostWithArray(map.comm(), size_local, size_global,
                                  ghosts.size(), ghosts.data(), x.data(), &vec);
    CHECK_ERROR("VecCreateGhostWithArray");
  }
  else
  {
    ierr = VecCreateGhostBlockWithArray(map.comm(), bs, size_local, size_global,
                                        ghosts.size(), ghosts.data(), x.data(),
                                        &vec);
    CHECK_ERROR("VecCreateGhostBlockWithArray");
  }

  assert(vec);
  return vec;
}
//-----------------------------------------------------------------------------
std::vector<IS> la::petsc::create_index_sets(
    const std::vector<
        std::pair<std::reference_wrapper<const common::IndexMap>, int>>& maps,
    const std::vector<int> is_bs, bool ghosted, GhostBlockLayout ghost_block_layout)
{
  assert(maps.size() == is_bs.size());
  std::vector<std::int32_t> size_local(maps.size());
  std::vector<std::int32_t> size_ghost(maps.size());
  std::vector<int> bs(maps.size());
  std::generate(size_local.begin(), size_local.end(), [i = 0, maps] () mutable {
    return maps[i++].first.get().size_local();
  });
  std::generate(size_ghost.begin(), size_ghost.end(), [i = 0, maps] () mutable {
    return maps[i++].first.get().num_ghosts();
  });
  std::generate(bs.begin(), bs.end(), [i = 0, maps, is_bs] () mutable {
    auto bs_i = maps[i].second;
    auto is_bs_i = is_bs[i];
    i++;
    assert(is_bs_i == bs_i || is_bs_i == 1);
    if (is_bs_i == 1)
      return bs_i;
    else
      return 1;
  });

  // Initialize storage for indices
  std::vector<std::vector<PetscInt>> index(maps.size());
  for (std::size_t i = 0; i < maps.size(); ++i)
  {
    if (ghosted)
    {
      index[i].resize(bs[i] * (size_local[i] + size_ghost[i]));
    }
    else
    {
      index[i].resize(bs[i] * size_local[i]);
    }
  }

  // Compute indices and offset
  std::size_t offset = 0;
  for (std::size_t i = 0; i < maps.size(); ++i)
  {
    if (ghosted && ghost_block_layout == GhostBlockLayout::intertwined)
    {
      std::iota(index[i].begin(), std::next(index[i].begin(), bs[i] * (size_local[i] + size_ghost[i])), offset);
    }
    else
    {
      std::iota(index[i].begin(), std::next(index[i].begin(), bs[i] * size_local[i]), offset);
    }

    offset += bs[i] * size_local[i];
    if (ghost_block_layout == GhostBlockLayout::intertwined)
    {
      offset += bs[i] * size_ghost[i];
    }
  }
  if (ghosted && ghost_block_layout == GhostBlockLayout::trailing)
  {
    for (std::size_t i = 0; i < maps.size(); ++i)
    {
      std::iota(std::next(index[i].begin(), bs[i] * size_local[i]),
                std::next(index[i].begin(), bs[i] * (size_local[i] + size_ghost[i])), offset);

      offset += bs[i] * size_ghost[i];
    }
  }

  // Initialize PETSc IS objects
  std::vector<IS> is(maps.size());
  for (std::size_t i = 0; i < maps.size(); ++i)
  {
    ISCreateBlock(PETSC_COMM_SELF, is_bs[i], index[i].size(), index[i].data(),
                  PETSC_COPY_VALUES, &is[i]);
  }
  return is;
}
//-----------------------------------------------------------------------------
Mat la::petsc::create_matrix(MPI_Comm comm,
                             const dolfinx::la::SparsityPattern& sp,
                             std::string type)
{
  PetscErrorCode ierr;
  Mat A;
  ierr = MatCreate(comm, &A);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "MatCreate");

  // Get IndexMaps from sparsity patterm, and block size
  std::array maps = {sp.index_map(0), sp.index_map(1)};
  const std::array bs = {sp.block_size(0), sp.block_size(1)};

  if (!type.empty())
    MatSetType(A, type.c_str());

  // Get global and local dimensions
  const std::int64_t M = bs[0] * maps[0]->size_global();
  const std::int64_t N = bs[1] * maps[1]->size_global();
  const std::int32_t m = bs[0] * maps[0]->size_local();
  const std::int32_t n = bs[1] * maps[1]->size_local();

  // Set matrix size
  ierr = MatSetSizes(A, m, n, M, N);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "MatSetSizes");

  // Apply PETSc options from the options database to the matrix (this
  // includes changing the matrix type to one specified by the user)
  ierr = MatSetFromOptions(A);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "MatSetFromOptions");

  // Find a common block size across rows/columns
  const int _bs = (bs[0] == bs[1] ? bs[0] : 1);

  // Build data to initialise sparsity pattern (modify for block size)
  std::vector<PetscInt> _nnz_diag, _nnz_offdiag;
  if (bs[0] == bs[1])
  {
    _nnz_diag.resize(maps[0]->size_local());
    _nnz_offdiag.resize(maps[0]->size_local());
    for (std::size_t i = 0; i < _nnz_diag.size(); ++i)
      _nnz_diag[i] = sp.nnz_diag(i);
    for (std::size_t i = 0; i < _nnz_offdiag.size(); ++i)
      _nnz_offdiag[i] = sp.nnz_off_diag(i);
  }
  else
  {
    // Expand for block size 1
    _nnz_diag.resize(maps[0]->size_local() * bs[0]);
    _nnz_offdiag.resize(maps[0]->size_local() * bs[0]);
    for (std::size_t i = 0; i < _nnz_diag.size(); ++i)
      _nnz_diag[i] = bs[1] * sp.nnz_diag(i / bs[0]);
    for (std::size_t i = 0; i < _nnz_offdiag.size(); ++i)
      _nnz_offdiag[i] = bs[1] * sp.nnz_off_diag(i / bs[0]);
  }

  // Allocate space for matrix
  ierr = MatXAIJSetPreallocation(A, _bs, _nnz_diag.data(), _nnz_offdiag.data(),
                                 nullptr, nullptr);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "MatXIJSetPreallocation");

  // Set block sizes
  ierr = MatSetBlockSizes(A, bs[0], bs[1]);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "MatSetBlockSizes");

  // Create PETSc local-to-global map/index sets
  ISLocalToGlobalMapping local_to_global0;
  const std::vector map0 = maps[0]->global_indices();
  const std::vector<PetscInt> _map0(map0.begin(), map0.end());
  ierr = ISLocalToGlobalMappingCreate(MPI_COMM_SELF, bs[0], _map0.size(),
                                      _map0.data(), PETSC_COPY_VALUES,
                                      &local_to_global0);

  if (ierr != 0)
    petsc::error(ierr, __FILE__, "ISLocalToGlobalMappingCreate");

  // Check for common index maps
  if (maps[0] == maps[1] and bs[0] == bs[1])
  {
    ierr = MatSetLocalToGlobalMapping(A, local_to_global0, local_to_global0);
    if (ierr != 0)
      petsc::error(ierr, __FILE__, "MatSetLocalToGlobalMapping");
  }
  else
  {
    ISLocalToGlobalMapping local_to_global1;
    const std::vector map1 = maps[1]->global_indices();
    const std::vector<PetscInt> _map1(map1.begin(), map1.end());
    ierr = ISLocalToGlobalMappingCreate(MPI_COMM_SELF, bs[1], _map1.size(),
                                        _map1.data(), PETSC_COPY_VALUES,
                                        &local_to_global1);
    if (ierr != 0)
      petsc::error(ierr, __FILE__, "ISLocalToGlobalMappingCreate");
    ierr = MatSetLocalToGlobalMapping(A, local_to_global0, local_to_global1);
    if (ierr != 0)
      petsc::error(ierr, __FILE__, "MatSetLocalToGlobalMapping");
    ierr = ISLocalToGlobalMappingDestroy(&local_to_global1);
    if (ierr != 0)
      petsc::error(ierr, __FILE__, "ISLocalToGlobalMappingDestroy");
  }

  // Clean up local-to-global 0
  ierr = ISLocalToGlobalMappingDestroy(&local_to_global0);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "ISLocalToGlobalMappingDestroy");

  // Note: This should be called after having set the local-to-global
  // map for MATIS (this is a dummy call if A is not of type MATIS)
  // ierr = MatISSetPreallocation(A, 0, _nnz_diag.data(), 0,
  // _nnz_offdiag.data()); if (ierr != 0)
  //   error(ierr, __FILE__, "MatISSetPreallocation");

  // Set some options on Mat object
  ierr = MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "MatSetOption");
  ierr = MatSetOption(A, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "MatSetOption");

  return A;
}
//-----------------------------------------------------------------------------
MatNullSpace la::petsc::create_nullspace(MPI_Comm comm,
                                         std::span<const Vec> basis)
{
  MatNullSpace ns = nullptr;
  PetscErrorCode ierr
      = MatNullSpaceCreate(comm, PETSC_FALSE, basis.size(), basis.data(), &ns);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "MatNullSpaceCreate");
  return ns;
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void petsc::options::set(std::string option)
{
  petsc::options::set<std::string>(option, "");
}
//-----------------------------------------------------------------------------
void petsc::options::clear(std::string option)
{
  if (option[0] != '-')
    option = '-' + option;

  PetscErrorCode ierr;
  ierr = PetscOptionsClearValue(nullptr, option.c_str());
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "PetscOptionsClearValue");
}
//-----------------------------------------------------------------------------
void petsc::options::clear()
{
  PetscErrorCode ierr = PetscOptionsClear(nullptr);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "PetscOptionsClear");
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
petsc::Vector::Vector(const common::IndexMap& map, int bs)
    : _x(la::petsc::create_vector(map, bs))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
petsc::Vector::Vector(Vec x, bool inc_ref_count) : _x(x)
{
  assert(x);
  if (inc_ref_count)
    PetscObjectReference((PetscObject)_x);
}
//-----------------------------------------------------------------------------
petsc::Vector::Vector(Vector&& v) : _x(std::exchange(v._x, nullptr)) {}
//-----------------------------------------------------------------------------
petsc::Vector::~Vector()
{
  if (_x)
    VecDestroy(&_x);
}
//-----------------------------------------------------------------------------
petsc::Vector& petsc::Vector::operator=(Vector&& v)
{
  std::swap(_x, v._x);
  return *this;
}
//-----------------------------------------------------------------------------
petsc::Vector petsc::Vector::copy() const
{
  Vec _y;
  VecDuplicate(_x, &_y);
  VecCopy(_x, _y);
  Vector y(_y, true);
  VecDestroy(&_y);
  return y;
}
//-----------------------------------------------------------------------------
std::int64_t petsc::Vector::size() const
{
  assert(_x);
  PetscInt n = 0;
  PetscErrorCode ierr = VecGetSize(_x, &n);
  CHECK_ERROR("VecGetSize");
  return n;
}
//-----------------------------------------------------------------------------
std::int32_t petsc::Vector::local_size() const
{
  assert(_x);
  PetscInt n = 0;
  PetscErrorCode ierr = VecGetLocalSize(_x, &n);
  CHECK_ERROR("VecGetLocalSize");
  return n;
}
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2> petsc::Vector::local_range() const
{
  assert(_x);
  PetscInt n0, n1;
  PetscErrorCode ierr = VecGetOwnershipRange(_x, &n0, &n1);
  CHECK_ERROR("VecGetOwnershipRange");
  assert(n0 <= n1);
  return {n0, n1};
}
//-----------------------------------------------------------------------------
MPI_Comm petsc::Vector::comm() const
{
  assert(_x);
  MPI_Comm mpi_comm = MPI_COMM_NULL;
  PetscErrorCode ierr = PetscObjectGetComm((PetscObject)(_x), &mpi_comm);
  CHECK_ERROR("PetscObjectGetComm");
  return mpi_comm;
}
//-----------------------------------------------------------------------------
void petsc::Vector::set_options_prefix(std::string options_prefix)
{
  assert(_x);
  PetscErrorCode ierr = VecSetOptionsPrefix(_x, options_prefix.c_str());
  CHECK_ERROR("VecSetOptionsPrefix");
}
//-----------------------------------------------------------------------------
std::string petsc::Vector::get_options_prefix() const
{
  assert(_x);
  const char* prefix = nullptr;
  PetscErrorCode ierr = VecGetOptionsPrefix(_x, &prefix);
  CHECK_ERROR("VecGetOptionsPrefix");
  return std::string(prefix);
}
//-----------------------------------------------------------------------------
void petsc::Vector::set_from_options()
{
  assert(_x);
  PetscErrorCode ierr = VecSetFromOptions(_x);
  CHECK_ERROR("VecSetFromOptions");
}
//-----------------------------------------------------------------------------
Vec petsc::Vector::vec() const { return _x; }
//-----------------------------------------------------------------------------
petsc::VecSubVectorReadWrapper::VecSubVectorReadWrapper(
  Vec x,
  IS index_set,
  bool ghosted
) : _ghosted(ghosted)
{
  PetscErrorCode ierr;

  // Get number of entries to extract from x
  PetscInt is_size;
  ierr = ISGetLocalSize(index_set, &is_size);
  if (ierr != 0) petsc::error(ierr, __FILE__, "ISGetLocalSize");

  // Get indices of entries to extract from x
  const PetscInt *indices;
  ierr = ISGetIndices(index_set, &indices);
  if (ierr != 0) petsc::error(ierr, __FILE__, "ISGetIndices");

  // Fetch vector content from x
  Vec x_local_form;
  if (_ghosted)
  {
    ierr = VecGhostGetLocalForm(x, &x_local_form);
    if (ierr != 0) petsc::error(ierr, __FILE__, "VecGhostGetLocalForm");
  }
  else
  {
    x_local_form = x;
  }
  _content.resize(is_size, 0.);
  ierr = VecGetValues(x_local_form, is_size, indices, _content.data());
  if (ierr != 0) petsc::error(ierr, __FILE__, "VecGetValues");
  if (_ghosted)
  {
    ierr = VecGhostRestoreLocalForm(x, &x_local_form);
    if (ierr != 0) petsc::error(ierr, __FILE__, "VecGhostRestoreLocalForm");
  }

  // Restore indices
  ierr = ISRestoreIndices(index_set, &indices);
  if (ierr != 0) petsc::error(ierr, __FILE__, "ISRestoreIndices");
}
//-----------------------------------------------------------------------------
petsc::VecSubVectorReadWrapper::VecSubVectorReadWrapper(
  Vec x,
  IS unrestricted_index_set,
  IS restricted_index_set,
  const std::map<std::int32_t, std::int32_t>& unrestricted_to_restricted,
  int unrestricted_to_restricted_bs,
  bool ghosted)
  : _ghosted(ghosted)
{
  PetscErrorCode ierr;

  // Get number of entries to extract from x
  PetscInt restricted_is_size;
  ierr = ISGetLocalSize(restricted_index_set, &restricted_is_size);
  if (ierr != 0) petsc::error(ierr, __FILE__, "ISGetLocalSize");

  // Get indices of entries to extract from x
  const PetscInt *restricted_indices;
  ierr = ISGetIndices(restricted_index_set, &restricted_indices);
  if (ierr != 0) petsc::error(ierr, __FILE__, "ISGetIndices");

  // Fetch vector content from x
  Vec x_local_form;
  if (_ghosted)
  {
    ierr = VecGhostGetLocalForm(x, &x_local_form);
    if (ierr != 0) petsc::error(ierr, __FILE__, "VecGhostGetLocalForm");
  }
  else
  {
    x_local_form = x;
  }
  std::vector<PetscScalar> restricted_content(restricted_is_size, 0.);
  ierr = VecGetValues(x_local_form, restricted_is_size, restricted_indices, restricted_content.data());
  if (ierr != 0) petsc::error(ierr, __FILE__, "VecGetValues");
  if (_ghosted)
  {
    ierr = VecGhostRestoreLocalForm(x, &x_local_form);
    if (ierr != 0) petsc::error(ierr, __FILE__, "VecGhostRestoreLocalForm");
  }

  // Restore indices
  ierr = ISRestoreIndices(restricted_index_set, &restricted_indices);
  if (ierr != 0) petsc::error(ierr, __FILE__, "ISRestoreIndices");

  // Get number of entries to be stored in _content
  PetscInt unrestricted_is_size;
  ierr = ISGetLocalSize(unrestricted_index_set, &unrestricted_is_size);
  if (ierr != 0) petsc::error(ierr, __FILE__, "ISGetLocalSize");

  // Assign vector content to an STL vector indexed with respect to the unrestricted index set
  _content.resize(unrestricted_is_size, 0.);
  for (PetscInt unrestricted_index = 0; unrestricted_index < unrestricted_is_size; unrestricted_index++)
  {
    if (unrestricted_to_restricted.count(unrestricted_index/unrestricted_to_restricted_bs) > 0)
    {
      _content[unrestricted_index] = restricted_content[
        unrestricted_to_restricted_bs*unrestricted_to_restricted.at(
          unrestricted_index/unrestricted_to_restricted_bs) +
            unrestricted_index%unrestricted_to_restricted_bs];
    }
  }
}
//-----------------------------------------------------------------------------
petsc::VecSubVectorReadWrapper::~VecSubVectorReadWrapper()
{
  // Nothing to be done
}
//-----------------------------------------------------------------------------
petsc::VecSubVectorWrapper::VecSubVectorWrapper(
  Vec x,
  IS index_set,
  bool ghosted)
  : VecSubVectorReadWrapper(x, index_set, ghosted),
    _global_vector(x), _is(index_set)
{
  PetscErrorCode ierr;

  // Get number of entries stored in _content
  PetscInt is_size;
  ierr = ISGetLocalSize(index_set, &is_size);
  if (ierr != 0) petsc::error(ierr, __FILE__, "ISGetLocalSize");

  // Fill in _restricted_to_unrestricted attribute with the identity map
  for (PetscInt index = 0; index < is_size; index++)
  {
    _restricted_to_unrestricted[index] = index;
  }
}
//-----------------------------------------------------------------------------
petsc::VecSubVectorWrapper::VecSubVectorWrapper(
  Vec x,
  IS unrestricted_index_set,
  IS restricted_index_set,
  const std::map<std::int32_t, std::int32_t>& unrestricted_to_restricted,
  int unrestricted_to_restricted_bs,
  bool ghosted)
  : VecSubVectorReadWrapper(x, unrestricted_index_set, restricted_index_set, unrestricted_to_restricted,
                            unrestricted_to_restricted_bs, ghosted),
    _global_vector(x), _is(restricted_index_set)
{
  PetscErrorCode ierr;

  // Get number of entries stored in _content
  PetscInt unrestricted_is_size;
  ierr = ISGetLocalSize(unrestricted_index_set, &unrestricted_is_size);
  if (ierr != 0) petsc::error(ierr, __FILE__, "ISGetLocalSize");

  // Fill in _restricted_to_unrestricted attribute
  for (PetscInt unrestricted_index = 0; unrestricted_index < unrestricted_is_size; unrestricted_index++)
  {
    if (unrestricted_to_restricted.count(unrestricted_index/unrestricted_to_restricted_bs) > 0)
    {
      _restricted_to_unrestricted[
        unrestricted_to_restricted_bs*unrestricted_to_restricted.at(
          unrestricted_index/unrestricted_to_restricted_bs) +
            unrestricted_index%unrestricted_to_restricted_bs] = unrestricted_index;
    }
  }
}
//-----------------------------------------------------------------------------
petsc::VecSubVectorWrapper::~VecSubVectorWrapper()
{
  // Sub vector should have been restored before destroying object
  assert(!_is);
  assert(_restricted_to_unrestricted.size() == 0);
  assert(_content.size() == 0);
}
//-----------------------------------------------------------------------------
void petsc::VecSubVectorWrapper::restore()
{
  PetscErrorCode ierr;

  // Get indices of entries to restore in x
  const PetscInt *restricted_indices;
  ierr = ISGetIndices(_is, &restricted_indices);
  if (ierr != 0) petsc::error(ierr, __FILE__, "ISGetIndices");

  // Restrict values from content attribute
  std::vector<PetscScalar> restricted_values(_restricted_to_unrestricted.size());
  for (auto& restricted_to_unrestricted_it: _restricted_to_unrestricted)
  {
    auto restricted_index = restricted_to_unrestricted_it.first;
    auto unrestricted_index = restricted_to_unrestricted_it.second;
    restricted_values[restricted_index] = _content[unrestricted_index];
  }

  // Insert values calling PETSc API
  Vec global_vector_local_form;
  if (_ghosted)
  {
    ierr = VecGhostGetLocalForm(_global_vector, &global_vector_local_form);
    if (ierr != 0) petsc::error(ierr, __FILE__, "VecGhostGetLocalForm");
  }
  else
  {
    global_vector_local_form = _global_vector;
  }
  PetscScalar* array_local_form;
  ierr = VecGetArray(global_vector_local_form, &array_local_form);
  if (ierr != 0) petsc::error(ierr, __FILE__, "VecGetArray");
  for (std::size_t i = 0; i < restricted_values.size(); ++i)
    array_local_form[restricted_indices[i]] = restricted_values[i];
  ierr = VecRestoreArray(global_vector_local_form, &array_local_form);
  if (ierr != 0) petsc::error(ierr, __FILE__, "VecRestoreArray");
  if (_ghosted)
  {
    ierr = VecGhostRestoreLocalForm(_global_vector, &global_vector_local_form);
    if (ierr != 0) petsc::error(ierr, __FILE__, "VecGhostRestoreLocalForm");
  }

  // Restore indices
  ierr = ISRestoreIndices(_is, &restricted_indices);
  if (ierr != 0) petsc::error(ierr, __FILE__, "ISRestoreIndices");

  // Clear storage
  _is = nullptr;
  _restricted_to_unrestricted.clear();
  _content.clear();
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
petsc::Operator::Operator(Mat A, bool inc_ref_count) : _matA(A)
{
  assert(A);
  if (inc_ref_count)
    PetscObjectReference((PetscObject)_matA);
}
//-----------------------------------------------------------------------------
petsc::Operator::Operator(Operator&& A) : _matA(std::exchange(A._matA, nullptr))
{
}
//-----------------------------------------------------------------------------
petsc::Operator::~Operator()
{
  // Decrease reference count (PETSc will destroy object once reference
  // counts reached zero)
  if (_matA)
    MatDestroy(&_matA);
}
//-----------------------------------------------------------------------------
petsc::Operator& petsc::Operator::operator=(Operator&& A)
{
  std::swap(_matA, A._matA);
  return *this;
}
//-----------------------------------------------------------------------------
std::array<std::int64_t, 2> petsc::Operator::size() const
{
  assert(_matA);
  PetscInt m(0), n(0);
  PetscErrorCode ierr = MatGetSize(_matA, &m, &n);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "MetGetSize");
  return {{m, n}};
}
//-----------------------------------------------------------------------------
Vec petsc::Operator::create_vector(std::size_t dim) const
{
  assert(_matA);
  PetscErrorCode ierr;

  Vec x = nullptr;
  if (dim == 0)
  {
    ierr = MatCreateVecs(_matA, nullptr, &x);
    if (ierr != 0)
      petsc::error(ierr, __FILE__, "MatCreateVecs");
  }
  else if (dim == 1)
  {
    ierr = MatCreateVecs(_matA, &x, nullptr);
    if (ierr != 0)
      petsc::error(ierr, __FILE__, "MatCreateVecs");
  }
  else
  {
    spdlog::error("Cannot initialize PETSc vector to match PETSc matrix. "
                  "Dimension must be 0 or 1, not {}",
                  dim);
    throw std::runtime_error("Invalid dimension");
  }

  return x;
}
//-----------------------------------------------------------------------------
Mat petsc::Operator::mat() const { return _matA; }
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
petsc::Matrix::Matrix(MPI_Comm comm, const SparsityPattern& sp,
                      std::string type)
    : Operator(petsc::create_matrix(comm, sp, type), false)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
petsc::Matrix::Matrix(Mat A, bool inc_ref_count) : Operator(A, inc_ref_count)
{
  // Reference count to A is incremented in base class
}
//-----------------------------------------------------------------------------
double petsc::Matrix::norm(Norm norm_type) const
{
  assert(_matA);
  PetscErrorCode ierr;
  PetscReal value = 0;
  switch (norm_type)
  {
  case Norm::l1:
    ierr = MatNorm(_matA, NORM_1, &value);
    break;
  case Norm::linf:
    ierr = MatNorm(_matA, NORM_INFINITY, &value);
    break;
  case Norm::frobenius:
    ierr = MatNorm(_matA, NORM_FROBENIUS, &value);
    break;
  default:
    throw std::runtime_error("Unknown PETSc Mat norm type");
  }

  if (ierr != 0)
    petsc::error(ierr, __FILE__, "MatNorm");
  return value;
}
//-----------------------------------------------------------------------------
void petsc::Matrix::apply(AssemblyType type)
{
  common::Timer timer("Apply (PETScMatrix)");

  assert(_matA);
  PetscErrorCode ierr;
  MatAssemblyType petsc_type = MAT_FINAL_ASSEMBLY;
  if (type == AssemblyType::FLUSH)
    petsc_type = MAT_FLUSH_ASSEMBLY;
  ierr = MatAssemblyBegin(_matA, petsc_type);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "MatAssemblyBegin");
  ierr = MatAssemblyEnd(_matA, petsc_type);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "MatAssemblyEnd");
}
//-----------------------------------------------------------------------------
void petsc::Matrix::set_options_prefix(std::string options_prefix)
{
  assert(_matA);
  MatSetOptionsPrefix(_matA, options_prefix.c_str());
}
//-----------------------------------------------------------------------------
std::string petsc::Matrix::get_options_prefix() const
{
  assert(_matA);
  const char* prefix = nullptr;
  MatGetOptionsPrefix(_matA, &prefix);
  return std::string(prefix);
}
//-----------------------------------------------------------------------------
void petsc::Matrix::set_from_options()
{
  assert(_matA);
  MatSetFromOptions(_matA);
}
//-----------------------------------------------------------------------------
petsc::MatSubMatrixWrapper::MatSubMatrixWrapper(
  Mat A,
  std::array<IS, 2> index_sets)
  : _global_matrix(A), _is(index_sets)
{
  PetscErrorCode ierr;

  // Get communicator from matrix object
  MPI_Comm mpi_comm = MPI_COMM_NULL;
  ierr = PetscObjectGetComm((PetscObject) A, &mpi_comm);
  if (ierr != 0) petsc::error(ierr, __FILE__, "PetscObjectGetComm");

  // Sub matrix inherits block size of the index sets. Check that they
  // are consistent with the ones of the global matrix.
  std::vector<PetscInt> bs_A(2);
  ierr = MatGetBlockSizes(A, &bs_A[0], &bs_A[1]);
  if (ierr != 0) petsc::error(ierr, __FILE__, "MatGetBlockSizes");
  std::vector<PetscInt> bs_is(2);
  ierr = ISGetBlockSize(_is[0], &bs_is[0]);
  if (ierr != 0) petsc::error(ierr, __FILE__, "ISGetBlockSize");
  ierr = ISGetBlockSize(_is[1], &bs_is[1]);
  if (ierr != 0) petsc::error(ierr, __FILE__, "ISGetBlockSize");
  assert(bs_A[0] == bs_is[0]);
  assert(bs_A[1] == bs_is[1]);

  // Extract sub matrix
  ierr = MatGetLocalSubMatrix(A, _is[0], _is[1], &_sub_matrix);
  if (ierr != 0) petsc::error(ierr, __FILE__, "MatGetLocalSubMatrix");
}
//-----------------------------------------------------------------------------
petsc::MatSubMatrixWrapper::MatSubMatrixWrapper(
  Mat A,
  std::array<IS, 2> unrestricted_index_sets,
  std::array<IS, 2> restricted_index_sets,
  std::array<std::map<std::int32_t, std::int32_t>, 2> unrestricted_to_restricted,
  std::array<int, 2> unrestricted_to_restricted_bs)
  : MatSubMatrixWrapper(A, restricted_index_sets)
{
  PetscErrorCode ierr;

  // Initialization of custom local to global PETSc map.
  // In order not to change the assembly routines, here "local" is intended
  // with respect to the *unrestricted* index sets (which where generated using
  // the index map that will be passed to the assembly routines). Instead,
  // "global" is intended with respect to the *restricted* index sets for
  // entries in the restriction, while it is set to -1 (i.e., values corresponding
  // to those indices will be discarded) for entries not in the restriction.

  // Get sub matrix (i.e., index sets) block sizes
  std::vector<PetscInt> bs(2);
  ierr = MatGetBlockSizes(_sub_matrix, &bs[0], &bs[1]);
  if (ierr != 0) petsc::error(ierr, __FILE__, "MatGetBlockSizes");

  // Compare sub matrix block sizes with unrestricted_to_restricted_bs:
  // they should either be the same (typically the case of restricted matrices or restricted
  // nest matrices) or unrestricted_to_restricted_bs may be larger than the sub matrix block
  // sizes (typically the case of restricted block matrices, because bs is forced to one).
  assert(bs[0] == unrestricted_to_restricted_bs[0] || (bs[0] == 1 && unrestricted_to_restricted_bs[0] > 1));
  assert(bs[1] == unrestricted_to_restricted_bs[1] || (bs[1] == 1 && unrestricted_to_restricted_bs[1] > 1));
  std::vector<PetscInt> unrestricted_to_restricted_correction(2);
  for (std::size_t i = 0; i < 2; ++i)
  {
    if (bs[i] == unrestricted_to_restricted_bs[i])
    {
      unrestricted_to_restricted_correction[i] = 1;
    }
    else
    {
      assert(bs[i] == 1);
      unrestricted_to_restricted_correction[i] = unrestricted_to_restricted_bs[i];
    }
  }

  // Get matrix local-to-global map
  std::array<ISLocalToGlobalMapping, 2> petsc_local_to_global_matrix;
  ierr = MatGetLocalToGlobalMapping(A, &petsc_local_to_global_matrix[0], &petsc_local_to_global_matrix[1]);
  if (ierr != 0) petsc::error(ierr, __FILE__, "MatGetLocalToGlobalMapping");

  // Allocate data for submatrix local-to-global maps in an STL vector
  std::array<std::vector<PetscInt>, 2> stl_local_to_global_submatrix;
  for (std::size_t i = 0; i < 2; ++i)
  {
    PetscInt unrestricted_is_size;
    ierr = ISBlockGetLocalSize(unrestricted_index_sets[i], &unrestricted_is_size);
    if (ierr != 0) petsc::error(ierr, __FILE__, "ISGetLocalSize");
    stl_local_to_global_submatrix[i].resize(unrestricted_is_size);

    const PetscInt *restricted_indices;
    ierr = ISBlockGetIndices(restricted_index_sets[i], &restricted_indices);
    if (ierr != 0) petsc::error(ierr, __FILE__, "ISGetIndices");

    std::vector<PetscInt> restricted_local_index(1);
    std::vector<PetscInt> restricted_global_index(1);

    for (PetscInt unrestricted_index = 0; unrestricted_index < unrestricted_is_size; unrestricted_index++)
    {
      if (unrestricted_to_restricted[i].count(unrestricted_index/unrestricted_to_restricted_correction[i]) > 0)
      {
        restricted_local_index[0] = restricted_indices[
          unrestricted_to_restricted_correction[i]*unrestricted_to_restricted[i].at(
            unrestricted_index/unrestricted_to_restricted_correction[i]) +
              unrestricted_index%unrestricted_to_restricted_correction[i]];
        ISLocalToGlobalMappingApplyBlock(petsc_local_to_global_matrix[i], restricted_local_index.size(),
                                         restricted_local_index.data(), restricted_global_index.data());
        stl_local_to_global_submatrix[i][unrestricted_index] = restricted_global_index[0];
      }
      else
      {
        stl_local_to_global_submatrix[i][unrestricted_index] = -1;
      }
    }

    ierr = ISBlockRestoreIndices(restricted_index_sets[i], &restricted_indices);
    if (ierr != 0) petsc::error(ierr, __FILE__, "ISRestoreIndices");
  }

  // Get communicator from submatrix object
  MPI_Comm mpi_comm = MPI_COMM_NULL;
  ierr = PetscObjectGetComm((PetscObject) _sub_matrix, &mpi_comm);
  if (ierr != 0) petsc::error(ierr, __FILE__, "PetscObjectGetComm");

  // Create submatrix local-to-global maps as index set
  std::array<ISLocalToGlobalMapping, 2> petsc_local_to_global_submatrix;
  for (std::size_t i = 0; i < 2; ++i)
  {
    ierr = ISLocalToGlobalMappingCreate(mpi_comm, bs[i], stl_local_to_global_submatrix[i].size(),
                                        stl_local_to_global_submatrix[i].data(), PETSC_COPY_VALUES,
                                        &petsc_local_to_global_submatrix[i]);
    if (ierr != 0) petsc::error(ierr, __FILE__, "ISLocalToGlobalMappingCreate");
  }

  // Set submatrix local-to-global maps
  ierr = MatSetLocalToGlobalMapping(_sub_matrix, petsc_local_to_global_submatrix[0],
                                    petsc_local_to_global_submatrix[1]);
  if (ierr != 0) petsc::error(ierr, __FILE__, "MatSetLocalToGlobalMapping");

  // Clean up submatrix local-to-global maps
  for (std::size_t i = 0; i < 2; ++i)
  {
    ierr = ISLocalToGlobalMappingDestroy(&petsc_local_to_global_submatrix[i]);
    if (ierr != 0) petsc::error(ierr, __FILE__, "ISLocalToGlobalMappingDestroy");
  }
}
//-----------------------------------------------------------------------------
petsc::MatSubMatrixWrapper::~MatSubMatrixWrapper()
{
  // Sub matrix should have been restored before destroying object
  assert(!_sub_matrix);
  assert(!_is[0]);
  assert(!_is[1]);
}
//-----------------------------------------------------------------------------
void petsc::MatSubMatrixWrapper::restore()
{
  // Restore the global matrix
  PetscErrorCode ierr;
  assert(_sub_matrix);
  ierr = MatRestoreLocalSubMatrix(_global_matrix, _is[0], _is[1], &_sub_matrix);
  if (ierr != 0) petsc::error(ierr, __FILE__, "MatRestoreLocalSubMatrix");

  // Clear pointers
  _sub_matrix = nullptr;
  _is.fill(nullptr);
}
//-----------------------------------------------------------------------------
Mat petsc::MatSubMatrixWrapper::mat() const
{
  return _sub_matrix;
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
petsc::KrylovSolver::KrylovSolver(MPI_Comm comm) : _ksp(nullptr)
{
  // Create PETSc KSP object
  PetscErrorCode ierr = KSPCreate(comm, &_ksp);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "KSPCreate");
}
//-----------------------------------------------------------------------------
petsc::KrylovSolver::KrylovSolver(KSP ksp, bool inc_ref_count) : _ksp(ksp)
{
  assert(_ksp);
  if (inc_ref_count)
  {
    PetscErrorCode ierr = PetscObjectReference((PetscObject)_ksp);
    if (ierr != 0)
      petsc::error(ierr, __FILE__, "PetscObjectReference");
  }
}
//-----------------------------------------------------------------------------
petsc::KrylovSolver::KrylovSolver(KrylovSolver&& solver)
    : _ksp(std::exchange(solver._ksp, nullptr))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
petsc::KrylovSolver::~KrylovSolver()
{
  if (_ksp)
    KSPDestroy(&_ksp);
}
//-----------------------------------------------------------------------------
petsc::KrylovSolver& petsc::KrylovSolver::operator=(KrylovSolver&& solver)
{
  std::swap(_ksp, solver._ksp);
  return *this;
}
//-----------------------------------------------------------------------------
void petsc::KrylovSolver::set_operator(const Mat A) { set_operators(A, A); }
//-----------------------------------------------------------------------------
void petsc::KrylovSolver::set_operators(const Mat A, const Mat P)
{
  assert(A);
  assert(_ksp);
  PetscErrorCode ierr = KSPSetOperators(_ksp, A, P);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "KSPSetOperators");
}
//-----------------------------------------------------------------------------
int petsc::KrylovSolver::solve(Vec x, const Vec b, bool transpose) const
{
  common::Timer timer("PETSc Krylov solver");
  assert(x);
  assert(b);

  // Get PETSc operators
  Mat _A, _P;
  KSPGetOperators(_ksp, &_A, &_P);
  assert(_A);

  PetscErrorCode ierr;

  // Solve linear system
  spdlog::info("PETSc Krylov solver starting to solve system.");

  // Solve system
  if (!transpose)
  {
    ierr = KSPSolve(_ksp, b, x);
    if (ierr != 0)
      petsc::error(ierr, __FILE__, "KSPSolve");
  }
  else
  {
    ierr = KSPSolveTranspose(_ksp, b, x);
    if (ierr != 0)
      petsc::error(ierr, __FILE__, "KSPSolve");
  }

  // Get the number of iterations
  PetscInt num_iterations = 0;
  ierr = KSPGetIterationNumber(_ksp, &num_iterations);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "KSPGetIterationNumber");

  // Check if the solution converged and print error/warning if not
  // converged
  KSPConvergedReason reason;
  ierr = KSPGetConvergedReason(_ksp, &reason);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "KSPGetConvergedReason");
  if (reason < 0)
  {
    /*
    // Get solver residual norm
    double rnorm = 0.0;
    ierr = KSPGetResidualNorm(_ksp, &rnorm);
    if (ierr != 0) error(ierr, __FILE__, "KSPGetResidualNorm");
    const char *reason_str = KSPConvergedReasons[reason];
    bool error_on_nonconvergence =
    this->parameters["error_on_nonconvergence"].is_set() ?
    this->parameters["error_on_nonconvergence"] : true;
    if (error_on_nonconvergence)
    {
      log::dolfin_error("PETScKrylovSolver.cpp",
                   "solve linear system using PETSc Krylov solver",
                   "Solution failed to converge in %i iterations (PETSc reason
    %s, residual norm ||r|| = %e)",
                   static_cast<int>(num_iterations), reason_str, rnorm);
    }
    else
    {
      log::warning("Krylov solver did not converge in %i iterations (PETSc
    reason %s,
    residual norm ||r|| = %e).",
              num_iterations, reason_str, rnorm);
    }
    */
  }

  // Report results
  // if (report && dolfinx::MPI::rank(this->comm()) == 0)
  //  write_report(num_iterations, reason);

  return num_iterations;
}
//-----------------------------------------------------------------------------
void petsc::KrylovSolver::set_options_prefix(std::string options_prefix)
{
  // Set options prefix
  assert(_ksp);
  PetscErrorCode ierr = KSPSetOptionsPrefix(_ksp, options_prefix.c_str());
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "KSPSetOptionsPrefix");
}
//-----------------------------------------------------------------------------
std::string petsc::KrylovSolver::get_options_prefix() const
{
  assert(_ksp);
  const char* prefix = nullptr;
  PetscErrorCode ierr = KSPGetOptionsPrefix(_ksp, &prefix);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "KSPGetOptionsPrefix");
  return std::string(prefix);
}
//-----------------------------------------------------------------------------
void petsc::KrylovSolver::set_from_options() const
{
  assert(_ksp);
  PetscErrorCode ierr = KSPSetFromOptions(_ksp);
  if (ierr != 0)
    petsc::error(ierr, __FILE__, "KSPSetFromOptions");
}
//-----------------------------------------------------------------------------
KSP petsc::KrylovSolver::ksp() const { return _ksp; }
//-----------------------------------------------------------------------------

#endif
