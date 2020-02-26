// Copyright (C) 2018-2020 Garth N. Wells and Francesco Ballarin
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "petsc.h"
#include "assembler.h"
#include "sparsitybuild.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/la/PETScMatrix.h>
#include <dolfinx/la/PETScVector.h>
#include <dolfinx/la/SparsityPattern.h>
#include <xtl/xspan.hpp>

using namespace dolfinx;

//-----------------------------------------------------------------------------
Mat dolfinx::fem::create_matrix(
    const mesh::Mesh& mesh,
    std::array<std::reference_wrapper<const common::IndexMap>, 2> index_maps,
    const std::array<int, 2> index_maps_bs,
    const std::set<fem::IntegralType>& integral_types,
    std::array<const graph::AdjacencyList<std::int32_t>*, 2> dofmaps,
    const std::string& matrix_type)
{
  // Build sparsity pattern
  const std::array<std::shared_ptr<const common::IndexMap>, 2> index_maps_shared_ptr
    {{std::shared_ptr<const common::IndexMap>(&index_maps[0].get(), [](const common::IndexMap*){}),
      std::shared_ptr<const common::IndexMap>(&index_maps[1].get(), [](const common::IndexMap*){})}};
  la::SparsityPattern pattern(mesh.mpi_comm(), index_maps_shared_ptr, index_maps_bs);
  if (integral_types.count(fem::IntegralType::cell) > 0)
  {
    sparsitybuild::cells(pattern, mesh.topology(), dofmaps);
  }
  if (integral_types.count(fem::IntegralType::interior_facet) > 0
      or integral_types.count(fem::IntegralType::exterior_facet) > 0)
  {
    // FIXME: cleanup these calls? Some of the happen internally again.
    const int tdim = mesh.topology().dim();
    mesh.topology_mutable().create_entities(tdim - 1);
    mesh.topology_mutable().create_connectivity(tdim - 1, tdim);
    if (integral_types.count(fem::IntegralType::interior_facet) > 0)
    {
      sparsitybuild::interior_facets(pattern, mesh.topology(), dofmaps);
    }
    if (integral_types.count(fem::IntegralType::exterior_facet) > 0)
    {
      sparsitybuild::exterior_facets(pattern, mesh.topology(), dofmaps);
    }
  }

  // Finalise communication
  pattern.assemble();

  return la::create_petsc_matrix(mesh.mpi_comm(), pattern, matrix_type);
}
//-----------------------------------------------------------------------------
Mat fem::create_matrix_block(
    const mesh::Mesh& mesh,
    std::array<std::vector<std::reference_wrapper<const common::IndexMap>>, 2> index_maps,
    const std::array<std::vector<int>, 2> index_maps_bs,
    const std::vector<std::vector<std::set<fem::IntegralType>>>& integral_types,
    const std::array<std::vector<const graph::AdjacencyList<std::int32_t>*>, 2>& dofmaps,
    const std::string& matrix_type)
{
  std::size_t rows = index_maps[0].size();
  assert(index_maps_bs[0].size() == rows);
  assert(integral_types.size() == rows);
  assert(dofmaps[0].size() == rows);
  std::size_t cols = index_maps[1].size();
  assert(index_maps_bs[1].size() == cols);
  assert(std::all_of(integral_types.begin(), integral_types.end(),
    [&cols](const std::vector<std::set<fem::IntegralType>>& integral_types_){
    return integral_types_.size() == cols;}));
  assert(index_maps_bs[1].size() == cols);
  assert(dofmaps[1].size() == cols);

  // Build sparsity pattern for each block
  std::vector<std::vector<std::unique_ptr<la::SparsityPattern>>> patterns(
      rows);
  for (std::size_t row = 0; row < rows; ++row)
  {
    for (std::size_t col = 0; col < cols; ++col)
    {
      if (integral_types[row][col].size() > 0)
      {
        // Create sparsity pattern for block
        const std::array<std::shared_ptr<const common::IndexMap>, 2> index_maps_row_col
          {{std::shared_ptr<const common::IndexMap>(&index_maps[0][row].get(), [](const common::IndexMap*){}),
            std::shared_ptr<const common::IndexMap>(&index_maps[1][col].get(), [](const common::IndexMap*){})}};
        const std::array<int, 2> index_maps_bs_row_col
          {{index_maps_bs[0][row], index_maps_bs[1][col]}};
        patterns[row].push_back(
            std::make_unique<la::SparsityPattern>(mesh.mpi_comm(), index_maps_row_col, index_maps_bs_row_col));
        assert(patterns[row].back());

        auto& sp = patterns[row].back();
        assert(sp);

        // Build sparsity pattern for block
        if (integral_types[row][col].count(IntegralType::cell) > 0)
        {
          sparsitybuild::cells(*sp, mesh.topology(),
                               {{dofmaps[0][row], dofmaps[1][col]}});
        }
        if (integral_types[row][col].count(IntegralType::interior_facet) > 0
            or integral_types[row][col].count(IntegralType::exterior_facet) > 0)
        {
          // FIXME: cleanup these calls? Some of the happen internally again.
          const int tdim = mesh.topology().dim();
          mesh.topology_mutable().create_entities(tdim - 1);
          mesh.topology_mutable().create_connectivity(tdim - 1, tdim);
          if (integral_types[row][col].count(IntegralType::interior_facet) > 0)
          {
            sparsitybuild::interior_facets(*sp, mesh.topology(),
                                           {{dofmaps[0][row], dofmaps[1][col]}});
          }
          if (integral_types[row][col].count(IntegralType::exterior_facet) > 0)
          {
            sparsitybuild::exterior_facets(*sp, mesh.topology(),
                                           {{dofmaps[0][row], dofmaps[1][col]}});
          }
        }
      }
      else
        patterns[row].push_back(nullptr);
    }
  }

  // Compute offsets for the fields
  std::array<std::vector<std::pair<
                 std::reference_wrapper<const common::IndexMap>, int>>,
             2>
      maps_and_bs;
  for (std::size_t d = 0; d < 2; ++d)
  {
    for (std::size_t f = 0; f < index_maps[d].size(); ++f)
    {
      maps_and_bs[d].push_back(
          {index_maps[d][f], index_maps_bs[d][f]});
    }
  }
  // FIXME: This is computed again inside the SparsityPattern
  // constructor, but we also need to outside to build the PETSc
  // local-to-global map. Compute outside and pass into SparsityPattern
  // constructor.
  auto [rank_offset, local_offset, ghosts, owner]
        = common::stack_index_maps(maps_and_bs[0]);

  // Create merged sparsity pattern
  std::vector<std::vector<const la::SparsityPattern*>> p(rows);
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t col = 0; col < cols; ++col)
      p[row].push_back(patterns[row][col].get());
  la::SparsityPattern pattern(mesh.mpi_comm(), p, maps_and_bs, index_maps_bs);
  pattern.assemble();

  // FIXME: Add option to pass customised local-to-global map to PETSc
  // Mat constructor

  // Initialise matrix
  Mat A = la::create_petsc_matrix(mesh.mpi_comm(), pattern, matrix_type);

  // Create row and column local-to-global maps (field0, field1, field2,
  // etc), i.e. ghosts of field0 appear before owned indices of field1
  std::array<std::vector<PetscInt>, 2> _maps;
  for (int d = 0; d < 2; ++d)
  {
    for (std::size_t f = 0; f < index_maps[d].size(); ++f)
    {
      const common::IndexMap& map = index_maps[d][f].get();
      const int bs = index_maps_bs[d][f];
      const std::int32_t size_local = bs * map.size_local();
      const std::vector<std::int64_t> global = map.global_indices();
      for (std::int32_t i = 0; i < size_local; ++i)
        _maps[d].push_back(i + rank_offset + local_offset[f]);
      for (std::size_t i = size_local; i < bs * global.size(); ++i)
        _maps[d].push_back(ghosts[f][i - size_local]);
    }
  }

  // Create PETSc local-to-global map/index sets and attach to matrix
  ISLocalToGlobalMapping petsc_local_to_global0;
  ISLocalToGlobalMappingCreate(MPI_COMM_SELF, 1, _maps[0].size(),
                               _maps[0].data(), PETSC_COPY_VALUES,
                               &petsc_local_to_global0);
  if (dofmaps[0] == dofmaps[1])
  {
    MatSetLocalToGlobalMapping(A, petsc_local_to_global0,
                               petsc_local_to_global0);
    ISLocalToGlobalMappingDestroy(&petsc_local_to_global0);
  }
  else
  {
    ISLocalToGlobalMapping petsc_local_to_global1;
    ISLocalToGlobalMappingCreate(MPI_COMM_SELF, 1, _maps[1].size(),
                                 _maps[1].data(), PETSC_COPY_VALUES,
                                 &petsc_local_to_global1);
    MatSetLocalToGlobalMapping(A, petsc_local_to_global0,
                               petsc_local_to_global1);
    MatSetLocalToGlobalMapping(A, petsc_local_to_global0,
                               petsc_local_to_global1);
    ISLocalToGlobalMappingDestroy(&petsc_local_to_global0);
    ISLocalToGlobalMappingDestroy(&petsc_local_to_global1);
  }

  return A;
}
//-----------------------------------------------------------------------------
Mat fem::create_matrix_nest(
    const mesh::Mesh& mesh,
    std::array<std::vector<std::reference_wrapper<const common::IndexMap>>, 2> index_maps,
    const std::array<std::vector<int>, 2> index_maps_bs,
    const std::vector<std::vector<std::set<fem::IntegralType>>>& integral_types,
    const std::array<std::vector<const graph::AdjacencyList<std::int32_t>*>, 2>& dofmaps,
    const std::vector<std::vector<std::string>>& matrix_types)
{
  std::size_t rows = index_maps[0].size();
  assert(index_maps_bs[0].size() == rows);
  assert(integral_types.size() == rows);
  assert(dofmaps[0].size() == rows);
  std::size_t cols = index_maps[1].size();
  assert(index_maps_bs[1].size() == cols);
  assert(std::all_of(integral_types.begin(), integral_types.end(),
    [&cols](const std::vector<std::set<fem::IntegralType>>& form_integrals_types_){
    return form_integrals_types_.size() == cols;}));
  assert(dofmaps[1].size() == cols);
  std::vector<std::vector<std::string>> _matrix_types(
      rows, std::vector<std::string>(cols));
  if (!matrix_types.empty())
    _matrix_types = matrix_types;

  // Loop over each form and create matrix
  std::vector<Mat> mats(rows * cols, nullptr);
  for (std::size_t i = 0; i < rows; ++i)
  {
    for (std::size_t j = 0; j < cols; ++j)
    {
      if (integral_types[i][j].size() > 0)
      {
        mats[i * cols + j] = create_matrix(
          mesh, {{index_maps[0][i], index_maps[1][j]}},
          {{index_maps_bs[0][i], index_maps_bs[1][j]}},
          integral_types[i][j],
          {{dofmaps[0][i], dofmaps[1][j]}},
          _matrix_types[i][j]);
      }
    }
  }

  // Initialise block (MatNest) matrix
  Mat A;
  MatCreate(mesh.mpi_comm(), &A);
  MatSetType(A, MATNEST);
  MatNestSetSubMats(A, rows, nullptr, cols, nullptr, mats.data());
  MatSetUp(A);

  // De-reference Mat objects
  for (std::size_t i = 0; i < mats.size(); ++i)
  {
    if (mats[i])
      MatDestroy(&mats[i]);
  }

  return A;
}
//-----------------------------------------------------------------------------
Vec fem::create_vector_block(
    const std::vector<
        std::pair<std::reference_wrapper<const common::IndexMap>, int>>& maps)
{
  // FIXME: handle constant block size > 1

  auto [rank_offset, local_offset, ghosts_new, ghost_new_owners]
      = common::stack_index_maps(maps);
  std::int32_t local_size = local_offset.back();

  std::vector<std::int64_t> ghosts;
  for (auto& sub_ghost : ghosts_new)
    ghosts.insert(ghosts.end(), sub_ghost.begin(), sub_ghost.end());

  std::vector<int> ghost_owners;
  for (auto& sub_owner : ghost_new_owners)
    ghost_owners.insert(ghost_owners.end(), sub_owner.begin(), sub_owner.end());

  std::vector<int> dest_ranks;
  for (auto& map : maps)
  {
    const auto [_, ranks] = dolfinx::MPI::neighbors(
        map.first.get().comm(common::IndexMap::Direction::forward));
    dest_ranks.insert(dest_ranks.end(), ranks.begin(), ranks.end());
  }
  std::sort(dest_ranks.begin(), dest_ranks.end());
  dest_ranks.erase(std::unique(dest_ranks.begin(), dest_ranks.end()),
                   dest_ranks.end());

  // Create map for combined problem, and create vector
  common::IndexMap index_map(
      maps[0].first.get().comm(common::IndexMap::Direction::forward),
      local_size, dest_ranks, ghosts, ghost_owners);

  return la::create_petsc_vector(index_map, 1);
}
//-----------------------------------------------------------------------------
Vec fem::create_vector_nest(
    const std::vector<
        std::pair<std::reference_wrapper<const common::IndexMap>, int>>& maps)
{
  assert(!maps.empty());

  // Loop over each form and create vector
  std::vector<std::shared_ptr<la::PETScVector>> vecs;
  std::vector<Vec> petsc_vecs;
  for (auto& map : maps)
  {
    vecs.push_back(std::make_shared<la::PETScVector>(map.first, map.second));
    petsc_vecs.push_back(vecs.back()->vec());
  }

  // Create nested (VecNest) vector
  Vec y;
  VecCreateNest(vecs[0]->mpi_comm(), petsc_vecs.size(), nullptr,
                petsc_vecs.data(), &y);
  return y;
}
//-----------------------------------------------------------------------------
void fem::assemble_vector_petsc(Vec b, const Form<PetscScalar>& L,
                                const xtl::span<const PetscScalar>& constants,
                                const array2d<PetscScalar>& coeffs)
{
  Vec b_local;
  VecGhostGetLocalForm(b, &b_local);
  PetscInt n = 0;
  VecGetSize(b_local, &n);
  PetscScalar* array = nullptr;
  VecGetArray(b_local, &array);
  xtl::span<PetscScalar> _b(array, n);
  fem::assemble_vector<PetscScalar>(_b, L, constants, coeffs);
  VecRestoreArray(b_local, &array);
  VecGhostRestoreLocalForm(b, &b_local);
}
//-----------------------------------------------------------------------------
void fem::assemble_vector_petsc(Vec b, const Form<PetscScalar>& L)
{
  Vec b_local;
  VecGhostGetLocalForm(b, &b_local);
  PetscInt n = 0;
  VecGetSize(b_local, &n);
  PetscScalar* array = nullptr;
  VecGetArray(b_local, &array);
  xtl::span<PetscScalar> _b(array, n);
  fem::assemble_vector<PetscScalar>(_b, L);
  VecRestoreArray(b_local, &array);
  VecGhostRestoreLocalForm(b, &b_local);
}
//-----------------------------------------------------------------------------
void fem::apply_lifting_petsc(
    Vec b, const std::vector<std::shared_ptr<const Form<PetscScalar>>>& a,
    const std::vector<xtl::span<const PetscScalar>>& constants,
    const std::vector<const array2d<PetscScalar>*>& coeffs,
    const std::vector<
        std::vector<std::shared_ptr<const DirichletBC<PetscScalar>>>>& bcs1,
    const std::vector<Vec>& x0, double scale)
{
  Vec b_local;
  VecGhostGetLocalForm(b, &b_local);
  PetscInt n = 0;
  VecGetSize(b_local, &n);
  PetscScalar* array = nullptr;
  VecGetArray(b_local, &array);
  xtl::span<PetscScalar> _b(array, n);

  if (x0.empty())
    fem::apply_lifting<PetscScalar>(_b, a, constants, coeffs, bcs1, {}, scale);
  else
  {
    std::vector<xtl::span<const PetscScalar>> x0_ref;
    std::vector<Vec> x0_local(a.size());
    std::vector<const PetscScalar*> x0_array(a.size());
    for (std::size_t i = 0; i < a.size(); ++i)
    {
      assert(x0[i]);
      VecGhostGetLocalForm(x0[i], &x0_local[i]);
      PetscInt n = 0;
      VecGetSize(x0_local[i], &n);
      VecGetArrayRead(x0_local[i], &x0_array[i]);
      x0_ref.emplace_back(x0_array[i], n);
    }

    std::vector x0_tmp(x0_ref.begin(), x0_ref.end());
    fem::apply_lifting<PetscScalar>(_b, a, constants, coeffs, bcs1, x0_tmp,
                                    scale);

    for (std::size_t i = 0; i < x0_local.size(); ++i)
    {
      VecRestoreArrayRead(x0_local[i], &x0_array[i]);
      VecGhostRestoreLocalForm(x0[i], &x0_local[i]);
    }
  }

  VecRestoreArray(b_local, &array);
  VecGhostRestoreLocalForm(b, &b_local);
}
//-----------------------------------------------------------------------------
void fem::apply_lifting_petsc(
    Vec b, const std::vector<std::shared_ptr<const Form<PetscScalar>>>& a,
    const std::vector<
        std::vector<std::shared_ptr<const DirichletBC<PetscScalar>>>>& bcs1,
    const std::vector<Vec>& x0, double scale)
{
  Vec b_local;
  VecGhostGetLocalForm(b, &b_local);
  PetscInt n = 0;
  VecGetSize(b_local, &n);
  PetscScalar* array = nullptr;
  VecGetArray(b_local, &array);
  xtl::span<PetscScalar> _b(array, n);

  if (x0.empty())
    fem::apply_lifting<PetscScalar>(_b, a, bcs1, {}, scale);
  else
  {
    std::vector<xtl::span<const PetscScalar>> x0_ref;
    std::vector<Vec> x0_local(a.size());
    std::vector<const PetscScalar*> x0_array(a.size());
    for (std::size_t i = 0; i < a.size(); ++i)
    {
      assert(x0[i]);
      VecGhostGetLocalForm(x0[i], &x0_local[i]);
      PetscInt n = 0;
      VecGetSize(x0_local[i], &n);
      VecGetArrayRead(x0_local[i], &x0_array[i]);
      x0_ref.emplace_back(x0_array[i], n);
    }

    std::vector x0_tmp(x0_ref.begin(), x0_ref.end());
    fem::apply_lifting<PetscScalar>(_b, a, bcs1, x0_tmp, scale);

    for (std::size_t i = 0; i < x0_local.size(); ++i)
    {
      VecRestoreArrayRead(x0_local[i], &x0_array[i]);
      VecGhostRestoreLocalForm(x0[i], &x0_local[i]);
    }
  }

  VecRestoreArray(b_local, &array);
  VecGhostRestoreLocalForm(b, &b_local);
}
//-----------------------------------------------------------------------------
void fem::set_bc_petsc(
    Vec b,
    const std::vector<std::shared_ptr<const DirichletBC<PetscScalar>>>& bcs,
    const Vec x0, double scale)
{
  PetscInt n = 0;
  VecGetLocalSize(b, &n);
  PetscScalar* array = nullptr;
  VecGetArray(b, &array);
  xtl::span<PetscScalar> _b(array, n);
  if (x0)
  {
    Vec x0_local;
    VecGhostGetLocalForm(x0, &x0_local);
    PetscInt n = 0;
    VecGetSize(x0_local, &n);
    const PetscScalar* array = nullptr;
    VecGetArrayRead(x0_local, &array);
    xtl::span<const PetscScalar> _x0(array, n);
    fem::set_bc<PetscScalar>(_b, bcs, _x0, scale);
    VecRestoreArrayRead(x0_local, &array);
    VecGhostRestoreLocalForm(x0, &x0_local);
  }
  else
    fem::set_bc<PetscScalar>(_b, bcs, scale);

  VecRestoreArray(b, &array);
}
//-----------------------------------------------------------------------------
