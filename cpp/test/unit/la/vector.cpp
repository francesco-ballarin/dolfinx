// Copyright (C) 2021 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
//
// Unit tests for Distributed la::Vector

#include <catch.hpp>
#include <dolfinx.h>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/la/Vector.h>

using namespace dolfinx;

namespace
{

void test_vector()
{
  const int mpi_size = dolfinx::MPI::size(MPI_COMM_WORLD);
  const int mpi_rank = dolfinx::MPI::rank(MPI_COMM_WORLD);
  constexpr int size_local = 100;

  // Create some ghost entries on next process
  int num_ghosts = (mpi_size - 1) * 3;
  std::vector<std::int64_t> ghosts(num_ghosts);
  for (int i = 0; i < num_ghosts; ++i)
    ghosts[i] = (mpi_rank + 1) % mpi_size * size_local + i;

  const std::vector<int> global_ghost_owner(ghosts.size(),
                                            (mpi_rank + 1) % mpi_size);

  // Create an IndexMap
  const auto index_map = std::make_shared<common::IndexMap>(
      MPI_COMM_WORLD, size_local,
      dolfinx::MPI::compute_graph_edges(
          MPI_COMM_WORLD,
          std::set<int>(global_ghost_owner.begin(), global_ghost_owner.end())),
      ghosts, global_ghost_owner);

  la::Vector<PetscScalar> v(index_map, 1);
  std::fill(v.mutable_array().begin(), v.mutable_array().end(), 1.0);

  const double norm2 = v.squared_norm();
  CHECK(norm2 == mpi_size * size_local);

  std::fill(v.mutable_array().begin(), v.mutable_array().end(), mpi_rank);

  const double sumn2
      = size_local * (mpi_size - 1) * mpi_size * (2 * mpi_size - 1) / 6;
  CHECK(v.squared_norm() == sumn2);
  CHECK(v.norm(la::Norm::l2) == std::sqrt(sumn2));
  CHECK(la::inner_product(v, v) == sumn2);
  CHECK(v.norm(la::Norm::linf) == static_cast<PetscScalar>(mpi_size - 1));
}

} // namespace

TEST_CASE("Linear Algebra Vector", "[la_vector]")
{
  CHECK_NOTHROW(test_vector());
}
