// Copyright (C) 2007-2023 Garth N. Wells and Francesco Ballarin
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <span>
#include <dolfinx/la/SparsityPattern.h>

namespace dolfinx::la
{
class SparsityPattern;
}

namespace dolfinx::fem
{
class DofMap;

/// Support for building sparsity patterns from degree-of-freedom maps.
namespace sparsitybuild
{
/// @brief Iterate over cells and insert entries into sparsity pattern.
///
/// @param[in,out] pattern The sparsity pattern to insert into
/// @param[in] cells The cell indices
/// @param[in] dofmaps_list Dofmaps list to used in building the sparsity pattern
/// @param[in] dofmaps_bounds Dofmaps bounds to used in building the sparsity pattern
/// @note The sparsity pattern is not finalised
void cells(la::SparsityPattern& pattern, std::span<const std::int32_t> cells,
           std::array<std::span<const std::int32_t>, 2> dofmaps_list,
           std::array<std::span<const std::size_t>, 2> dofmaps_bounds);

/// @brief Iterate over interior facets and insert entries into sparsity
/// pattern.
///
/// @param[in,out] pattern Sparsity pattern to insert into
/// @param[in] facets Facets as `(cell0, cell1)` pairs for each facet.
/// @param[in] dofmaps_list The dofmap list to use in building the sparsity
/// pattern.
/// @param[in] dofmaps_bounds The dofmap bounds to use in building the sparsity
/// pattern.
///
/// @note The sparsity pattern is not finalised.
void interior_facets(
    la::SparsityPattern& pattern, std::span<const std::int32_t> facets,
    std::array<std::span<const std::int32_t>, 2> dofmaps_list,
    std::array<std::span<const std::size_t>, 2> dofmaps_bounds);

} // namespace sparsitybuild
} // namespace dolfinx::fem
