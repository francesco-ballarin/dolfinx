# Copyright (C) 2018-2022 Garth N. Wells and Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Assembly functions into PETSc objects for variational forms.

Functions in this module generally apply functions in :mod:`dolfinx.fem`
to PETSc linear algebra objects and handle any PETSc-specific
preparation."""

# mypy: ignore-errors

from __future__ import annotations

import contextlib
import functools
import typing

from petsc4py import PETSc

# ruff: noqa: E402
import dolfinx

assert dolfinx.has_petsc4py

import numpy as np

import dolfinx.cpp as _cpp
import ufl
from dolfinx import la
from dolfinx.cpp.fem import pack_coefficients as _pack_coefficients
from dolfinx.cpp.fem import pack_constants as _pack_constants
from dolfinx.cpp.fem.petsc import discrete_curl as _discrete_curl
from dolfinx.cpp.fem.petsc import discrete_gradient as _discrete_gradient
from dolfinx.cpp.fem.petsc import interpolation_matrix as _interpolation_matrix
from dolfinx.fem import assemble as _assemble
from dolfinx.fem.bcs import DirichletBC
from dolfinx.fem.bcs import bcs_by_block as _bcs_by_block
from dolfinx.fem.forms import Form
from dolfinx.fem.forms import extract_function_spaces as _extract_spaces
from dolfinx.fem.forms import form as _create_form
from dolfinx.fem.function import Function as _Function
from dolfinx.fem.function import FunctionSpace as _FunctionSpace
from dolfinx.la import create_petsc_vector

__all__ = [
    "LinearProblem",
    "NonlinearProblem",
    "apply_lifting",
    "apply_lifting_nest",
    "assemble_matrix",
    "assemble_matrix_block",
    "assemble_matrix_nest",
    "assemble_vector",
    "assemble_vector_block",
    "assemble_vector_nest",
    "create_matrix",
    "create_matrix_block",
    "create_matrix_nest",
    "create_vector",
    "create_vector_block",
    "create_vector_nest",
    "discrete_curl",
    "discrete_gradient",
    "interpolation_matrix",
    "set_bc",
    "set_bc_nest",
]


def _get_block_function_spaces(a):
    rows = len(a)
    cols = len(a[0])
    assert all(len(a_i) == cols for a_i in a)
    assert all(a[i][j] is None or a[i][j].rank == 2 for i in range(rows) for j in range(cols))
    function_spaces_0 = list()
    for i in range(rows):
        function_spaces_0_i = None
        for j in range(cols):
            if a[i][j] is not None:
                function_spaces_0_i = a[i][j].function_spaces[0]
                break
        assert function_spaces_0_i is not None
        function_spaces_0.append(function_spaces_0_i)
    function_spaces_1 = list()
    for j in range(cols):
        function_spaces_1_j = None
        for i in range(rows):
            if a[i][j] is not None:
                function_spaces_1_j = a[i][j].function_spaces[1]
                break
        assert function_spaces_1_j is not None
        function_spaces_1.append(function_spaces_1_j)
    function_spaces = [function_spaces_0, function_spaces_1]
    assert all(a[i][j] is None or a[i][j].function_spaces[0] == function_spaces[0][i]
               for i in range(rows) for j in range(cols))
    assert all(a[i][j] is None or a[i][j].function_spaces[1] == function_spaces[1][j]
               for i in range(rows) for j in range(cols))
    return function_spaces


def _same_dofmap(dofmap1, dofmap2):
    try:
        dofmap1 = dofmap1._cpp_object
    except AttributeError:
        pass

    try:
        dofmap2 = dofmap2._cpp_object
    except AttributeError:
        pass

    return dofmap1 == dofmap2


# -- Vector instantiation ----------------------------------------------------


def create_vector(L: Form) -> PETSc.Vec:
    """Create a PETSc vector that is compatible with a linear form.

    Note:
        Due to subtle issues in the interaction between petsc4py memory management
        and the Python garbage collector, it is recommended that the method ``PETSc.Vec.destroy()``
        is called on the returned object once the object is no longer required. Note that
        ``PETSc.Vec.destroy()`` is collective over the object's MPI communicator.

    Args:
        L: A linear form.

    Returns:
        A PETSc vector with a layout that is compatible with ``L``.
    """
    dofmap = L.function_spaces[0].dofmap
    return create_petsc_vector(dofmap.index_map, dofmap.index_map_bs)


def create_vector_block(L: list[Form]) -> PETSc.Vec:
    """Create a PETSc vector (blocked) that is compatible with a list of linear forms.

    Note:
        Due to subtle issues in the interaction between petsc4py memory management
        and the Python garbage collector, it is recommended that the method ``PETSc.Vec.destroy()``
        is called on the returned object once the object is no longer required. Note that
        ``PETSc.Vec.destroy()`` is collective over the object's MPI communicator.

    Args:
        L: List of linear forms.

    Returns:
        A PETSc vector with a layout that is compatible with ``L``.

    """
    maps = [
        (form.function_spaces[0].dofmap.index_map, form.function_spaces[0].dofmap.index_map_bs)
        for form in L
    ]
    return _cpp.fem.petsc.create_vector_block(maps)


def create_vector_nest(L: list[Form]) -> PETSc.Vec:
    """Create a PETSc nested vector (``VecNest``) that is compatible
    with a list of linear forms.

    Args:
        L: List of linear forms.

    Returns:
        A PETSc nested vector (``VecNest``) with a layout that is
        compatible with ``L``.
    """
    maps = [
        (form.function_spaces[0].dofmap.index_map, form.function_spaces[0].dofmap.index_map_bs)
        for form in L
    ]
    return _cpp.fem.petsc.create_vector_nest(maps)


# -- Matrix instantiation ----------------------------------------------------


def create_matrix(a: Form,
                  restriction: typing.Optional[tuple[_cpp.fem.DofMapRestriction]] = None,
                  mat_type=None) -> PETSc.Mat:
    """Create a PETSc matrix that is compatible with a bilinear form.

    Note:
        Due to subtle issues in the interaction between petsc4py memory
        management and the Python garbage collector, it is recommended
        that the method ``PETSc.Mat.destroy()`` is called on the
        returned object once the object is no longer required. Note that
        ``PETSc.Mat.destroy()`` is collective over the object's MPI
        communicator.

    Args:
        a: A bilinear form.
        mat_type: The PETSc matrix type (``MatType``).

    Returns:
        A PETSc matrix with a layout that is compatible with ``a``.
    """
    assert a.rank == 2
    function_spaces = a.function_spaces
    assert all(function_space.mesh == a.mesh for function_space in function_spaces)
    if restriction is None:
        index_maps = [function_space.dofmap.index_map for function_space in function_spaces]
        index_maps_bs = [function_space.dofmap.index_map_bs for function_space in function_spaces]
        dofmaps_list = [function_space.dofmap.map() for function_space in function_spaces]
        dofmaps_bounds = [
            np.arange(dofmap_list.shape[0] + 1, dtype=np.uint64) * dofmap_list.shape[1] for dofmap_list in dofmaps_list]
    else:
        assert len(restriction) == 2
        index_maps = [restriction_.index_map for restriction_ in restriction]
        index_maps_bs = [restriction_.index_map_bs for restriction_ in restriction]
        dofmaps_list = [restriction_.map()[0] for restriction_ in restriction]
        dofmaps_bounds = [restriction_.map()[1] for restriction_ in restriction]
    if mat_type is not None:
        return _cpp.fem.petsc.create_matrix(
            a._cpp_object, index_maps, index_maps_bs, dofmaps_list, dofmaps_bounds, mat_type)
    else:
        return _cpp.fem.petsc.create_matrix(
            a._cpp_object, index_maps, index_maps_bs, dofmaps_list, dofmaps_bounds)


def _create_matrix_block_or_nest(a, restriction, mat_type, cpp_create_function):
    function_spaces = _get_block_function_spaces(a)
    rows, cols = len(function_spaces[0]), len(function_spaces[1])
    mesh = None
    for j in range(cols):
        for i in range(rows):
            if a[i][j] is not None:
                mesh = a[i][j].mesh
                break
    assert mesh is not None
    assert all(a[i][j] is None or a[i][j].mesh == mesh for i in range(rows) for j in range(cols))
    assert all(function_space.mesh == mesh for function_space in function_spaces[0])
    assert all(function_space.mesh == mesh for function_space in function_spaces[1])
    if restriction is None:
        index_maps = (
            [function_spaces[0][i].dofmap.index_map for i in range(rows)],
            [function_spaces[1][j].dofmap.index_map for j in range(cols)])
        index_maps_bs = (
            [function_spaces[0][i].dofmap.index_map_bs for i in range(rows)],
            [function_spaces[1][j].dofmap.index_map_bs for j in range(cols)])
        dofmaps_list = (
            [function_spaces[0][i].dofmap.map() for i in range(rows)],
            [function_spaces[1][j].dofmap.map() for j in range(cols)])
        dofmaps_bounds = (
            [np.arange(dofmaps_list[0][i].shape[0] + 1, dtype=np.uint64) * dofmaps_list[0][i].shape[1]
             for i in range(rows)],
            [np.arange(dofmaps_list[1][j].shape[0] + 1, dtype=np.uint64) * dofmaps_list[1][j].shape[1]
             for j in range(cols)])
    else:
        assert len(restriction) == 2
        assert len(restriction[0]) == rows
        assert len(restriction[1]) == cols
        index_maps = (
            [restriction[0][i].index_map for i in range(rows)],
            [restriction[1][j].index_map for j in range(cols)])
        index_maps_bs = (
            [restriction[0][i].index_map_bs for i in range(rows)],
            [restriction[1][j].index_map_bs for j in range(cols)])
        dofmaps_list = (
            [restriction[0][i].map()[0] for i in range(rows)],
            [restriction[1][j].map()[0] for j in range(cols)])
        dofmaps_bounds = (
            [restriction[0][i].map()[1] for i in range(rows)],
            [restriction[1][j].map()[1] for j in range(cols)])
    a_cpp = [[None if form is None else form._cpp_object for form in forms] for forms in a]
    if mat_type is not None:
        return cpp_create_function(a_cpp, index_maps, index_maps_bs, dofmaps_list, dofmaps_bounds, mat_type)
    else:
        return cpp_create_function(a_cpp, index_maps, index_maps_bs, dofmaps_list, dofmaps_bounds)


def create_matrix_block(a: list[list[Form]],
                        restriction: typing.Optional[tuple[
                                                     list[_cpp.fem.DofMapRestriction]]] = None,
                        mat_type=None) -> PETSc.Mat:
    """Create a PETSc matrix that is compatible with a rectangular array of bilinear forms.

    Note:
        Due to subtle issues in the interaction between petsc4py memory
        management and the Python garbage collector, it is recommended
        that the method ``PETSc.Mat.destroy()`` is called on the
        returned object once the object is no longer required. Note that
        ``PETSc.Mat.destroy()`` is collective over the object's MPI
        communicator.

    Args:
        a: Rectangular array of bilinear forms.

    Returns:
        A PETSc matrix with a blocked layout that is compatible with
        ``a``.
    """
    return _create_matrix_block_or_nest(a, restriction, mat_type, _cpp.fem.petsc.create_matrix_block)


def create_matrix_nest(a: list[list[Form]],
                       restriction: typing.Optional[tuple[
                                                    list[_cpp.fem.DofMapRestriction]]] = None,
                       mat_types=None) -> PETSc.Mat:
    """Create a PETSc matrix (``MatNest``) that is compatible with a rectangular array of bilinear forms.

    Note:
        Due to subtle issues in the interaction between petsc4py memory
        management and the Python garbage collector, it is recommended
        that the method ``PETSc.Mat.destroy()`` is called on the
        returned object once the object is no longer required. Note that
        ``PETSc.Mat.destroy()`` is collective over the object's MPI
        communicator.

    Args:
        a: Rectangular array of bilinear forms.

    Returns:
        A PETSc matrix (``MatNest``) that is compatible with ``a``.
    """
    return _create_matrix_block_or_nest(a, restriction, mat_types, _cpp.fem.petsc.create_matrix_nest)


# -- Vector assembly ---------------------------------------------------------


@functools.singledispatch
def assemble_vector(L: typing.Any, constants=None, coeffs=None) -> PETSc.Vec:
    """Assemble linear form into a new PETSc vector.

    Note:
        The returned vector is not finalised, i.e. ghost values are not
        accumulated on the owning processes.

    Note:
        Due to subtle issues in the interaction between petsc4py memory management
        and the Python garbage collector, it is recommended that the method ``PETSc.Vec.destroy()``
        is called on the returned object once the object is no longer required. Note that
        ``PETSc.Vec.destroy()`` is collective over the object's MPI communicator.

    Args:
        L: A linear form.

    Returns:
        An assembled vector.
    """
    b = create_petsc_vector(
        L.function_spaces[0].dofmap.index_map, L.function_spaces[0].dofmap.index_map_bs
    )
    with b.localForm() as b_local:
        _assemble._assemble_vector_array(b_local.array_w, L, constants, coeffs)
    return b


@assemble_vector.register(PETSc.Vec)
def _assemble_vector_vec(b: PETSc.Vec, L: Form, constants=None, coeffs=None) -> PETSc.Vec:
    """Assemble linear form into an existing PETSc vector.

    Note:
        The vector is not zeroed before assembly and it is not
        finalised, i.e. ghost values are not accumulated on the owning
        processes.

    Args:
        b: Vector to assemble the contribution of the linear form into.
        L: A linear form to assemble into ``b``.

    Returns:
        An assembled vector.
    """
    with b.localForm() as b_local:
        _assemble._assemble_vector_array(b_local.array_w, L, constants, coeffs)
    return b


@functools.singledispatch
def assemble_vector_nest(L: typing.Any, constants=None, coeffs=None) -> PETSc.Vec:
    """Assemble linear forms into a new nested PETSc (``VecNest``) vector.

    The returned vector is not finalised, i.e. ghost values are not
    accumulated on the owning processes.

    Note:
        Due to subtle issues in the interaction between petsc4py memory
        management and the Python garbage collector, it is recommended
        that the method ``PETSc.Vec.destroy()`` is called on the
        returned object once the object is no longer required. Note that
        ``PETSc.Vec.destroy()`` is collective over the object's MPI
        communicator.
    """
    maps = [
        (form.function_spaces[0].dofmap.index_map, form.function_spaces[0].dofmap.index_map_bs)
        for form in L
    ]
    b = _cpp.fem.petsc.create_vector_nest(maps)
    for b_sub in b.getNestSubVecs():
        with b_sub.localForm() as b_local:
            b_local.set(0.0)
    return _assemble_vector_nest_vec(b, L, constants, coeffs)


@assemble_vector_nest.register
def _assemble_vector_nest_vec(
    b: PETSc.Vec, L: list[Form], constants=None, coeffs=None
) -> PETSc.Vec:
    """Assemble linear forms into a nested PETSc (``VecNest``) vector.

    The vector is not zeroed before assembly and it is not finalised,
    i.e. ghost values are not accumulated on the owning processes.
    """
    constants = [None] * len(L) if constants is None else constants
    coeffs = [None] * len(L) if coeffs is None else coeffs
    for b_sub, L_sub, const, coeff in zip(b.getNestSubVecs(), L, constants, coeffs):
        with b_sub.localForm() as b_local:
            _assemble._assemble_vector_array(b_local.array_w, L_sub, const, coeff)
    return b


# FIXME: Revise this interface
@functools.singledispatch
def assemble_vector_block(
    L: list[Form],
    a: list[list[Form]],
    bcs: list[DirichletBC] = [],
    x0: typing.Optional[PETSc.Vec] = None,
    alpha: float = 1,
    constants_L=None,
    coeffs_L=None,
    constants_a=None,
    coeffs_a=None,
) -> PETSc.Vec:
    """Assemble linear forms into a monolithic vector.

    The vector is not finalised, i.e. ghost values are not accumulated.

    Note:
        Due to subtle issues in the interaction between petsc4py memory
        management and the Python garbage collector, it is recommended
        that the method ``PETSc.Vec.destroy()`` is called on the
        returned object once the object is no longer required. Note that
        ``PETSc.Vec.destroy()`` is collective over the object's MPI
        communicator.
    """
    maps = [
        (form.function_spaces[0].dofmap.index_map, form.function_spaces[0].dofmap.index_map_bs)
        for form in L
    ]
    b = _cpp.fem.petsc.create_vector_block(maps)
    with b.localForm() as b_local:
        b_local.set(0.0)
    return _assemble_vector_block_vec(
        b, L, a, bcs, x0, alpha, constants_L, coeffs_L, constants_a, coeffs_a
    )


@assemble_vector_block.register
def _assemble_vector_block_vec(
    b: PETSc.Vec,
    L: list[Form],
    a: list[list[Form]],
    bcs: list[DirichletBC] = [],
    x0: typing.Optional[PETSc.Vec] = None,
    alpha: float = 1,
    constants_L=None,
    coeffs_L=None,
    constants_a=None,
    coeffs_a=None,
) -> PETSc.Vec:
    """Assemble linear forms into a monolithic vector.

    The vector is not zeroed and it is not finalised, i.e. ghost values
    are not accumulated.
    """
    maps = [
        (form.function_spaces[0].dofmap.index_map, form.function_spaces[0].dofmap.index_map_bs)
        for form in L
    ]
    if x0 is not None:
        x0_local = _cpp.la.petsc.get_local_vectors(x0, maps)
        x0_sub = x0_local
    else:
        x0_local = []
        x0_sub = [None] * len(maps)

    constants_L = (
        [form and _pack_constants(form._cpp_object) for form in L]
        if constants_L is None
        else constants_L
    )
    coeffs_L = (
        [{} if form is None else _pack_coefficients(form._cpp_object) for form in L]
        if coeffs_L is None
        else coeffs_L
    )

    constants_a = (
        [
            [
                _pack_constants(form._cpp_object)
                if form is not None
                else np.array([], dtype=PETSc.ScalarType)
                for form in forms
            ]
            for forms in a
        ]
        if constants_a is None
        else constants_a
    )

    coeffs_a = (
        [
            [{} if form is None else _pack_coefficients(form._cpp_object) for form in forms]
            for forms in a
        ]
        if coeffs_a is None
        else coeffs_a
    )

    _bcs = [bc._cpp_object for bc in bcs]
    bcs1 = _bcs_by_block(_extract_spaces(a, 1), _bcs)
    b_local = _cpp.la.petsc.get_local_vectors(b, maps)
    for b_sub, L_sub, a_sub, const_L, coeff_L, const_a, coeff_a in zip(
        b_local, L, a, constants_L, coeffs_L, constants_a, coeffs_a
    ):
        _cpp.fem.assemble_vector(b_sub, L_sub._cpp_object, const_L, coeff_L)
        _a_sub = [None if form is None else form._cpp_object for form in a_sub]
        _cpp.fem.apply_lifting(b_sub, _a_sub, const_a, coeff_a, bcs1, x0_local, alpha)

    _cpp.la.petsc.scatter_local_vectors(b, b_local, maps)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    bcs0 = _bcs_by_block(_extract_spaces(L), _bcs)
    offset = 0
    b_array = b.getArray(readonly=False)
    for submap, bcs, _x0 in zip(maps, bcs0, x0_sub):
        size = submap[0].size_local * submap[1]
        for bc in bcs:
            bc.set(b_array[offset : offset + size], _x0, alpha)
        offset += size

    return b


# -- Matrix assembly ---------------------------------------------------------


class _MatSubMatrixWrapper(object):
    def __init__(self,
                 A: PETSc.Mat, unrestricted_index_sets: tuple[PETSc.IS],
                 restricted_index_sets: typing.Optional[tuple[PETSc.IS]] = None,
                 unrestricted_to_restricted: typing.Optional[tuple[dict[int, int]]] = None,
                 unrestricted_to_restricted_bs: typing.Optional[tuple[int]] = None):
        if restricted_index_sets is None:
            assert unrestricted_to_restricted is None
            assert unrestricted_to_restricted_bs is None
            self._cpp_object = _cpp.la.petsc.MatSubMatrixWrapper(A, unrestricted_index_sets)
        else:
            self._cpp_object = _cpp.la.petsc.MatSubMatrixWrapper(
                A, unrestricted_index_sets,
                restricted_index_sets,
                unrestricted_to_restricted,
                unrestricted_to_restricted_bs)

    def __enter__(self):
        return self._cpp_object.mat()

    def __exit__(self, exception_type, exception_value, traceback):
        self._cpp_object.restore()


class MatSubMatrixWrapper(object):
    def __init__(self,
                 A: PETSc.Mat, dofmaps: tuple[_cpp.fem.DofMap],
                 restriction: typing.Optional[tuple[_cpp.fem.DofMapRestriction]] = None):
        assert len(dofmaps) == 2
        if restriction is None:
            index_maps = ((dofmaps[0].index_map, dofmaps[0].index_map_bs),
                          (dofmaps[1].index_map, dofmaps[1].index_map_bs))
            index_sets = (_cpp.la.petsc.create_index_sets([index_maps[0]], [dofmaps[0].index_map_bs])[0],
                          _cpp.la.petsc.create_index_sets([index_maps[1]], [dofmaps[1].index_map_bs])[0])
            self._wrapper = _MatSubMatrixWrapper(A, index_sets)
            self._unrestricted_index_sets = index_sets
            self._restricted_index_sets = None
            self._unrestricted_to_restricted = None
            self._unrestricted_to_restricted_bs = None
        else:
            assert len(restriction) == 2
            assert all([_same_dofmap(dofmaps[i], restriction[i].dofmap) for i in range(2)])
            unrestricted_index_maps = ((dofmaps[0].index_map, dofmaps[0].index_map_bs),
                                       (dofmaps[1].index_map, dofmaps[1].index_map_bs))
            unrestricted_index_sets = (_cpp.la.petsc.create_index_sets([unrestricted_index_maps[0]],
                                                                       [dofmaps[0].index_map_bs])[0],
                                       _cpp.la.petsc.create_index_sets([unrestricted_index_maps[1]],
                                                                       [dofmaps[1].index_map_bs])[0])
            restricted_index_maps = ((restriction[0].index_map, restriction[0].index_map_bs),
                                     (restriction[1].index_map, restriction[1].index_map_bs))
            restricted_index_sets = (_cpp.la.petsc.create_index_sets([restricted_index_maps[0]],
                                                                     [restriction[0].index_map_bs])[0],
                                     _cpp.la.petsc.create_index_sets([restricted_index_maps[1]],
                                                                     [restriction[1].index_map_bs])[0])
            unrestricted_to_restricted = (restriction[0].unrestricted_to_restricted,
                                          restriction[1].unrestricted_to_restricted)
            unrestricted_to_restricted_bs = (restriction[0].index_map_bs,
                                             restriction[1].index_map_bs)
            self._wrapper = _MatSubMatrixWrapper(A, unrestricted_index_sets,
                                                 restricted_index_sets, unrestricted_to_restricted,
                                                 unrestricted_to_restricted_bs)
            self._unrestricted_index_sets = unrestricted_index_sets
            self._restricted_index_sets = restricted_index_sets
            self._unrestricted_to_restricted = unrestricted_to_restricted
            self._unrestricted_to_restricted_bs = unrestricted_to_restricted_bs

    def __enter__(self):
        return self._wrapper.__enter__()

    def __exit__(self, exception_type, exception_value, traceback):
        self._wrapper.__exit__(exception_type, exception_value, traceback)
        self._unrestricted_index_sets[0].destroy()
        self._unrestricted_index_sets[1].destroy()
        if self._restricted_index_sets is not None:
            self._restricted_index_sets[0].destroy()
            self._restricted_index_sets[1].destroy()


class BlockMatSubMatrixWrapper(object):
    def __init__(self,
                 A: PETSc.Mat, dofmaps: tuple[list[_cpp.fem.DofMap]],
                 restriction: typing.Optional[tuple[list[_cpp.fem.DofMapRestriction]]] = None):
        self._A = A
        assert len(dofmaps) == 2
        if restriction is None:
            index_maps = ([(dofmap.index_map, dofmap.index_map_bs) for dofmap in dofmaps[0]],
                          [(dofmap.index_map, dofmap.index_map_bs) for dofmap in dofmaps[1]])
            index_sets = (_cpp.la.petsc.create_index_sets(index_maps[0], [1] * len(index_maps[0])),
                          _cpp.la.petsc.create_index_sets(index_maps[1], [1] * len(index_maps[1])))
            self._unrestricted_index_sets = index_sets
            self._restricted_index_sets = None
            self._unrestricted_to_restricted = None
            self._unrestricted_to_restricted_bs = None
        else:
            assert len(restriction) == 2
            for i in range(2):
                assert len(dofmaps[i]) == len(restriction[i])
                assert all([_same_dofmap(dofmap, restriction_.dofmap)
                            for (dofmap, restriction_) in zip(dofmaps[i], restriction[i])])
            unrestricted_index_maps = ([(dofmap.index_map, dofmap.index_map_bs) for dofmap in dofmaps[0]],
                                       [(dofmap.index_map, dofmap.index_map_bs) for dofmap in dofmaps[1]])
            unrestricted_index_sets = (_cpp.la.petsc.create_index_sets(unrestricted_index_maps[0],
                                                                       [1] * len(unrestricted_index_maps[0])),
                                       _cpp.la.petsc.create_index_sets(unrestricted_index_maps[1],
                                                                       [1] * len(unrestricted_index_maps[1])))
            restricted_index_maps = ([(restriction_.index_map, restriction_.index_map_bs)
                                      for restriction_ in restriction[0]],
                                     [(restriction_.index_map, restriction_.index_map_bs)
                                      for restriction_ in restriction[1]])
            restricted_index_sets = (_cpp.la.petsc.create_index_sets(restricted_index_maps[0],
                                                                     [1] * len(restricted_index_maps[0])),
                                     _cpp.la.petsc.create_index_sets(restricted_index_maps[1],
                                                                     [1] * len(restricted_index_maps[1])))
            unrestricted_to_restricted = ([restriction_.unrestricted_to_restricted for restriction_ in restriction[0]],
                                          [restriction_.unrestricted_to_restricted for restriction_ in restriction[1]])
            unrestricted_to_restricted_bs = ([restriction_.index_map_bs for restriction_ in restriction[0]],
                                             [restriction_.index_map_bs for restriction_ in restriction[1]])
            self._unrestricted_index_sets = unrestricted_index_sets
            self._restricted_index_sets = restricted_index_sets
            self._unrestricted_to_restricted = unrestricted_to_restricted
            self._unrestricted_to_restricted_bs = unrestricted_to_restricted_bs

    def __iter__(self):
        with contextlib.ExitStack() as wrapper_stack:
            for index0, _ in enumerate(self._unrestricted_index_sets[0]):
                for index1, _ in enumerate(self._unrestricted_index_sets[1]):
                    if self._restricted_index_sets is None:
                        wrapper = _MatSubMatrixWrapper(self._A,
                                                       (self._unrestricted_index_sets[0][index0],
                                                        self._unrestricted_index_sets[1][index1]))
                    else:
                        wrapper = _MatSubMatrixWrapper(self._A,
                                                       (self._unrestricted_index_sets[0][index0],
                                                        self._unrestricted_index_sets[1][index1]),
                                                       (self._restricted_index_sets[0][index0],
                                                        self._restricted_index_sets[1][index1]),
                                                       (self._unrestricted_to_restricted[0][index0],
                                                        self._unrestricted_to_restricted[1][index1]),
                                                       (self._unrestricted_to_restricted_bs[0][index0],
                                                        self._unrestricted_to_restricted_bs[1][index1]))
                    yield (index0, index1, wrapper_stack.enter_context(wrapper))

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        for i in range(2):
            for index_set in self._unrestricted_index_sets[i]:
                index_set.destroy()
        if self._restricted_index_sets is not None:
            for i in range(2):
                for index_set in self._restricted_index_sets[i]:
                    index_set.destroy()


class NestMatSubMatrixWrapper(object):
    def __init__(self,
                 A: PETSc.Mat, dofmaps: tuple[list[_cpp.fem.DofMap]],
                 restriction: typing.Optional[tuple[list[_cpp.fem.DofMapRestriction]]] = None):
        self._A = A
        self._dofmaps = dofmaps
        self._restriction = restriction

    def __iter__(self):
        with contextlib.ExitStack() as wrapper_stack:
            for index0, _ in enumerate(self._dofmaps[0]):
                for index1, _ in enumerate(self._dofmaps[1]):
                    A_sub = self._A.getNestSubMatrix(index0, index1)
                    if self._restriction is None:
                        wrapper_content = A_sub
                    else:
                        wrapper = MatSubMatrixWrapper(A_sub,
                                                      (self._dofmaps[0][index0], self._dofmaps[1][index1]),
                                                      (self._restriction[0][index0], self._restriction[1][index1]))
                        wrapper_content = wrapper_stack.enter_context(wrapper)
                    yield (index0, index1, wrapper_content)
                    A_sub.destroy()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass


@functools.singledispatch
def assemble_matrix(a: typing.Any, bcs: list[DirichletBC] = [],
                    diagonal: float = 1.0,
                    constants=None, coeffs=None,
                    restriction: typing.Optional[tuple[_cpp.fem.DofMapRestriction]] = None) -> PETSc.Mat:
    return _assemble_matrix_form(a, bcs, diagonal, constants, coeffs, restriction)


@assemble_matrix.register(Form)
def _assemble_matrix_form(a: Form, bcs: list[DirichletBC] = [],
                          diagonal: float = 1.0,
                          constants=None, coeffs=None,
                          restriction: typing.Optional[tuple[_cpp.fem.DofMapRestriction]] = None) -> PETSc.Mat:
    """Assemble bilinear form into a matrix.

    The returned matrix is not finalised, i.e. ghost values are not
    accumulated.

    Note:
        The returned matrix is not 'assembled', i.e. ghost contributions
        have not been communicated.

    Note:
        Due to subtle issues in the interaction between petsc4py memory
        management and the Python garbage collector, it is recommended
        that the method ``PETSc.Mat.destroy()`` is called on the
        returned object once the object is no longer required. Note that
        ``PETSc.Mat.destroy()`` is collective over the object's MPI
        communicator.

    Args:
        a: Bilinear form to assembled into a matrix.
        bc: Dirichlet boundary conditions applied to the system.
        diagonal: Value to set on the matrix diagonal for Dirichlet
            boundary condition constrained degrees-of-freedom belonging
            to the same trial and test space.
        constants: Constants appearing the in the form.
        coeffs: Coefficients appearing the in the form.

    Returns:
        Matrix representing the bilinear form.
    """
    A = create_matrix(a, restriction)
    assemble_matrix_mat(A, a, bcs, diagonal, constants, coeffs, restriction)
    return A


@assemble_matrix.register(PETSc.Mat)
def assemble_matrix_mat(A: PETSc.Mat, a: Form, bcs: list[DirichletBC] = [],
                        diagonal: float = 1.0, constants=None, coeffs=None,
                        restriction: typing.Optional[tuple[_cpp.fem.DofMapRestriction]] = None) -> PETSc.Mat:
    """Assemble bilinear form into a matrix.

    The returned matrix is not finalised, i.e. ghost values are not
    accumulated.
    """
    constants = _pack_constants(a._cpp_object) if constants is None else constants
    coeffs = _pack_coefficients(a._cpp_object) if coeffs is None else coeffs
    bcs_cpp = [bc._cpp_object for bc in bcs]
    function_spaces = a.function_spaces
    if restriction is None:
        # Assemble form
        _cpp.fem.petsc.assemble_matrix(A, a._cpp_object, constants, coeffs, bcs_cpp)

        if a.function_spaces[0] is a.function_spaces[1]:
            # Flush to enable switch from add to set in the matrix
            A.assemble(PETSc.Mat.AssemblyType.FLUSH)

            # Set diagonal
            _cpp.fem.petsc.insert_diagonal(A, function_spaces[0], bcs_cpp, diagonal)
    else:
        dofmaps = (function_spaces[0].dofmap, function_spaces[1].dofmap)

        # Assemble form
        with MatSubMatrixWrapper(A, dofmaps, restriction) as A_sub:
            _cpp.fem.petsc.assemble_matrix(A_sub, a._cpp_object, constants, coeffs, bcs_cpp)

        if a.function_spaces[0] is a.function_spaces[1]:
            # Flush to enable switch from add to set in the matrix
            A.assemble(PETSc.Mat.AssemblyType.FLUSH)

            # Set diagonal
            with MatSubMatrixWrapper(A, dofmaps, restriction) as A_sub:
                _cpp.fem.petsc.insert_diagonal(A_sub, function_spaces[0], bcs_cpp, diagonal)
    return A


# FIXME: Revise this interface
@functools.singledispatch
def assemble_matrix_nest(a: list[list[Form]],
                         bcs: list[DirichletBC] = [], mat_types=[],
                         diagonal: float = 1.0,
                         constants=None, coeffs=None,
                         restriction: typing.Optional[tuple[
                                                      list[_cpp.fem.DofMapRestriction]]] = None) -> PETSc.Mat:
    """Create a nested matrix and assembled bilinear forms into the matrix.

    Note:
        Due to subtle issues in the interaction between petsc4py memory
        management and the Python garbage collector, it is recommended
        that the method ``PETSc.Mat.destroy()`` is called on the
        returned object once the object is no longer required. Note that
        ``PETSc.Mat.destroy()`` is collective over the object's MPI
        communicator.

    Args:
        a: Rectangular (list-of-lists) array for bilinear forms.
        bcs: Dirichlet boundary conditions.
        mat_types: PETSc matrix type for each matrix block.
        diagonal: Value to set on the matrix diagonal for Dirichlet
            boundary condition constrained degrees-of-freedom belonging
            to the same trial and test space.
        constants: Constants appearing the in the form.
        coeffs: Coefficients appearing the in the form.

    Returns:
        PETSc matrix (``MatNest``) representing the block of bilinear
        forms.
    """
    A = create_matrix_nest(a, restriction, mat_types)
    _assemble_matrix_nest_mat(A, a, bcs, diagonal, constants, coeffs, restriction)
    return A


@assemble_matrix_nest.register
def _assemble_matrix_nest_mat(A: PETSc.Mat, a: list[list[Form]],
                              bcs: list[DirichletBC] = [], diagonal: float = 1.0,
                              constants=None, coeffs=None,
                              restriction: typing.Optional[tuple[
                                                           list[_cpp.fem.DofMapRestriction]]
                                                           ] = None) -> PETSc.Mat:
    """Assemble bilinear forms into a nested matrix

    Args:
        A: PETSc ``MatNest`` matrix. Matrix must have been correctly
            initialized for the bilinear forms.
        a: Rectangular (list-of-lists) array for bilinear forms.
        bcs: Dirichlet boundary conditions.
        mat_types: PETSc matrix type for each matrix block.
        diagonal: Value to set on the matrix diagonal for Dirichlet
            boundary condition constrained degrees-of-freedom belonging
            to the same trial and test space.
        constants: Constants appearing the in the form.
        coeffs: Coefficients appearing the in the form.

    Returns:
        PETSc matrix (``MatNest``) representing the block of bilinear
        forms.
    """
    function_spaces = _get_block_function_spaces(a)
    dofmaps = ([function_space.dofmap for function_space in function_spaces[0]],
               [function_space.dofmap for function_space in function_spaces[1]])

    # Assemble form
    constants = [[form and _pack_constants(form._cpp_object) for form in forms]
                 for forms in a] if constants is None else constants
    coeffs = [[{} if form is None else _pack_coefficients(
        form._cpp_object) for form in forms] for forms in a] if coeffs is None else coeffs
    bcs_cpp = [bc._cpp_object for bc in bcs]
    with NestMatSubMatrixWrapper(A, dofmaps, restriction) as nest_A:
        for i, j, A_sub in nest_A:
            a_sub = a[i][j]
            if a_sub is not None:
                const_sub = constants[i][j]
                coeff_sub = coeffs[i][j]
                _cpp.fem.petsc.assemble_matrix(A_sub, a_sub._cpp_object, const_sub, coeff_sub, bcs_cpp)
            elif i == j:
                for bc in bcs:
                    if function_spaces[0][i].contains(bc.function_space):
                        raise RuntimeError(
                            f"Diagonal sub-block ({i}, {j}) cannot be 'None' "
                            " and have DirichletBC applied."
                            " Consider assembling a zero block."
                        )

    # Flush to enable switch from add to set in the matrix
    A.assemble(PETSc.Mat.AssemblyType.FLUSH)

    # Set diagonal
    with NestMatSubMatrixWrapper(A, dofmaps, restriction) as nest_A:
        for i, j, A_sub in nest_A:
            if function_spaces[0][i] is function_spaces[1][j]:
                a_sub = a[i][j]
                if a_sub is not None:
                    _cpp.fem.petsc.insert_diagonal(A_sub, function_spaces[0][i], bcs_cpp, diagonal)

    return A


# FIXME: Revise this interface
@functools.singledispatch
def assemble_matrix_block(a: list[list[Form]],
                          bcs: list[DirichletBC] = [],
                          diagonal: float = 1.0,
                          constants=None, coeffs=None,
                          restriction: typing.Optional[tuple[
                                                       list[_cpp.fem.DofMapRestriction]]] = None) -> PETSc.Mat:
    """Assemble bilinear forms into a blocked matrix."""
    A = create_matrix_block(a, restriction)
    return _assemble_matrix_block_mat(A, a, bcs, diagonal, constants, coeffs, restriction)


@assemble_matrix_block.register
def _assemble_matrix_block_mat(A: PETSc.Mat, a: list[list[Form]],
                               bcs: list[DirichletBC] = [], diagonal: float = 1.0,
                               constants=None, coeffs=None,
                               restriction: typing.Optional[tuple[
                                                            list[_cpp.fem.DofMapRestriction]]
                                                            ] = None) -> PETSc.Mat:
    """Assemble bilinear forms into a blocked matrix."""
    constants = [[_pack_constants(form._cpp_object) if form is not None else np.array(
        [], dtype=PETSc.ScalarType) for form in forms] for forms in a] if constants is None else constants
    coeffs = [[{} if form is None else _pack_coefficients(
        form._cpp_object) for form in forms] for forms in a] if coeffs is None else coeffs
    function_spaces = _get_block_function_spaces(a)
    dofmaps = ([function_space.dofmap for function_space in function_spaces[0]],
               [function_space.dofmap for function_space in function_spaces[1]])

    # Assemble form
    bcs_cpp = [bc._cpp_object for bc in bcs]
    with BlockMatSubMatrixWrapper(A, dofmaps, restriction) as block_A:
        for i, j, A_sub in block_A:
            a_sub = a[i][j]
            if a_sub is not None:
                const_sub = constants[i][j]
                coeff_sub = coeffs[i][j]
                _cpp.fem.petsc.assemble_matrix(A_sub, a_sub._cpp_object, const_sub, coeff_sub, bcs_cpp, True)
            elif i == j:
                for bc in bcs:
                    if function_spaces[0][i].contains(bc.function_space):
                        raise RuntimeError(
                            f"Diagonal sub-block ({i}, {j}) cannot be 'None' and have DirichletBC applied."
                            " Consider assembling a zero block.")

    # Flush to enable switch from add to set in the matrix
    A.assemble(PETSc.Mat.AssemblyType.FLUSH)

    # Set diagonal
    with BlockMatSubMatrixWrapper(A, dofmaps, restriction) as block_A:
        for i, j, A_sub in block_A:
            if function_spaces[0][i] is function_spaces[1][j]:
                a_sub = a[i][j]
                if a_sub is not None:
                    _cpp.fem.petsc.insert_diagonal(A_sub, function_spaces[0][i], bcs_cpp, diagonal)

    return A


# -- Modifiers for Dirichlet conditions ---------------------------------------


def apply_lifting(
    b: PETSc.Vec,
    a: list[Form],
    bcs: list[list[DirichletBC]],
    x0: list[PETSc.Vec] = [],
    alpha: float = 1,
    constants=None,
    coeffs=None,
) -> None:
    """Apply the function :func:`dolfinx.fem.apply_lifting` to a PETSc Vector."""
    with contextlib.ExitStack() as stack:
        x0 = [stack.enter_context(x.localForm()) for x in x0]
        x0_r = [x.array_r for x in x0]
        b_local = stack.enter_context(b.localForm())
        _assemble.apply_lifting(b_local.array_w, a, bcs, x0_r, alpha, constants, coeffs)


def apply_lifting_nest(
    b: PETSc.Vec,
    a: list[list[Form]],
    bcs: list[DirichletBC],
    x0: typing.Optional[PETSc.Vec] = None,
    alpha: float = 1,
    constants=None,
    coeffs=None,
) -> PETSc.Vec:
    """Apply the function :func:`dolfinx.fem.apply_lifting` to each sub-vector
    in a nested PETSc Vector."""
    x0 = [] if x0 is None else x0.getNestSubVecs()
    bcs1 = _bcs_by_block(_extract_spaces(a, 1), bcs)
    constants = (
        [
            [
                _pack_constants(form._cpp_object)
                if form is not None
                else np.array([], dtype=PETSc.ScalarType)
                for form in forms
            ]
            for forms in a
        ]
        if constants is None
        else constants
    )
    coeffs = (
        [
            [{} if form is None else _pack_coefficients(form._cpp_object) for form in forms]
            for forms in a
        ]
        if coeffs is None
        else coeffs
    )
    for b_sub, a_sub, const, coeff in zip(b.getNestSubVecs(), a, constants, coeffs):
        apply_lifting(b_sub, a_sub, bcs1, x0, alpha, const, coeff)
    return b


def set_bc(
    b: PETSc.Vec, bcs: list[DirichletBC], x0: typing.Optional[PETSc.Vec] = None, alpha: float = 1
) -> None:
    """Apply the function :func:`dolfinx.fem.set_bc` to a PETSc Vector."""
    if x0 is not None:
        x0 = x0.array_r
    for bc in bcs:
        bc.set(b.array_w, x0, alpha)


def set_bc_nest(
    b: PETSc.Vec,
    bcs: list[list[DirichletBC]],
    x0: typing.Optional[PETSc.Vec] = None,
    alpha: float = 1,
) -> None:
    """Apply the function :func:`dolfinx.fem.set_bc` to each sub-vector
    of a nested PETSc Vector.
    """
    _b = b.getNestSubVecs()
    x0 = len(_b) * [None] if x0 is None else x0.getNestSubVecs()
    for b_sub, bc, x_sub in zip(_b, bcs, x0):
        set_bc(b_sub, bc, x_sub, alpha)


class LinearProblem:
    """Class for solving a linear variational problem.

    Solves of the form :math:`a(u, v) = L(v) \\,  \\forall v \\in V`
    using PETSc as a linear algebra backend.
    """

    def __init__(
        self,
        a: ufl.Form,
        L: ufl.Form,
        bcs: list[DirichletBC] = [],
        u: typing.Optional[_Function] = None,
        petsc_options: typing.Optional[dict] = None,
        form_compiler_options: typing.Optional[dict] = None,
        jit_options: typing.Optional[dict] = None,
    ):
        """Initialize solver for a linear variational problem.

        Args:
            a: A bilinear UFL form, the left hand side of the
                variational problem.
            L: A linear UFL form, the right hand side of the variational
                problem.
            bcs: A list of Dirichlet boundary conditions.
            u: The solution function. It will be created if not provided.
            petsc_options: Options that are passed to the linear
                algebra backend PETSc. For available choices for the
                'petsc_options' kwarg, see the `PETSc documentation
                <https://petsc4py.readthedocs.io/en/stable/manual/ksp/>`_.
            form_compiler_options: Options used in FFCx compilation of
                this form. Run ``ffcx --help`` at the commandline to see
                all available options.
            jit_options: Options used in CFFI JIT compilation of C
                code generated by FFCx. See `python/dolfinx/jit.py` for
                all available options. Takes priority over all other
                option values.

        Example::

            problem = LinearProblem(a, L, [bc0, bc1], petsc_options={"ksp_type": "preonly",
                                                                     "pc_type": "lu",
                                                                     "pc_factor_mat_solver_type":
                                                                       "mumps"})
        """
        self._a = _create_form(
            a,
            dtype=PETSc.ScalarType,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
        )
        self._A = create_matrix(self._a)
        self._L = _create_form(
            L,
            dtype=PETSc.ScalarType,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
        )
        self._b = create_vector(self._L)

        if u is None:
            # Extract function space from TrialFunction (which is at the
            # end of the argument list as it is numbered as 1, while the
            # Test function is numbered as 0)
            self.u = _Function(a.arguments()[-1].ufl_function_space())
        else:
            self.u = u

        self._x = la.create_petsc_vector_wrap(self.u.x)
        self.bcs = bcs

        self._solver = PETSc.KSP().create(self.u.function_space.mesh.comm)
        self._solver.setOperators(self._A)

        # Give PETSc solver options a unique prefix
        problem_prefix = f"dolfinx_solve_{id(self)}"
        self._solver.setOptionsPrefix(problem_prefix)

        # Set PETSc options
        opts = PETSc.Options()
        opts.prefixPush(problem_prefix)
        if petsc_options is not None:
            for k, v in petsc_options.items():
                opts[k] = v
        opts.prefixPop()
        self._solver.setFromOptions()

        # Set matrix and vector PETSc options
        self._A.setOptionsPrefix(problem_prefix)
        self._A.setFromOptions()
        self._b.setOptionsPrefix(problem_prefix)
        self._b.setFromOptions()

    def __del__(self):
        self._solver.destroy()
        self._A.destroy()
        self._b.destroy()
        self._x.destroy()

    def solve(self) -> _Function:
        """Solve the problem."""

        # Assemble lhs
        self._A.zeroEntries()
        assemble_matrix_mat(self._A, self._a, bcs=self.bcs)
        self._A.assemble()

        # Assemble rhs
        with self._b.localForm() as b_loc:
            b_loc.set(0)
        assemble_vector(self._b, self._L)

        # Apply boundary conditions to the rhs
        apply_lifting(self._b, [self._a], bcs=[self.bcs])
        self._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        for bc in self.bcs:
            bc.set(self._b.array_w)

        # Solve linear system and update ghost values in the solution
        self._solver.solve(self._b, self._x)
        self.u.x.scatter_forward()

        return self.u

    @property
    def L(self) -> Form:
        """The compiled linear form"""
        return self._L

    @property
    def a(self) -> Form:
        """The compiled bilinear form"""
        return self._a

    @property
    def A(self) -> PETSc.Mat:
        """Matrix operator"""
        return self._A

    @property
    def b(self) -> PETSc.Vec:
        """Right-hand side vector"""
        return self._b

    @property
    def solver(self) -> PETSc.KSP:
        """Linear solver object"""
        return self._solver


class NonlinearProblem:
    """Nonlinear problem class for solving the non-linear problems.

    Solves problems of the form :math:`F(u, v) = 0 \\ \\forall v \\in V` using
    PETSc as the linear algebra backend.
    """

    def __init__(
        self,
        F: ufl.form.Form,
        u: _Function,
        bcs: list[DirichletBC] = [],
        J: ufl.form.Form = None,
        form_compiler_options: typing.Optional[dict] = None,
        jit_options: typing.Optional[dict] = None,
    ):
        """Initialize solver for solving a non-linear problem using Newton's method`.

        Args:
            F: The PDE residual F(u, v)
            u: The unknown
            bcs: List of Dirichlet boundary conditions
            J: UFL representation of the Jacobian (Optional)
            form_compiler_options: Options used in FFCx
                compilation of this form. Run ``ffcx --help`` at the
                command line to see all available options.
            jit_options: Options used in CFFI JIT compilation of C
                code generated by FFCx. See ``python/dolfinx/jit.py``
                for all available options. Takes priority over all other
                option values.

        Example::

            problem = LinearProblem(F, u, [bc0, bc1])
        """
        self._L = _create_form(
            F, form_compiler_options=form_compiler_options, jit_options=jit_options
        )

        # Create the Jacobian matrix, dF/du
        if J is None:
            V = u.function_space
            du = ufl.TrialFunction(V)
            J = ufl.derivative(F, u, du)

        self._a = _create_form(
            J, form_compiler_options=form_compiler_options, jit_options=jit_options
        )
        self.bcs = bcs

    @property
    def L(self) -> Form:
        """Compiled linear form (the residual form)"""
        return self._L

    @property
    def a(self) -> Form:
        """Compiled bilinear form (the Jacobian form)"""
        return self._a

    def form(self, x: PETSc.Vec) -> None:
        """This function is called before the residual or Jacobian is
        computed. This is usually used to update ghost values.

        Args:
           x: The vector containing the latest solution
        """
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def F(self, x: PETSc.Vec, b: PETSc.Vec) -> None:
        """Assemble the residual F into the vector b.

        Args:
            x: The vector containing the latest solution
            b: Vector to assemble the residual into
        """
        # Reset the residual vector
        with b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(b, self._L)

        # Apply boundary condition
        apply_lifting(b, [self._a], bcs=[self.bcs], x0=[x], alpha=-1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, self.bcs, x, -1.0)

    def J(self, x: PETSc.Vec, A: PETSc.Mat) -> None:
        """Assemble the Jacobian matrix.

        Args:
            x: The vector containing the latest solution
        """
        A.zeroEntries()
        assemble_matrix_mat(A, self._a, self.bcs)
        A.assemble()


def discrete_curl(space0: _FunctionSpace, space1: _FunctionSpace) -> PETSc.Mat:
    """Assemble a discrete curl operator.

    Args:
        space0: H1 space to interpolate the gradient from.
        space1: H(curl) space to interpolate into.

    Returns:
        Discrete curl operator.
    """
    return _discrete_curl(space0._cpp_object, space1._cpp_object)


def discrete_gradient(space0: _FunctionSpace, space1: _FunctionSpace) -> PETSc.Mat:
    """Assemble a discrete gradient operator.

    The discrete gradient operator interpolates the gradient of a H1
    finite element function into a H(curl) space. It is assumed that the
    H1 space uses an identity map and the H(curl) space uses a covariant
    Piola map.

    Note:
        Due to subtle issues in the interaction between petsc4py memory
        management and the Python garbage collector, it is recommended
        that the method ``PETSc.Mat.destroy()`` is called on the
        returned object once the object is no longer required. Note that
        ``PETSc.Mat.destroy()`` is collective over the object's MPI
        communicator.

    Args:
        space0: H1 space to interpolate the gradient from.
        space1: H(curl) space to interpolate into.

    Returns:
        Discrete gradient operator.
    """
    return _discrete_gradient(space0._cpp_object, space1._cpp_object)


def interpolation_matrix(space0: _FunctionSpace, space1: _FunctionSpace) -> PETSc.Mat:
    """Assemble an interpolation operator matrix.

    Note:
        Due to subtle issues in the interaction between petsc4py memory
        management and the Python garbage collector, it is recommended
        that the method ``PETSc.Mat.destroy()`` is called on the
        returned object once the object is no longer required. Note that
        ``PETSc.Mat.destroy()`` is collective over the object's MPI
        communicator.

    Args:
        space0: Space to interpolate from.
        space1: Space to interpolate into.

    Returns:
        Interpolation matrix.
    """
    return _interpolation_matrix(space0._cpp_object, space1._cpp_object)
