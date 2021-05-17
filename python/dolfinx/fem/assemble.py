# Copyright (C) 2018-2019 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Assembly functions for variational forms."""

import contextlib
import functools
import typing

from petsc4py import PETSc

import ufl
from dolfinx import cpp
from dolfinx.fem.dirichletbc import DirichletBC
from dolfinx.fem.form import Form


def _create_cpp_form(form):
    """Recursively look for ufl.Forms and convert to dolfinx.fem.Form, otherwise
    return form argument
    """
    if isinstance(form, Form):
        return form._cpp_object
    elif isinstance(form, ufl.Form):
        return Form(form)._cpp_object
    elif isinstance(form, (tuple, list)):
        return list(map(lambda sub_form: _create_cpp_form(sub_form), form))
    return form


def _get_block_function_spaces(_a):
    rows = len(_a)
    cols = len(_a[0])
    assert all(len(a_i) == cols for a_i in _a)
    assert all(_a[i][j] is None or _a[i][j].rank == 2 for i in range(rows) for j in range(cols))
    function_spaces_0 = list()
    for i in range(rows):
        function_spaces_0_i = None
        for j in range(cols):
            if _a[i][j] is not None:
                function_spaces_0_i = _a[i][j].function_spaces[0]
                break
        assert function_spaces_0_i is not None
        function_spaces_0.append(function_spaces_0_i)
    function_spaces_1 = list()
    for j in range(cols):
        function_spaces_1_j = None
        for i in range(rows):
            if _a[i][j] is not None:
                function_spaces_1_j = _a[i][j].function_spaces[1]
                break
        assert function_spaces_1_j is not None
        function_spaces_1.append(function_spaces_1_j)
    function_spaces = [function_spaces_0, function_spaces_1]
    assert all(_a[i][j] is None or _a[i][j].function_spaces[0] == function_spaces[0][i]
               for i in range(rows) for j in range(cols))
    assert all(_a[i][j] is None or _a[i][j].function_spaces[1] == function_spaces[1][j]
               for i in range(rows) for j in range(cols))
    return function_spaces


# -- Vector instantiation ----------------------------------------------------


def create_vector(L: typing.Union[Form, cpp.fem.Form]) -> PETSc.Vec:
    dofmap = _create_cpp_form(L).function_spaces[0].dofmap
    return cpp.la.create_vector(dofmap.index_map, dofmap.index_map_bs)


def create_vector_block(L: typing.List[typing.Union[Form, cpp.fem.Form]]) -> PETSc.Vec:
    maps = [(form.function_spaces[0].dofmap.index_map, form.function_spaces[0].dofmap.index_map_bs)
            for form in _create_cpp_form(L)]
    return cpp.fem.create_vector_block(maps)


def create_vector_nest(L: typing.List[typing.Union[Form, cpp.fem.Form]]) -> PETSc.Vec:
    maps = [(form.function_spaces[0].dofmap.index_map, form.function_spaces[0].dofmap.index_map_bs)
            for form in _create_cpp_form(L)]
    return cpp.fem.create_vector_nest(maps)


# -- Matrix instantiation ----------------------------------------------------


def create_matrix(a: typing.Union[Form, cpp.fem.Form], mat_type=None) -> PETSc.Mat:
    _a = _create_cpp_form(a)
    assert _a.rank == 2
    function_spaces = _a.function_spaces
    dofmaps = [function_space.dofmap for function_space in function_spaces]
    index_maps = [dofmap.index_map for dofmap in dofmaps]
    index_maps_bs = [dofmap.index_map_bs for dofmap in dofmaps]
    integral_types = cpp.fem.get_integral_types_from_form(_a)
    integral_types = list(integral_types)  # TODO Remove this when pybind11#2122 is fixed.
    dofmaps_lists = [dofmap.list() for dofmap in dofmaps]
    mesh = _a.mesh
    assert all(function_space.mesh == mesh for function_space in function_spaces)
    if mat_type is not None:
        return cpp.fem.create_matrix(mesh, index_maps, index_maps_bs, integral_types, dofmaps_lists, mat_type)
    else:
        return cpp.fem.create_matrix(mesh, index_maps, index_maps_bs, integral_types, dofmaps_lists)


def _create_matrix_block_or_nest(a, mat_type, cpp_create_function):
    _a = _create_cpp_form(a)
    function_spaces = _get_block_function_spaces(_a)
    rows, cols = len(function_spaces[0]), len(function_spaces[1])
    dofmaps = [[function_spaces[0][i].dofmap for i in range(rows)],
               [function_spaces[1][j].dofmap for j in range(cols)]]
    index_maps = [[dofmaps[0][i].index_map for i in range(rows)],
                  [dofmaps[1][j].index_map for j in range(cols)]]
    index_maps_bs = [[dofmaps[0][i].index_map_bs for i in range(rows)],
                     [dofmaps[1][j].index_map_bs for j in range(cols)]]
    integral_types = [[set() for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if _a[i][j] is not None:
                integral_types[i][j].update(cpp.fem.get_integral_types_from_form(_a[i][j]))
    integral_types = [[list(integral_types[row][col]) for col in range(cols)]
                      for row in range(rows)]  # TODO Remove this when pybind11#2122 is fixed.
    dofmaps_lists = [[dofmaps[0][i].list() for i in range(rows)],
                     [dofmaps[1][j].list() for j in range(cols)]]
    mesh = None
    for j in range(cols):
        for i in range(rows):
            if _a[i][j] is not None:
                mesh = _a[i][j].mesh
    assert mesh is not None
    assert all(_a[i][j] is None or _a[i][j].mesh == mesh for i in range(rows) for j in range(cols))
    assert all(function_space.mesh == mesh for function_space in function_spaces[0])
    assert all(function_space.mesh == mesh for function_space in function_spaces[1])
    if mat_type is not None:
        return cpp_create_function(mesh, index_maps, index_maps_bs, integral_types, dofmaps_lists, mat_type)
    else:
        return cpp_create_function(mesh, index_maps, index_maps_bs, integral_types, dofmaps_lists)


def create_matrix_block(a: typing.List[typing.List[typing.Union[Form, cpp.fem.Form]]], mat_type=None) -> PETSc.Mat:
    return _create_matrix_block_or_nest(a, mat_type, cpp.fem.create_matrix_block)


def create_matrix_nest(a: typing.List[typing.List[typing.Union[Form, cpp.fem.Form]]], mat_types=None) -> PETSc.Mat:
    return _create_matrix_block_or_nest(a, mat_types, cpp.fem.create_matrix_nest)


# -- Scalar assembly ---------------------------------------------------------


def assemble_scalar(M: typing.Union[Form, cpp.fem.Form]) -> PETSc.ScalarType:
    """Assemble functional. The returned value is local and not accumulated
    across processes.

    """
    return cpp.fem.assemble_scalar(_create_cpp_form(M))

# -- Vector assembly ---------------------------------------------------------


@functools.singledispatch
def assemble_vector(L: typing.Union[Form, cpp.fem.Form]) -> PETSc.Vec:
    """Assemble linear form into a new PETSc vector. The returned vector is
    not finalised, i.e. ghost values are not accumulated on the owning
    processes.

    """
    _L = _create_cpp_form(L)
    b = cpp.la.create_vector(_L.function_spaces[0].dofmap.index_map,
                             _L.function_spaces[0].dofmap.index_map_bs)
    with b.localForm() as b_local:
        b_local.set(0.0)
        cpp.fem.assemble_vector(b_local.array_w, _L)
    return b


@assemble_vector.register(PETSc.Vec)
def _(b: PETSc.Vec, L: typing.Union[Form, cpp.fem.Form]) -> PETSc.Vec:
    """Assemble linear form into an existing PETSc vector. The vector is not
    zeroed before assembly and it is not finalised, qi.e. ghost values are
    not accumulated on the owning processes.

    """
    with b.localForm() as b_local:
        cpp.fem.assemble_vector(b_local.array_w, _create_cpp_form(L))
    return b


@functools.singledispatch
def assemble_vector_nest(L: typing.Union[Form, cpp.fem.Form]) -> PETSc.Vec:
    """Assemble linear forms into a new nested PETSc (VecNest) vector. The
    returned vector is not finalised, i.e. ghost values are not accumulated
    on the owning processes.

    """
    maps = [(form.function_spaces[0].dofmap.index_map, form.function_spaces[0].dofmap.index_map_bs)
            for form in _create_cpp_form(L)]
    b = cpp.fem.create_vector_nest(maps)
    for b_sub in b.getNestSubVecs():
        with b_sub.localForm() as b_local:
            b_local.set(0.0)
    return assemble_vector_nest(b, L)


@assemble_vector_nest.register(PETSc.Vec)
def _(b: PETSc.Vec, L: typing.List[typing.Union[Form, cpp.fem.Form]]) -> PETSc.Vec:
    """Assemble linear forms into a nested PETSc (VecNest) vector. The vector is not
    zeroed before assembly and it is not finalised, i.e. ghost values
    are not accumulated on the owning processes.

    """
    for b_sub, L_sub in zip(b.getNestSubVecs(), _create_cpp_form(L)):
        with b_sub.localForm() as b_local:
            cpp.fem.assemble_vector(b_local.array_w, L_sub)
    return b


# FIXME: Revise this interface
@functools.singledispatch
def assemble_vector_block(L: typing.List[typing.Union[Form, cpp.fem.Form]],
                          a: typing.List[typing.List[typing.Union[Form, cpp.fem.Form]]],
                          bcs: typing.List[DirichletBC] = [],
                          x0: typing.Optional[PETSc.Vec] = None,
                          scale: float = 1.0) -> PETSc.Vec:
    """Assemble linear forms into a monolithic vector. The vector is not
    finalised, i.e. ghost values are not accumulated.

    """
    maps = [(form.function_spaces[0].dofmap.index_map, form.function_spaces[0].dofmap.index_map_bs)
            for form in _create_cpp_form(L)]
    b = cpp.fem.create_vector_block(maps)
    with b.localForm() as b_local:
        b_local.set(0.0)
    return assemble_vector_block(b, L, a, bcs, x0, scale)


@assemble_vector_block.register(PETSc.Vec)
def _(b: PETSc.Vec,
      L: typing.List[typing.Union[Form, cpp.fem.Form]],
      a,
      bcs: typing.List[DirichletBC] = [],
      x0: typing.Optional[PETSc.Vec] = None,
      scale: float = 1.0) -> PETSc.Vec:
    """Assemble linear forms into a monolithic vector. The vector is not
    zeroed and it is not finalised, i.e. ghost values are not
    accumulated.

    """
    maps = [(form.function_spaces[0].dofmap.index_map, form.function_spaces[0].dofmap.index_map_bs)
            for form in _create_cpp_form(L)]
    if x0 is not None:
        x0_local = cpp.la.get_local_vectors(x0, maps)
        x0_sub = x0_local
    else:
        x0_local = []
        x0_sub = [None] * len(maps)

    bcs1 = cpp.fem.bcs_cols(_create_cpp_form(a), bcs)
    b_local = cpp.la.get_local_vectors(b, maps)
    for b_sub, L_sub, a_sub, bc in zip(b_local, L, a, bcs1):
        cpp.fem.assemble_vector(b_sub, _create_cpp_form(L_sub))
        cpp.fem.apply_lifting(b_sub, _create_cpp_form(a_sub), bc, x0_local, scale)

    cpp.la.scatter_local_vectors(b, b_local, maps)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    bcs0 = cpp.fem.bcs_rows(_create_cpp_form(L), bcs)
    offset = 0
    b_array = b.getArray(readonly=False)
    for submap, bc, _x0 in zip(maps, bcs0, x0_sub):
        size = submap[0].size_local * submap[1]
        cpp.fem.set_bc(b_array[offset:offset + size], bc, _x0, scale)
        offset += size

    return b


# -- Matrix assembly ---------------------------------------------------------


@functools.singledispatch
def assemble_matrix(a: typing.Union[Form, cpp.fem.Form],
                    bcs: typing.List[DirichletBC] = [],
                    diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear form into a matrix. The returned matrix is not
    finalised, i.e. ghost values are not accumulated.

    """
    _a = _create_cpp_form(a)
    A = create_matrix(_a)
    return assemble_matrix(A, _a, bcs, diagonal)


@assemble_matrix.register(PETSc.Mat)
def _(A: PETSc.Mat,
      a: typing.Union[Form, cpp.fem.Form],
      bcs: typing.List[DirichletBC] = [],
      diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear form into a matrix. The returned matrix is not
    finalised, i.e. ghost values are not accumulated.

    """
    _a = _create_cpp_form(a)
    cpp.fem.assemble_matrix_petsc(A, _a, bcs)
    if _a.function_spaces[0].id == _a.function_spaces[1].id:
        A.assemblyBegin(PETSc.Mat.AssemblyType.FLUSH)
        A.assemblyEnd(PETSc.Mat.AssemblyType.FLUSH)
        cpp.fem.insert_diagonal(A, _a.function_spaces[0], bcs, diagonal)
    return A


# FIXME: Revise this interface
@functools.singledispatch
def assemble_matrix_nest(a: typing.List[typing.List[typing.Union[Form, cpp.fem.Form]]],
                         bcs: typing.List[DirichletBC] = [], mat_types=[],
                         diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear forms into matrix"""
    _a = _create_cpp_form(a)
    A = create_matrix_nest(_a, mat_types)
    A.zeroEntries()
    return assemble_matrix_nest(A, _a, bcs, diagonal)


@assemble_matrix_nest.register(PETSc.Mat)
def _(A: PETSc.Mat,
      a: typing.List[typing.List[typing.Union[Form, cpp.fem.Form]]],
      bcs: typing.List[DirichletBC] = [],
      diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear forms into matrix"""
    _a = _create_cpp_form(a)

    # Assemble form
    for i, a_row in enumerate(_a):
        for j, a_block in enumerate(a_row):
            if a_block is not None:
                Asub = A.getNestSubMatrix(i, j)
                cpp.fem.assemble_matrix_petsc(Asub, a_block, bcs)

    # Flush to enable switch from add to set in the matrix
    A.assemble(PETSc.Mat.AssemblyType.FLUSH)

    # Set diagonal
    for i, a_row in enumerate(_a):
        for j, a_block in enumerate(a_row):
            if a_block is not None:
                Asub = A.getNestSubMatrix(i, j)
                if a_block.function_spaces[0].id == a_block.function_spaces[1].id:
                    cpp.fem.insert_diagonal(Asub, a_block.function_spaces[0], bcs, diagonal)

    return A


# FIXME: Revise this interface
@functools.singledispatch
def assemble_matrix_block(a: typing.List[typing.List[typing.Union[Form, cpp.fem.Form]]],
                          bcs: typing.List[DirichletBC] = [],
                          diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear forms into matrix"""
    _a = _create_cpp_form(a)
    A = create_matrix_block(_a)
    return assemble_matrix_block(A, _a, bcs, diagonal)


@assemble_matrix_block.register(PETSc.Mat)
def _(A: PETSc.Mat,
      a: typing.List[typing.List[typing.Union[Form, cpp.fem.Form]]],
      bcs: typing.List[DirichletBC] = [],
      diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear forms into matrix"""
    _a = _create_cpp_form(a)
    V = _get_block_function_spaces(_a)
    is_rows = cpp.la.create_petsc_index_sets([(Vsub.dofmap.index_map, Vsub.dofmap.index_map_bs) for Vsub in V[0]])
    is_cols = cpp.la.create_petsc_index_sets([(Vsub.dofmap.index_map, Vsub.dofmap.index_map_bs) for Vsub in V[1]])

    # Assemble form
    for i, a_row in enumerate(_a):
        for j, a_sub in enumerate(a_row):
            if a_sub is not None:
                Asub = A.getLocalSubMatrix(is_rows[i], is_cols[j])
                cpp.fem.assemble_matrix_petsc_unrolled(Asub, a_sub, bcs)
                A.restoreLocalSubMatrix(is_rows[i], is_cols[j], Asub)

    # Flush to enable switch from add to set in the matrix
    A.assemble(PETSc.Mat.AssemblyType.FLUSH)

    # Set diagonal
    for i, a_row in enumerate(_a):
        for j, a_sub in enumerate(a_row):
            if a_sub is not None:
                Asub = A.getLocalSubMatrix(is_rows[i], is_cols[j])
                if a_sub.function_spaces[0].id == a_sub.function_spaces[1].id:
                    cpp.fem.insert_diagonal(Asub, a_sub.function_spaces[0], bcs, diagonal)
                A.restoreLocalSubMatrix(is_rows[i], is_cols[j], Asub)

    return A


# -- Modifiers for Dirichlet conditions ---------------------------------------

def apply_lifting(b: PETSc.Vec,
                  a: typing.List[typing.Union[Form, cpp.fem.Form]],
                  bcs: typing.List[typing.List[DirichletBC]],
                  x0: typing.Optional[typing.List[PETSc.Vec]] = [],
                  scale: float = 1.0) -> None:
    """Modify RHS vector b for lifting of Dirichlet boundary conditions.
    It modifies b such that:

        b <- b - scale * A_j (g_j - x0_j)

    where j is a block (nest) index. For a non-blocked problem j = 0.
    The boundary conditions bcs are on the trial spaces V_j. The forms
    in [a] must have the same test space as L (from which b was built),
    but the trial space may differ. If x0 is not supplied, then it is
    treated as zero.

    Ghost contributions are not accumulated (not sent to owner). Caller
    is responsible for calling VecGhostUpdateBegin/End.
    """
    with contextlib.ExitStack() as stack:
        x0 = [stack.enter_context(x.localForm()) for x in x0]
        x0_r = [x.array_r for x in x0]
        b_local = stack.enter_context(b.localForm())
        cpp.fem.apply_lifting(b_local.array_w, _create_cpp_form(a), bcs, x0_r, scale)


def apply_lifting_nest(b: PETSc.Vec,
                       a: typing.List[typing.List[typing.Union[Form, cpp.fem.Form]]],
                       bcs: typing.List[DirichletBC],
                       x0: typing.Optional[PETSc.Vec] = None,
                       scale: float = 1.0) -> PETSc.Vec:
    """Modify nested vector for lifting of Dirichlet boundary conditions.

    """
    x0 = [] if x0 is None else x0.getNestSubVecs()
    _a = _create_cpp_form(a)
    bcs1 = cpp.fem.bcs_cols(_a, bcs)
    for b_sub, a_sub, bc1 in zip(b.getNestSubVecs(), _a, bcs1):
        apply_lifting(b_sub, a_sub, bc1, x0, scale)
    return b


def set_bc(b: PETSc.Vec,
           bcs: typing.List[DirichletBC],
           x0: typing.Optional[PETSc.Vec] = None,
           scale: float = 1.0) -> None:
    """Insert boundary condition values into vector. Only local (owned)
    entries are set, hence communication after calling this function is
    not required unless ghost entries need to be updated to the boundary
    condition value.

    """
    if x0 is not None:
        x0 = x0.array_r
    cpp.fem.set_bc(b.array_w, bcs, x0, scale)


def set_bc_nest(b: PETSc.Vec,
                bcs: typing.List[typing.List[DirichletBC]],
                x0: typing.Optional[PETSc.Vec] = None,
                scale: float = 1.0) -> None:
    """Insert boundary condition values into nested vector. Only local (owned)
    entries are set, hence communication after calling this function is
    not required unless the ghost entries need to be updated to the
    boundary condition value.

    """
    _b = b.getNestSubVecs()
    x0 = len(_b) * [None] if x0 is None else x0.getNestSubVecs()
    for b_sub, bc, x_sub in zip(_b, bcs, x0):
        set_bc(b_sub, bc, x_sub, scale)
