# -*- coding: utf-8 -*-
# Copyright (C) 2020 Francesco Ballarin
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import typing

from dolfinx import cpp
from dolfinx.fem.dofmap import DofMap


class DofMapRestriction(cpp.fem.DofMapRestriction):
    def __init__(
            self,
            dofmap: typing.Union[cpp.fem.DofMap, DofMap],
            restriction: typing.List[int]):
        """Restriction of a DofMap to a list of active degrees of freedom
        """
        # Extract cpp dofmap
        try:
            _dofmap = dofmap._cpp_object
        except AttributeError:
            _dofmap = dofmap
        super().__init__(_dofmap, restriction)
