#   Copyright (C) 2017 Mark Niehues, Stefaan Hessmann
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#


import numpy as np


class Kernels:
    """
    Class that holds different Kernels
    """
    def __init__(self, gamma):
        self.gamma = gamma
        self.kernels = {
            "rbf" : self.kernel_rbf,
            "linear": self.kernel_lin}

    def get_kernel(self, kernel_name):
        return self.kernels[kernel_name]

    def kernel_lin(self, x, y):
        """
        Linear kernel
        """
        return x.dot(y)

    def kernel_rbf(self, x, y):
        """
        RBF Kernel
        """
        d = x - y
        return np.exp(-np.dot(d, d) * self.gamma)