# -*- coding: utf-8 -*-
#
#  Tools to create some toy problem data.
#
#  Copyright (c) Honda Research Institute Europe GmbH
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its
#     contributors may be used to endorse or promote products derived from
#     this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
#  IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
#  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
#  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
#  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#

from __future__ import annotations

import numpy
import pandas


def read_data(file_path: str) -> pandas.DataFrame:
    data = pandas.read_csv(file_path)
    for required_column in ["ground_truth", "prediction", "certainty"]:
        if required_column not in data.columns:
            raise AssertionError(f"Missing column '{required_column}' in '{file_path}'.")

    return data


def create_gauss_data(mu: list, sigma: list, label: int, number: int) -> pandas.DataFrame:
    x = numpy.random.default_rng().normal(mu[0], sigma[0], size=(1, number))[0]
    y = numpy.random.default_rng().normal(mu[1], sigma[1], size=(1, number))[0]

    data = pandas.DataFrame(
        {
            "x": x,
            "y": y,
            "ground_truth": numpy.array([label] * number),
            "mu_x": mu[0] * numpy.ones([number]),
            "mu_y": mu[1] * numpy.ones([number]),
            "sigma_x": sigma[0] * numpy.ones([number]),
            "sigma_y": sigma[1] * numpy.ones([number]),
        }
    )
    return data


def gaussian_clusters(definitions: list[dict]) -> pandas.DataFrame:
    data = [create_gauss_data(**definition) for definition in definitions]
    return pandas.concat(data, ignore_index=True, axis=0)
