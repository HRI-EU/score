#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Generate some gaussian data with Bayes confidence estimation.
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

import argparse
import numpy
import pandas
from matplotlib import pyplot as plt


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


def bayes_confidence(data: pandas.DataFrame) -> tuple[pandas.Series, pandas.Series]:
    """Compute the Bayesian confidence (for 2D data only)."""
    required_columns = ["x", "y", "ground_truth", "mu_x", "mu_y", "sigma_x", "sigma_y"]
    if not data.columns.isin(required_columns).all():
        raise AssertionError(f"A required column is missing.\nExpected {required_columns}\nGot { data.columns}")

    cluster_columns = ["ground_truth", "mu_x", "mu_y", "sigma_x", "sigma_y"]

    # Calculate the cluster/class probability P(C_i).
    probability_clusters_ii = data[cluster_columns].value_counts(sort=False, normalize=True)

    # Get characteristics of each class/cluster (multimodal classes have multiple clusters).
    characteristic_clusters_ii = data[cluster_columns].drop_duplicates(keep="first")

    # Calculate conditional densities p(x | C_i) for each class and cluster independently for all data points.
    conditional_densities = pandas.DataFrame()
    for index, row in characteristic_clusters_ii.iterrows():
        conditional_densities[f"conditional_density_{index}"] = (
            1.0
            / (2.0 * numpy.pi * row.sigma_x * row.sigma_y)
            * numpy.exp(-0.5 * (((data.x - row.mu_x) / row.sigma_x) ** 2 + ((data.y - row.mu_y) / row.sigma_y) ** 2))
        )

    # Calculate probabilities P(x) and P(C_i | x).
    confidence_help_1 = pandas.DataFrame()
    for index, (_, column) in enumerate(conditional_densities.items()):
        confidence_help_1[f"{index}"] = column * probability_clusters_ii.values[index]

    p_x = confidence_help_1.sum(axis=1)

    # Sum all P(C_i| x) that belong to the same class.
    label_cluster = characteristic_clusters_ii["ground_truth"]
    confidence_help_2 = pandas.DataFrame()
    for label in label_cluster.unique():
        confidence_help_2[f"{label}"] = confidence_help_1.iloc[:, label_cluster.values == label].sum(axis=1)

    confidences = confidence_help_2.divide(p_x, axis=0).max(axis=1)
    predicted_labels = confidence_help_2.divide(p_x, axis=0).idxmax(axis=1)

    return confidences, predicted_labels


def plot_class_data(data: pandas.DataFrame) -> None:
    """
    Plot some Gaussian data.

    :param data: A dataframe containing some Gaussian data.
    :return:
    """
    ax = None
    for label in data.ground_truth.unique():
        ax = data.loc[data.ground_truth == label, ["x", "y"]].plot(
            x="x", y="y", marker="+", linestyle="none", ax=ax, label=f"data points of class {label}"
        )
        data.loc[data.ground_truth == label, ["mu_x", "mu_y"]].plot(
            x="mu_x", y="mu_y", marker="s", linestyle="none", ax=ax, label=f"Gaussian centers of class {label}"
        )
    if ax is not None:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.axis("equal")
        ax.set_title("data points")
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate some gaussian data with Bayes confidence estimation.",
    )
    parser.add_argument("file", type=str, help="save the generated data to that csv file")
    args = parser.parse_args()

    # Create gaussian data.
    data = gaussian_clusters(
        [
            {"mu": [0.0, 0.0], "sigma": [2.0, 1.0], "number": 40, "label": 1},
            {"mu": [2.0, 0.0], "sigma": [1.0, 0.5], "number": 100, "label": 2},
            {"mu": [4.0, 0.0], "sigma": [2.0, 1.0], "number": 40, "label": 1},
            {"mu": [6.0, 0.0], "sigma": [1.0, 1.0], "number": 60, "label": 2},
        ]
    )

    plot_class_data(data)

    data["certainty"], data["prediction"] = bayes_confidence(data)
    data.sort_values(by=["certainty"], ignore_index=True).to_csv(args.file)


if __name__ == "__main__":
    main()
