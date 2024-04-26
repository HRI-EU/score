# -*- coding: utf-8 -*-
#
#  Functions to do stacked rejection plots.
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

from typing import Optional
import itertools
from enum import Enum
import math
import numpy
import pandas

from matplotlib import pyplot as plt
from matplotlib import axes
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Polygon
import seaborn

pandas.options.mode.copy_on_write = True

seaborn.set_theme(style="whitegrid", font_scale=1.2, palette=seaborn.color_palette("colorblind", desat=1))
_color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]


class Type(Enum):
    """Define the basic plot type."""

    STACK = 0
    """Draw confusions as stack."""
    PIE = 1
    """Draw confusions as pie."""
    REJECT = 2
    """Draw confusions as some standard reject plots."""

    def __str__(self):
        return str(self.name.lower())


class Order(Enum):
    """Define the order of the confusion columns."""

    AS_IS = 0
    """Keep the order as occurring in the original data."""
    CORRECT_LAST = 1
    """Put the correct columns at the end."""

    def __str__(self):
        return str(self.name.lower())


class Alignment(Enum):
    """Define the alignment of the confusion columns."""

    BOTTOM = 0
    """The zero line is at the bottom of the stack."""
    CORRECT_START = 1
    """The zero line is before the first correct confusion column."""
    CORRECT_CENTER = 2
    """The zero line is in the center of all correct confusion columns."""

    def __str__(self):
        return str(self.name.lower())


def evaluate_confusion(data: pandas.DataFrame, condense_errors: bool = False) -> pandas.DataFrame:
    """
    Evaluate the confusion inside the data as preparation for plotting.

    This method sorts the data by descending certainty. For each certainty it counts the confusions
    between each pair of ground_truth and prediction.

    :param data: The dataframe with the columns 'ground_truth', 'prediction', and 'certainty'.
    :param condense_errors: Whether to treat all wrong predictions of a ground truth class as a single confusion case.
    :return: The sorted dataframe with confusion columns appended.
    """
    # Sort the data based on decreasing certainty.
    data = data.sort_values(by=["certainty"], ascending=False, ignore_index=True)

    if condense_errors and (len(set(data["ground_truth"])) > 2):
        # For wrong predictions replace predicted label with 'x'.
        data["prediction"] = numpy.where(data["ground_truth"] == data["prediction"], data["prediction"], "x")
    if condense_errors and (len(set(data["ground_truth"])) < 3):
        print("INFO: The condense flag -c is ignored since the data is binary.")

    # Make a hot-one-encoding of all the existent confusions between ground_truth and prediction.
    data["confusion"] = data["ground_truth"].astype(str) + "_" + data["prediction"].astype(str)
    data = pandas.get_dummies(data, columns=["confusion"], dtype=int)
    confusion_columns = __get_confusion_columns(data)

    # Accumulate the hot-one-encoding over rows.
    data[confusion_columns] = data[confusion_columns].cumsum(axis="rows")

    # Rows with identical certainty should have the same values.
    data[confusion_columns] = data.groupby("certainty")[confusion_columns].transform("last")

    # Compute the number of samples.
    data["number"] = data[confusion_columns].sum(axis=1)

    return data


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
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.set_title("data points")
    plt.show()


def plot(data: pandas.DataFrame, plot_type: Type, **kwargs) -> None:
    if plot_type == Type.PIE:
        plot_pie(
            data,
            order=kwargs["order"],
            alignment=kwargs["alignment"],
            angle_offset=numpy.pi / 2,
            save_path=kwargs["save_path"],
            figure_kwargs=kwargs["figure_kwargs"],
            legend_kwargs=kwargs["legend_kwargs"],
        )
    elif plot_type == Type.STACK:
        plot_stack(
            data,
            order=kwargs["order"],
            alignment=kwargs["alignment"],
            normalize=kwargs["normalize"],
            save_path=kwargs["save_path"],
            figure_kwargs=kwargs["figure_kwargs"],
            legend_kwargs=kwargs["legend_kwargs"],
        )
    elif plot_type == Type.REJECT:
        plot_reject_curves(data, save_path=kwargs["save_path"])
    else:
        raise AssertionError(f"Unknown plot type '{plot_type}'. Known plot types are {list(Type)}.")


def plot_reject_curves(data: pandas.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Plot some standard reject curves.

    :param data: The evaluated dataframe as provided by :func:`evaluate_confusions`.
    :param save_path: The path where to store the figure instead of showing it.
    :return:
    """
    # Get number of correctly classified samples.
    correct_confusion_columns = __get_correct_confusion_columns(data)
    correct = data[correct_confusion_columns].sum(axis=1)
    acceptance_rate = data["number"] / len(data["number"])

    plt.figure(figsize=(6, 5))
    ax = plt.gca()

    # Plot with evaluation measure.
    accuracy = correct / data["number"]
    # For binary settings only
    if len(set(data.ground_truth.values)) == 2:
        precision = data.confusion_1_1 / (data.confusion_1_1 + data.confusion_2_1)
        recall = data.confusion_1_1 / (data.confusion_1_1 + data.confusion_1_2)
        ax.plot(acceptance_rate, precision, label="precision", linestyle="dotted", zorder=20)
        ax.plot(acceptance_rate, recall, label="recall", linestyle="dashdot", zorder=20)

    ax.plot(acceptance_rate, accuracy, label="accuracy", linestyle="solid", zorder=20)
    ax.plot(acceptance_rate, 1 - accuracy, label="error rate", linestyle="dashed", zorder=20)
    ax.set_xlabel("acceptance rate")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.invert_xaxis()
    ax.set_ylabel("standard measures")
    ax.legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_stack(
    data: pandas.DataFrame,
    order: Order = Order.AS_IS,
    alignment: Alignment = Alignment.BOTTOM,
    normalize: bool = False,
    save_path: Optional[str] = None,
    figure_kwargs: Optional[dict] = None,
    legend_kwargs: Optional[dict] = None,
) -> None:
    """
    Plot confusions as a stack.

    :param data: The evaluated dataframe as provided by :func:`evaluate_confusions`.
    :param order: Define the order in which confusions are shown. See :class:`Order`.
    :param alignment: Define where to put the zero line. See :class:`Alignment`.
    :param normalize: Whether to normalize the confusions with the total number of samples.
    :param save_path: The path where to store the figure instead of showing it.
    :param figure_kwargs: Override the default figure kwargs.
    :param legend_kwargs: Override the default legend kwargs.
    :return:
    """
    if figure_kwargs is None:
        figure_kwargs = {"figsize": (8, 5)}
    if legend_kwargs is None:
        legend_kwargs = {"bbox_to_anchor": (1.05, 1.0), "loc": 2, "borderaxespad": 0.0}
    hatches = itertools.cycle(["/", "+", "-", "//", "x", "\\", "*", "o", "0", "."])
    plt.figure(**figure_kwargs)
    confusions = __prepare_stacked_confusions(data, order, alignment, normalize)
    acceptance_rate = data["number"] / len(data["number"])
    previous_confusion = confusions[confusions.columns[0]]
    for column_name in confusions.columns[1:]:
        confusion = confusions[column_name]
        color = __get_color(column_name, len(data.ground_truth.unique()))
        if order == Order.CORRECT_LAST and __is_border_case(column_name, confusions.columns.values.tolist()):
            line_width = 2.5
        else:
            line_width = 1
        plt.plot(acceptance_rate, confusion, "white", linewidth=line_width)
        plt.fill_between(
            acceptance_rate,
            previous_confusion,
            confusion,
            color=color,
            alpha=0.8,
            label=column_name,
            zorder=0,
            hatch=next(hatches),
            edgecolor="white",
        )
        previous_confusion = confusion

    ax = plt.gca()
    ax.set_xlim([0.0, 1.0])
    ax.invert_xaxis()
    ax.set_xlabel("acceptance rate")
    ax.set_ylabel(f"stacked confusion {'rates' if normalize else 'counts'}")
    ax.autoscale(enable=True, axis="y", tight=True)
    if not normalize:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    if legend_kwargs.pop("visible", True):
        plt.legend(**legend_kwargs)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_pie(
    data: pandas.DataFrame,
    order: Order = Order.AS_IS,
    alignment: Alignment = Alignment.BOTTOM,
    angle_range: float = numpy.pi * 2,
    angle_offset: float = 0.0,
    save_path: Optional[str] = None,
    figure_kwargs: Optional[dict] = None,
    legend_kwargs: Optional[dict] = None,
) -> None:
    """
    Plot confusions as a pie.

    :param data: The evaluated dataframe as provided by :func:`evaluate_confusions`.
    :param order: Define the order in which confusions are shown. See :class:`Order`.
    :param alignment: Define where to put the zero line. See :class:`Alignment`.
    :param angle_range: The angle range to use in radians.
    :param angle_offset: The angle offset to use in radians.
    :param save_path: The path where to store the figure instead of showing it.
    :param figure_kwargs: Override the default figure kwargs.
    :param legend_kwargs: Override the default legend kwargs.
    :return:
    """
    if figure_kwargs is None:
        figure_kwargs = {"figsize": (8, 5)}
    if legend_kwargs is None:
        legend_kwargs = {"bbox_to_anchor": (1.1, 1.0), "loc": 2, "borderaxespad": 0.0}

    hatches = itertools.cycle(["/", "+", "-", "//", "x", "\\", "*", "o", "0", "."])
    confusions = __prepare_stacked_confusions(data, order, alignment, True)
    figure = plt.figure(**figure_kwargs)
    ax = figure.add_subplot(111, projection="polar")
    acceptance_rate = data["number"] / len(data["number"])
    previous_column = __get_interpolated_polar_data(confusions[confusions.columns[0]], acceptance_rate, angle_range)
    for column_name in confusions.columns[1:]:
        current_column = __get_interpolated_polar_data(confusions[column_name], acceptance_rate, angle_range)
        color = __get_color(column_name, len(data.ground_truth.unique()))
        if order == Order.CORRECT_LAST and __is_border_case(column_name, confusions.columns.values.tolist()):
            line_width = 2.5
        else:
            line_width = 1
        plt.polar(current_column["angles"], current_column["radii"], "white", linewidth=line_width)
        __draw_filled_pie_segment(previous_column, current_column, column_name, ax, color, next(hatches))
        previous_column = current_column

    ax.set_theta_offset(angle_offset)
    ax.set_thetamax(math.degrees(angle_range))
    percentages = numpy.linspace(0.0, 1.0, 10, endpoint=False)
    if angle_range < numpy.pi * 2:
        percentages = numpy.append(percentages, 1.0)
    ax.set_xticks(percentages * angle_range)
    ax.set_xticklabels([f"{value:.1f}" for value in percentages])
    ax.set_ylim([0.0, 1.0])

    if legend_kwargs.pop("visible", True):
        plt.legend(**legend_kwargs)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def __get_color(column_name: str, class_number: int) -> tuple[float, float, float]:
    """
    Get a color from the standard color cycle based on the given column name.

    This method should ensure that color assignment is consistent across different plots.
    Correct classifications are assigned to colors with an index smaller than the given class number.
    Wrong classifications are assigned to colors with higher indices.

    :param column_name: The name of a confusions column, like 'confusion_1_2'.
    :param class_number: The number of classes in the data.
    :return: The determined color.
    """
    tokens = column_name.split("_")
    ground_truth_index = int(tokens[1]) - 1
    if tokens[1] == tokens[2]:
        prediction_index = 0
    elif tokens[2] == "x":
        prediction_index = 1
    else:
        prediction_index = int(tokens[2]) - 1
        if ground_truth_index > prediction_index:
            prediction_index += 1

    color_index = ground_truth_index + class_number * prediction_index
    return _color_cycle[color_index % len(_color_cycle)]


def __get_confusion_columns(data: pandas.DataFrame) -> list[str]:
    return [column for column in data.columns if column.startswith("confusion_")]


def __is_correct_confusion(confusion: str) -> bool:
    tokens = confusion.split("_")
    return len(tokens) == 3 and tokens[1] == tokens[2]


def __get_correct_confusion_columns(data: pandas.DataFrame) -> list[str]:
    return [column for column in data.columns if column.startswith("confusion_") and __is_correct_confusion(column)]


def __get_wrong_confusion_columns(data: pandas.DataFrame) -> list[str]:
    return [column for column in data.columns if column.startswith("confusion_") and not __is_correct_confusion(column)]


def __get_first_correct_confusion_index(data: pandas.DataFrame) -> Optional[int]:
    for column_index, column in enumerate(data.columns):
        if column.startswith("confusion_") and __is_correct_confusion(column):
            return column_index

    return None


def __is_border_case(column_name: str, column_names: list[str]) -> bool:
    column_index = column_names.index(column_name)
    return __is_correct_confusion(column_name) != __is_correct_confusion(
        column_names[(column_index + 1) % len(column_names)]
    )


def __prepare_stacked_confusions(
    data: pandas.DataFrame, order: Order = Order.AS_IS, alignment: Alignment = Alignment.BOTTOM, normalize: bool = False
) -> pandas.DataFrame:
    """
    Prepare a stacked confusion plot by computing the cumulative sum over the confusion columns.

    This method computes the  cumulative sum over the confusion columns.
    In addition, it provides means to order, align, and normalize the stack.

    :param data: The evaluated dataframe as provided by :func:`evaluate_confusions`.
    :param order: Define the order in which confusions are shown. See :class:`Order`.
    :param alignment: Define where to put the zero line. See :class:`Alignment`.
    :param normalize: Whether to normalize the confusions with the total number of samples.
    :return:
    """
    # Extract a copy of the confusion columns and potentially order them.
    if order == Order.AS_IS:
        confusions = data[__get_confusion_columns(data)].copy()
    elif order == Order.CORRECT_LAST:
        confusions = data[__get_wrong_confusion_columns(data) + __get_correct_confusion_columns(data)].copy()
    else:
        raise AssertionError(f"Unknown order '{order}'. Known orders are {list(Order)}.")

    # Accumulate over columns to get the stacked confusions.
    confusions = confusions.cumsum(axis=1)

    # Insert a special confusion column with zeros as base of the stack.
    confusions.insert(0, "confusion", 0)

    # Adjust where to put the zero line by subtracting an offset.
    if alignment not in Alignment:
        raise AssertionError(f"Unknown alignment '{alignment}'. Known alignments are {list(Alignment)}.")

    # Align the stack by subtracting an offset.
    alignment_offset = None
    if alignment in [Alignment.CORRECT_START, Alignment.CORRECT_CENTER]:
        if order != Order.CORRECT_LAST:
            raise AssertionError(f"Alignment '{alignment}' only works for order '{Order.CORRECT_LAST}'.")

        first_correct_confusion_index = __get_first_correct_confusion_index(confusions)
        if first_correct_confusion_index is not None:
            # Use column before first correct confusion as offset.
            alignment_offset = confusions.iloc[:, first_correct_confusion_index - 1]
            if alignment == Alignment.CORRECT_CENTER:
                # Use mean between confusion before first correct confusion and last correct confusion.
                alignment_offset = (alignment_offset + confusions.iloc[:, -1]) / 2

    if alignment_offset is not None:
        confusions = confusions.sub(alignment_offset, axis=0)

    # Normalize the confusion counts with the overall sample number.
    if normalize:
        confusions = confusions.div(data["number"], axis=0)

    return confusions


def __get_interpolated_polar_data(
    confusion: pandas.Series,
    number: pandas.Series,
    angle_range: float,
    maximal_angle_step_size: float = 0.1,
) -> pandas.DataFrame:
    interpolated = pandas.DataFrame()
    # Use the number as radius.
    interpolated["radii"] = number
    # Convert the (already normalized) confusion to an angle.
    interpolated["angles"] = confusion * angle_range
    # Determine how many interpolation steps (including the original row) are required between successive angles.
    interpolated["steps"] = numpy.maximum(
        numpy.ceil(interpolated["angles"].diff().fillna(0).abs() / maximal_angle_step_size).astype("int"), 1
    )
    # Change the index of the original data by taking the cumulated interpolation steps into account.
    interpolated.index = interpolated["steps"].cumsum() - 1
    # Add NAN rows between the original data.
    interpolated = interpolated.reindex(index=range(interpolated["steps"].sum()))
    # Interpolate the NAN rows linearly.
    return interpolated[["radii", "angles"]].interpolate(axis=0)


def __draw_filled_pie_segment(
    column_1: pandas.DataFrame,
    column_2: pandas.DataFrame,
    label: str,
    ax: axes.Axes,
    color: tuple[float, float, float],
    hatch: str,
):
    angles_1 = column_1["angles"].to_numpy()
    radii_1 = column_1["radii"].to_numpy()
    angles_2 = column_2["angles"].to_numpy()
    radii_2 = column_2["radii"].to_numpy()

    # First: go from center to outside along angles_1.
    vertices_1 = numpy.c_[angles_1, radii_1]

    # Second: go along arc from most outside angle_1 to most outside angle_2.
    arc_angles = numpy.linspace(angles_1[-1], angles_2[-1], 101)
    arc_radii = numpy.ones_like(arc_angles) * radii_1[-1]
    vertices_2 = numpy.c_[arc_angles, arc_radii]

    # Third: go from outside to center along angles_2, i.e. in reversed order.
    vertices_3 = numpy.c_[angles_2, radii_2][::-1]

    # Fourth: go along arc from most inside angle_2 to most inside angle_1.
    arc_angles = numpy.linspace(angles_2[0], angles_1[0], 101)
    arc_radii = numpy.ones_like(arc_angles) * radii_1[0]
    vertices_4 = numpy.c_[arc_angles, arc_radii]

    # Draw the filled polygon that is made of the four parts.
    return ax.add_artist(
        Polygon(
            numpy.concatenate([vertices_1, vertices_2, vertices_3, vertices_4]),
            fc=color,
            alpha=0.80,
            zorder=0,
            label=label,
            hatch=hatch,
        )
    )
