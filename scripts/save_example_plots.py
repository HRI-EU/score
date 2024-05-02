#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Save some example plots that are used in documentation and papers.
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

import argparse

from score import read_data, evaluate_confusion, plot, Order, Alignment, Type


plot_definitions = [
    {"data_path": "data/bayes_2_classes.csv", "type": Type.REJECT, "save_path": "bayes_2_classes_reject.eps"},
    {
        "data_path": "data/bayes_2_classes.csv",
        "type": Type.STACK,
        "save_path": "bayes_2_classes_stack.eps",
        "figure_kwargs": {"figsize": (6, 5)},
        "legend_kwargs": {"bbox_to_anchor": (0.96, 0.98), "loc": 1, "borderaxespad": 0.0},
    },
    {
        "data_path": "data/bayes_2_classes.csv",
        "type": Type.STACK,
        "order": Order.CORRECT_LAST,
        "save_path": "bayes_2_classes_stack_ordered.eps",
        "figure_kwargs": {"figsize": (6, 5)},
        "legend_kwargs": {"bbox_to_anchor": (0.96, 0.98), "loc": 1, "borderaxespad": 0.0},
    },
    {
        "data_path": "data/bayes_2_classes.csv",
        "type": Type.STACK,
        "order": Order.CORRECT_LAST,
        "alignment": Alignment.CORRECT_START,
        "save_path": "bayes_2_classes_stack_ordered_aligned.eps",
        "figure_kwargs": {"figsize": (6, 5)},
        "legend_kwargs": {"bbox_to_anchor": (0.96, 0.98), "loc": 1, "borderaxespad": 0.0},
    },
    {
        "data_path": "data/bayes_2_classes.csv",
        "type": Type.STACK,
        "order": Order.CORRECT_LAST,
        "alignment": Alignment.CORRECT_START,
        "normalize": True,
        "save_path": "bayes_2_classes_stack_ordered_aligned_normalized.eps",
        "figure_kwargs": {"figsize": (6, 5)},
        "legend_kwargs": {"bbox_to_anchor": (0.96, 0.59), "loc": 1, "borderaxespad": 0.0},
    },
    {
        "data_path": "data/bayes_2_classes.csv",
        "type": Type.PIE,
        "order": Order.CORRECT_LAST,
        "alignment": Alignment.CORRECT_CENTER,
        "normalize": True,
        "save_path": "bayes_2_classes_pie.eps",
        "figure_kwargs": {"figsize": (6, 5)},
        "legend_kwargs": {"bbox_to_anchor": (1.06, 0.79), "loc": 1, "borderaxespad": 0.0},
    },
    {
        "data_path": "data/bayes_3_classes.csv",
        "type": Type.STACK,
        "order": Order.CORRECT_LAST,
        "alignment": Alignment.CORRECT_START,
        "normalize": True,
        "save_path": "bayes_3_classes_stack_ordered_aligned_normalized.eps",
        "figure_kwargs": {"figsize": (6, 5)},
        "legend_kwargs": {"bbox_to_anchor": (0.47, 0.99), "loc": 1, "borderaxespad": 0.0},
    },
    {
        "data_path": "data/bayes_3_classes.csv",
        "type": Type.STACK,
        "order": Order.CORRECT_LAST,
        "alignment": Alignment.CORRECT_START,
        "normalize": True,
        "condense_errors": True,
        "save_path": "bayes_3_classes_stack_ordered_aligned_normalized_condensed.eps",
        "figure_kwargs": {"figsize": (6, 5)},
        "legend_kwargs": {"bbox_to_anchor": (0.98, 0.83), "loc": 1, "borderaxespad": 0.0},
    },
]


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Plot the examples from the paper."
    )
    parser.add_argument("--extension", "-e", default="eps", help="define the format to save the plots as")
    args = parser.parse_args()

    for plot_definition in plot_definitions:
        if "save_path" in plot_definition:
            plot_definition["save_path"] = plot_definition["save_path"].replace(".eps", f".{args.extension}")

        data = read_data(plot_definition["data_path"])
        evaluated_data = evaluate_confusion(data, condense_errors=plot_definition.get("condense_errors", False))
        plot(
            evaluated_data,
            plot_definition["type"],
            order=plot_definition.get("order", Order.AS_IS),
            alignment=plot_definition.get("alignment", Alignment.BOTTOM),
            normalize=plot_definition.get("normalize", False),
            save_path=plot_definition["save_path"],
            figure_kwargs=plot_definition.get("figure_kwargs"),
            legend_kwargs=plot_definition.get("legend_kwargs"),
        )


if __name__ == "__main__":
    main()
