#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Do different plots for data from a csv file.
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
import enum

from confusion_reject_plot import read_data, evaluate_confusion, plot, Order, Alignment, Type


class EnumAction(argparse.Action):
    """
    Argparse action for handling Enums.
    """

    def __init__(self, **kwargs):
        # Pop off the type value
        try:
            enum_type = kwargs.pop("type")
        except KeyError as e:
            raise ValueError("Type must be assigned an Enum when using EnumAction.") from e

        # Ensure an Enum subclass is provided
        if not issubclass(enum_type, enum.Enum):
            raise TypeError("Type must be an Enum when using EnumAction.")

        # Override 'type' with a function that looks up the enum given the name.
        # This is e.g. done when the default is a string.
        # kwargs["type"] = self.__get_type__

        # Generate choices from the Enum
        kwargs.setdefault("choices", tuple(e.name.lower() for e in enum_type))

        super().__init__(**kwargs)

        self._enum = enum_type

    def __get_type__(self, string):
        return self._enum[string.upper()]

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        value = self._enum[values.upper()]
        setattr(namespace, self.dest, value)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Plot csv file."
    )
    parser.add_argument("file", type=str, help="csv file")
    parser.add_argument("--save-path", "-s", type=str, help="save figure using this path")
    parser.add_argument(
        "--condense-errors",
        "-c",
        action="store_true",
        default=False,
        help="neglect with which class samples were confused",
    )
    parser.add_argument(
        "--type",
        "-t",
        type=Type,
        action=EnumAction,
        default=Type.STACK,
        help="define the basic plot type",
    )
    parser.add_argument(
        "--order",
        "-o",
        type=Order,
        action=EnumAction,
        default=Order.AS_IS,
        help="define how to order confusion columns",
    )
    parser.add_argument(
        "--alignment",
        "-a",
        type=Alignment,
        action=EnumAction,
        default=Alignment.BOTTOM,
        help="define how to align confusion columns",
    )
    parser.add_argument(
        "--normalize",
        "-n",
        action="store_true",
        default=False,
        help="normalize confusion counts",
    )
    args = parser.parse_args()

    # Read data from csv file, evaluate confusions, and plot.
    data = read_data(args.file)
    evaluated_data = evaluate_confusion(data, condense_errors=args.condense_errors)
    plot(
        evaluated_data,
        args.type,
        order=args.order,
        alignment=args.alignment,
        normalize=args.normalize,
        save_path=args.save_path,
        figure_kwargs=None,
        legend_kwargs=None,
    )


if __name__ == "__main__":
    main()
