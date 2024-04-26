#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Example to show usage of the module.
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

import pandas
import numpy

from confusion_reject_plot import evaluate_confusion, plot_pie, plot_stack, plot_reject_curves, Order, Alignment


def main():
    # Make dataframe with simple two-class data.
    data = {
        "ground_truth": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2],
        "prediction": [1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 2],
        "certainty": [0.6, 0.9, 0.8, 0.5, 0.2, 0.8, 0.6, 0.9, 0.9, 0.1, 0.3, 0.4, 0.5, 0.8, 0.7, 0.4, 0.8, 0.7, 0.6],
    }
    data = pandas.DataFrame(data)

    # Evaluate the confusions.
    evaluated_data = evaluate_confusion(data)

    # Do some plots.
    plot_pie(evaluated_data, order=Order.CORRECT_LAST, alignment=Alignment.CORRECT_CENTER, angle_offset=numpy.pi / 2)
    plot_stack(evaluated_data, order=Order.CORRECT_LAST, alignment=Alignment.CORRECT_START)
    plot_stack(evaluated_data, order=Order.CORRECT_LAST, alignment=Alignment.CORRECT_START, normalize=True)
    plot_reject_curves(evaluated_data)


if __name__ == "__main__":
    main()
