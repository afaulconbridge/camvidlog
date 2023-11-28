# SPDX-FileCopyrightText: 2023-present U.N. Owen <void@some.where>
#
# SPDX-License-Identifier: MIT

import os


def get_data(path):
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), "data", path)
