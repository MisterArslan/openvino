# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

ie_add_sample(NAME memleak
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/infer_request_wrap.hpp"
              OPENCV_DEPENDENCIES imgcodecs imgproc highgui core)
