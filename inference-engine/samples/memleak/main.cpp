//  Copyright (C) 2020 Intel Corporation
//  SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <string>
#include <map>

#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

#include "infer_request_wrap.hpp"

constexpr size_t batch_size = 1;
constexpr size_t num_of_requests = 1;

const std::string device = "GPU";
const std::string format = "BGRx";

using namespace InferenceEngine;

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Usage : ./memleak <path_to_model>" << std::endl;
        return EXIT_FAILURE;
    }

    /* read network */
    const std::string model_path{argv[1]};

    InferenceEngine::Core core;
    InferenceEngine::CNNNetwork network = core.ReadNetwork(model_path);

    network.setBatchSize(batch_size);

    /* pre-process */
    InputInfo::Ptr input_info = network.getInputsInfo().begin()->second;
    std::string input_name = network.getInputsInfo().begin()->first;

    input_info->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
    input_info->getPreProcess().setColorFormat(ColorFormat::BGR);

    input_info->setLayout(Layout::NCHW);
    input_info->setPrecision(Precision::U8);

    DataPtr output_info = network.getOutputsInfo().begin()->second;
    std::string output_name = network.getOutputsInfo().begin()->first;

    output_info->setPrecision(Precision::FP32);

    /* load network */
    std::map<std::string, std::string> networkConfig;
    ExecutableNetwork executable_network = core.LoadNetwork(network, device, networkConfig);

    std::vector<InferReqWrap::Ptr> inferRequests;
    inferRequests.reserve(num_of_requests);

    /* inference */
    for (size_t i = 0; i < num_of_requests; i++) {
        inferRequests.push_back(std::make_shared<InferReqWrap>(executable_network, input_name));
    }

    inferRequests[0]->infer();

    using namespace std::chrono;
    minutes working_time{20};
    auto start = high_resolution_clock::now();
    while (duration_cast<minutes>(high_resolution_clock::now() - start) < working_time) {
        inferRequests[0]->infer();
    }

    return 0;
}
