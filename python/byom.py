"""
 SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""  # noqa: E501

import os
from argparse import ArgumentParser

from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import (
    FormatConverterOp,
    InferenceOp,
    SegmentationPostprocessorOp,
    VideoStreamReplayerOp,
)
from holoscan.resources import UnboundedAllocator
import cupy as cp
import numpy as np

class DataGenOp(Operator):
    """Print the received signal to the terminal."""

    def setup(self, spec: OperatorSpec):
        spec.output("output_tensor_img")

    def compute(self, op_input, op_output, context):
        out_msg = dict()
        out_msg['source_video'] = cp.random.random((256, 256, 1), np.float32)
        print("Generated")
        signal = op_output.emit(out_msg, "output_tensor_img")

class IdentityOp(Operator):
    """Print the received signal to the terminal."""

    def setup(self, spec: OperatorSpec):
        spec.input("input_tensor")
        spec.output("output_tensor")

    def compute(self, op_input, op_output, context):
        #op_input['input_tensor']
        #cupy_signal = signal['out_tensor']
        in_message = op_input.receive("input_tensor")
        out_msg = dict()
        for k, v in in_message.items():
            breakpoint()
            cp_array = cp.asarray(v)
            out_msg[k] = cp_array
        print("Received")
        signal = op_output.emit(out_msg, "output_tensor")

class PrintSignalOp(Operator):
    """Print the received signal to the terminal."""

    def setup(self, spec: OperatorSpec):
        spec.input("signal")

    def compute(self, op_input, op_output, context):
        signal = op_input.receive("signal")
        cupy_signal = signal['out_tensor']
        #cp_array = signal.asarray()
        print("Received")

class BYOMApp(Application):
    def __init__(self, data):
        """Initialize the application

        Parameters
        ----------
        data : Location to the data
        """

        super().__init__()

        # set name
        self.name = "BYOM App"

        if data == "none":
            data = os.environ.get("HOLOSCAN_INPUT_PATH", "../data")

        self.sample_data_path = data

        self.model_path = os.path.join(os.path.dirname(__file__), "../model")
        self.model_path_map = {
            "byom_model": os.path.join(self.model_path, "identity_model.onnx"),
        }

        self.video_dir = os.path.join(self.sample_data_path, "racerx")
        if not os.path.exists(self.video_dir):
            raise ValueError(f"Could not find video data: {self.video_dir=}")

    def compose(self):
        host_allocator = UnboundedAllocator(self, name="host_allocator")

        source = VideoStreamReplayerOp(
            self, name="replayer", directory=self.video_dir, **self.kwargs("replayer")
        )

        preprocessor = FormatConverterOp(
            self, name="preprocessor", pool=host_allocator, **self.kwargs("preprocessor")
        )

        inference = InferenceOp(
            self,
            name="inference",
            allocator=host_allocator,
            model_path_map=self.model_path_map,
            **self.kwargs("inference"),
        )

        postprocessor = SegmentationPostprocessorOp(
            self, name="postprocessor", allocator=host_allocator, **self.kwargs("postprocessor")
        )


        sinkop = PrintSignalOp(self, name="Sink")
        genop = DataGenOp(self, name="Generator")
        # Define the workflow
        self.add_flow(genop, inference, {("", "receivers")})
        self.add_flow(inference, postprocessor, {("transmitter", "in_tensor")})
        self.add_flow(postprocessor, sinkop, {("out_tensor", "signal")})
#        idop = IdentityOp(self, name="Passthrough")
#        # Define the workflow
#        self.add_flow(source, preprocessor, {("output", "source_video")})
#        #self.add_flow(datagenerator, inference, {("tensor", "receivers")})
#        self.add_flow(preprocessor, idop)
#        self.add_flow(idop, inference, {("", "receivers")})
#        self.add_flow(inference, postprocessor, {("transmitter", "in_tensor")})
#        self.add_flow(postprocessor, sinkop, {("out_tensor", "signal")})


def main(config_file, data):
    app = BYOMApp(data=data)
    # if the --config command line argument was provided, it will override this config_file
    app.config(config_file)
    app.run()


if __name__ == "__main__":
    # Parse args
    parser = ArgumentParser(description="BYOM demo application.")
    parser.add_argument(
        "-d",
        "--data",
        default="none",
        help=("Set the data path"),
    )

    args = parser.parse_args()
    config_file = os.path.join(os.path.dirname(__file__), "byom.yaml")
    main(config_file=config_file, data=args.data)
