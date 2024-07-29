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
"""

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
from holoscan.core import Tracker, DataFlowMetric
import cupy as cp
import numpy as np

from STIRLoader import STIRLoader
import torch
import numpy as np
from tqdm import tqdm

device = "cuda"

def todevice(cpudict):
    outdict = {}
    for k, v in cpudict.items():
        if k != "ims_ori":
            outdict[k] = [x.to(device) for x in v]
        else:
            outdict[k] = v
    return outdict

# Loading a single dataset for demonstration
datasets = STIRLoader.getclips(datadir="/workspace/data")
dataset = datasets[0]

class DataGenOp(Operator):
    """Generate dummy data for point tracking"""
    def __init__(self, fragment, *args, **kwargs):
        self.count = 1
        self.pointlist_start = np.array(dataset.dataset.getstartcenters())
        self.image1 = None
        dataloader = torch.utils.data.DataLoader(
                        dataset, batch_size=1, num_workers=0, pin_memory=True
                    )
        self.dataloaderiter = iter(dataloader)
        self.out_dict = dict()
        self.out_dict["pointlist"] = cp.zeros((1,32,2), np.float32)
        self.out_dict["image1"] = cp.zeros((1,3,512, 640), np.float32)
        self.out_dict["image2"] = cp.zeros((1,3,512,640), np.float32)

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out_dict")

    def compute(self, op_input, op_output, context):
        self.count+=1
        #if self.image1 is None:
            #self.image1 = todevice(next(self.dataloaderiter))
        #self.image2 = todevice(next(self.dataloaderiter))
        #print the types of structures for image1, image2 and pointlist_start
        #out_dict["pointlist"] = cp.array(self.pointlist_start, np.float32)
        #out_dict["image1"] = cp.array(self.image1['ims_ori'][0], np.float32)
        #out_dict["image2"] = cp.array(self.image2['ims_ori'][0], np.float32)
        #out_dict["pointlist"] = cp.random.random((1,32,2), np.float32)
        #out_dict["image1"] = cp.random.random((1,3,512, 640), np.float32)
        #out_dict["image2"] = cp.random.random((1,3,512,640), np.float32)
        #self.image1 = self.image2
        # print(f"Generated {self.count}")
        op_output.emit(self.out_dict, "out_dict")

class IdentityOp(Operator):
    """Print the received signal to the terminal."""

    def setup(self, spec: OperatorSpec):
        spec.input("input_tensor")
        spec.output("output_tensor")

    def compute(self, op_input, op_output, context):
        # op_input['input_tensor']
        # cupy_signal = signal['out_tensor']
        in_message = op_input.receive("input_tensor")
        out_msg = dict()
        for k, v in in_message.items():
            # breakpoint()
            cp_array = cp.asarray(v)
            out_msg[k] = cp_array
        # print("Received")
        signal = op_output.emit(out_msg, "output_tensor")

class PrintSignalOp(Operator):
    """Print the received signal to the terminal."""

    def setup(self, spec: OperatorSpec):
        spec.input("signal")

    def compute(self, op_input, op_output, context):
        signal = op_input.receive("signal")
        #numpy_signal = signal['out_tensor']

class PointTrackerApp(Application):
    def __init__(self):
        """Initialize the application"""

        super().__init__()

        self.name = "Point Tracker App"

        # check if model_path exists
        model_path = './model/raft_pointtrackSTIR.onnx'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")

        self.model_path_map = {
            "raft_model": os.path.join('./model/raft_pointtrackSTIR.onnx'),
        }

    def compose(self):
        host_allocator = UnboundedAllocator(self, name="host_allocator")

        inference = InferenceOp(
            self,
            name="inference",
            allocator=host_allocator,
            transmit_on_cuda = False,
            infer_on_cpu = False,
            output_on_cuda=False,
            input_on_cuda=False,
            model_path_map=self.model_path_map,
            **self.kwargs("inference"),
        )

        sinkop = PrintSignalOp(self, name="Sink")
        genop = DataGenOp(self, name="Generator")
        # Define the workflow
        self.add_flow(genop, inference, {("", "receivers")})
        self.add_flow(inference, sinkop, {("transmitter", "signal")})


def main(config_file, data):
    app = PointTrackerApp()
    # if the --config command line argument was provided, it will override this config_file
    app.config(config_file)
    with Tracker(app, filename="timestamps.log") as tracker:
        app.run()


if __name__ == "__main__":
    # Parse args
    parser = ArgumentParser(description="Point Tracker demo application.")
    parser.add_argument(
        "-d",
        "--data",
        default="none",
        help=("Set the data path"),
    )

    # check if "./STIRDataset" directory exists
    if not os.path.exists("/workspace/data"):
        raise FileNotFoundError(f"Data directory '/workspace/data' not found")

    args = parser.parse_args()
    config_file = os.path.join(os.path.dirname(__file__), "pointtracker.yaml")
    main(config_file=config_file, data=args.data)
