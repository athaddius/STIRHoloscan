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
from holoscan.conditions import CountCondition
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
import cv2

from STIRLoader import STIRLoader
import torch
import torchvision
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

# Loading a single dataset for demonstration, use a longer sequence
from pathlib import Path
datasequence = STIRLoader.STIRStereoClip(leftseqpath=Path("/workspace/data/13/left/seq20"))
dataset = STIRLoader.DataSequenceFull(datasequence)  # wraps in dataset
cur_num_points = 0
pointlocs_last = cp.zeros((1, 32, 2), np.float32)

class DataGenOp(Operator):
    """Generate dummy data for point tracking"""
    def __init__(self, fragment, *args, **kwargs):
        global cur_num_points
        self.count = 0
        self.pointlist_start = np.array(dataset.dataset.getstartcenters())
        cur_num_points = self.pointlist_start.shape[1]
        self.image1 = None
        self.dataloader = torch.utils.data.DataLoader(
                        dataset, batch_size=1, num_workers=1, pin_memory=True
                    )
        self.dataloaderiter = iter(self.dataloader)

        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out_dict")

    @staticmethod
    def resize(data):
        """ Takes a np image resizes it to half scale """
        im = data['ims'][0][0,...].to(device) # gets image, presuming 1-batch size
        im = torchvision.transforms.functional.resize(im, (512, 640))
        return im.unsqueeze(0)

    def compute(self, op_input, op_output, context):
        load_data = False
        out_dict = {}
        if load_data:
            try:
                if self.image1 is None:
                    self.image1 = DataGenOp.resize(next(self.dataloaderiter))
                self.image2 = DataGenOp.resize(next(self.dataloaderiter))
            except StopIteration:
                print("done")
                exit()
                #self.dataloaderiter = iter(self.dataloader)
                #self.image1 = DataGenOp.resize(next(self.dataloaderiter))
            out_dict["image1"] = cp.array(self.image1, np.float32)
            out_dict["image2"] = cp.array(self.image2, np.float32)
            self.image1 = self.image2
            if self.count == 0:
                out_dict["pointlist"] = cp.array(self.pointlist_start/2.0, np.float32) # dividing by 2 since we downscale ims
                #print(out_dict["pointlist"])
            else:
                out_dict["pointlist"] = pointlocs_last / 2.0
        else:
            out_dict["image1"] = cp.random.random((1,3,512, 640), np.float32)
            out_dict["image2"] = cp.random.random((1,3,512,640), np.float32)
            out_dict["pointlist"] = cp.random.random((1,32,2), np.float32)
        op_output.emit(out_dict, "out_dict")
        self.count+=1

class PrintSignalOp(Operator):
    """Print the received signal to the terminal."""

    def __init__(self, fragment, *args, **kwargs):
        """Initialize the application"""
        super().__init__(fragment, *args, **kwargs)
        self.iter_number = 0


    def setup(self, spec: OperatorSpec):
        spec.input("signal")

    def compute(self, op_input, op_output, context):
        global pointlocs_last
        signal = op_input.receive("signal")
        pointlocs_last = cp.array(signal['end_points']) * 2.0
        self.iter_number += 1
        if self.iter_number % 50 == 0:
            print(f"{self.iter_number} iters Done")
            #print(pointlocs_last[:,:cur_num_points,:])

class PointTrackerApp(Application):
    def __init__(self):
        """Initialize the application"""
        super().__init__()
        self.name = "Point Tracker App"
        model_path = './model/raft_pointtrack.engine'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found, make sure you have generated it with trtexec")

        self.model_path_map = {
            "raft_model": model_path
        }

    def compose(self):
        host_allocator = UnboundedAllocator(self, name="host_allocator")
        inference = InferenceOp(
            self,
            name="inference",
            allocator=host_allocator,
            transmit_on_cuda = True,
            infer_on_cpu = False,
            output_on_cuda=True,
            input_on_cuda=True,
            model_path_map=self.model_path_map,
            is_engine_path=True,
            backend='trt',
            **self.kwargs("inference"),
        )

        sinkop = PrintSignalOp(self, name="Sink")
        genop = DataGenOp(self, CountCondition(self, 2000), name="Generator") # run for 2000 iters
        # Define the workflow
        self.add_flow(genop, inference, {("", "receivers")})
        self.add_flow(inference, sinkop, {("transmitter", "signal")})


def main(config_file, data):
    app = PointTrackerApp()
    # if the --config command line argument was provided, it will override this config_file
    app.config(config_file)
    with Tracker(app, filename="/workspace/output/timestamps.log") as tracker:
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
