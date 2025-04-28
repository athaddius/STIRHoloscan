# STIR Holoscan PointTracker Benchmarking

**For participants in the efficiency component of the 2024 STIR Challenge**

This example shows how to run inference on the NVIDIA Holoscan platform using a simple point tracking model. We provide a RAFT model that can be exported to onnx using the code in [RAFT_STIR](https://github.com/athaddius/RAFT_STIR).

For the challenge, models should be exported to onnx or tensorrt, and follow the same API as in the `python/pointtracker_holoscan.py`. In essence, to prepare your model to be benchmarked, set it up to ingress a pointlist of dimension (1, 32, 2) and two images of dimension (1, 3, 512, 640). We expect the entire model to be exported into an onnx or tensorrt file, and embedded as an inference operator within holoscan. Additional Pre/Post-Processing can be added. If doing so, make sure to put the processing in new and separate Operators from both the DataLoader and the Output Operator so we can time the full latency of your model (pre-process+inference+post-process).

**Note: The pointcount will be fixed to 32 points for the efficiency component only.** For the accuracy component of the challenge, allow a flexible amount of points for inference. 
If you have a **fixed-size base model**, this can be done via masking unused inputs (of the 32) in your model for the accuracy evaluation [STIRMetrics](https://github.com/athaddius/STIRMetrics). If your model is **dynamically sized** (like our implementation in RAFT_STIR), then complete STIRMetrics as-is, and export a fixed size onnx (or tensorrt) model for this efficiency component.


## Steps

### Read Challenge Rules, Sign Up, and Submission Instructions
Challenge details for submission are on our [2024 STIR Challenge Synapse Page](https://www.synapse.org/Synapse:syn54126082/wiki/626617).

### Download the dataset

To evaluate your method, use the STIR Dataset available [here](https://ieee-dataport.org/open-access/stir-surgical-tattoos-infrared). Download this to a folder, and use the location for `<datasetlocation>` when you run the docker container.


### Build a docker container for your submission

Confirm you are logged in to nvcr.io (ngc.nvidia.com) via docker:

```
$ docker login nvcr.io
Authenticating with existing credentials...
WARNING! Your password will be stored unencrypted in /home/ubuntu/.docker/config.json.
Configure a credential helper to remove this warning. See
https://docs.docker.com/engine/reference/commandline/login/#credentials-store

Login Succeeded
```

Clone the HoloHub repository for easy access to docker container build and run commands:

```
$ git clone git@github.com:nvidia-holoscan/holohub.git
```

Build a docker container where you can run your model within a Holoscan app:

```
$ ./holohub/dev_container build\
    --base_img nvcr.io/nvidia/clara-holoscan/holoscan:v2.1.0-dgpu 
    --docker_file ./Dockerfile --img stircontainer:2.1
```

To test out the example RAFT model, follow the instructions below:
First make an output directory to store the timestamps in any directory of your choosing. Then:
```
$ ./rundocker.sh <datasetlocation> <holohublocation> <output directory>
# inside docker container
$ ./genengine.sh  # generate a trt engine file from the onnx file
$ python python/pointtracker_holoscan.py # run for more than 20 frames
```
If this all runs (outputs point locations over time), then you are in a good place! Now make your onnx/tensorrt code do the same :)
Insert your model, and modify any of the code to make it run as efficient as possible while still creating correct predictions.



### Get Efficiency Metrics

The above command already generates the timestamp logs for the challenge (`timestamps.log`) to the docker `/workspace/output` directory (`<output_directory>` on your machine).

The following command (inside the docker container) will produce metrics like 99, 95
percentile, min, max, median and average latencies.

```
python analyze.py -l /workspace/output/timestamps.log 
```

We will use a combination of these numbers to evaluate the efficiency metric. More details are available on the [Synapse Page](https://www.synapse.org/Synapse:syn54126082/wiki/626617).

### Submission
For the evaluation, we expect participants to submit docker images that can be run in the same manner as `rundocker.sh` above. Use this codebase to begin, and set up your Dockerfile to import your model. Include the engine and anything that is needed for your model in this docker image. We will be running it with `./rundocker.sh`, verifying the model and then analyzing the `timestamps.log` file that it outputs. See the [Synapse Page](https://www.synapse.org/Synapse:syn54126082/wiki/626617) for additional details.

## Holoscan Documentation

*Visit the [SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/examples/byom.html) for step-by-step documentation of the bring-your-own-model example.*
