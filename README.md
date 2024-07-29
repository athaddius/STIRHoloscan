# PointTracker Holoscan Bring Your Own Model


**For participants in the efficiency component of the STIR Challenge**

This example shows how to run inference on the NVIDIA Holoscan platform using a simple point tracking model. We provide a RAFT model that can be exported to onnx using the code in [RAFT_STIR](https://github.com/athaddius/RAFT_STIR).

For the challenge, models should be exported to onnx or tensorrt, and follow the same API as in the `python/pointtracker_holoscan.py`. In essence, to prepare your model to be benchmarked, set it up to ingress a pointlist of dimension (1, 32, 2) and two images of dimension (1, 3, 512, 640). We expect the entire model to be exported into an onnx or tensorrt file, as the scaffolding code will not be editable by challenge participants.

**Note: The pointcount will be fixed to 32 points for the efficiency component only.** For the accuracy component of the challenge, allow a flexible amount of points for inference. 
If you have a **fixed-size base model**, this can be done via masking unused inputs (of the 32) in your model for the accuracy evaluation [STIRMetrics](https://github.com/athaddius/STIRMetrics). If your model is **dynamically sized** (like our implemenation in RAFT_STIR), then complete STIRMetrics as-is, and export a fixed size onnx (or tensorrt) model for this efficiency component.

## Steps

### Download the dataset

Please make sure the dataset is available in the current directory in `STIRDataset`:
https://ieee-dataport.org/open-access/stir-surgical-tattoos-infrared


### Build docker container for your solution

Please make sure you are logged in to nvcr.io (ngc.nvidia.com) via docker:

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
```
$  ./rundocker.sh <datasetlocation> <holohublocation>
# inside docker container
$ python python/pointtracker_holoscan.py # run for more than 20 frames
```
If this all runs (outputs a number of "Received"), then you are in a good place! Now make your onnx/tensorrt code do the same :)


### Get Efficiency Metrics

The above command will already generate the timestamp logs for the challnege (`timestamps.log`). 
The following command (inside the docker container) will produce metrics like 99.9, 99, 95
percentile latencies and average latency.
```
python /workspace/holohub/benchmarks/holoscan_flow_benchmarking/analyze.py -m -p 99.9 99 95 -a -g timestamps.log
```

It will output the maximum, 99.9 percentile, 99 percentile, 95 percentile and average end-to-end
latency. We will use a combination of these numbers to evaluate on the efficiency metric.

## Holoscan

*Visit the [SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/examples/byom.html) for step-by-step documentation of the bring-your-own-model example.*

