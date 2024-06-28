# PointTracker Holoscan Bring Your Own Model


**For participants in the efficiency component of the STIR Challenge**

This example shows how to run inference on the NVIDIA Holoscan platform using a simple point tracking model. We provide a RAFT model that can be exported to onnx using the code in [RAFT_STIR](https://github.com/athaddius/RAFT_STIR).

For the challenge, models should be exported to onnx or tensorrt, and follow the same API as in the `python/pointtracker_holoscan.py`. In essence, to prepare your model to be benchmarked, set it up to ingress a pointlist of dimension (1, 32, 2) and two images of dimension (1, 3, 512, 640). We expect the entire model to be exported into an onnx or tensorrt file, as the scaffolding code will not be editable by challenge participants.

**Note: The pointcount will be fixed to 32 points for the efficiency component only.** For the accuracy component of the challenge, allow a flexible amount of points for inference. 
If you have a **fixed-size base model**, this can be done via masking unused inputs (of the 32) in your model for the accuracy evaluation [STIRMetrics](https://github.com/athaddius/STIRMetrics). If your model is **dynamically sized** (like our implemenation in RAFT_STIR), then complete STIRMetrics as-is, and export a fixed size onnx (or tensorrt) model for this efficiency component.

To test out the example RAFT model, follow the instructions under holoscan for installation instructions of the NVIDIA Holoscan platform.
  ```
  ./rundocker.sh
  cd <CODEDIR_DST> # change directory to where you set your code to be (this folder + your model file).
  python python/pointtracker_holoscan.py
  ```
If this all runs (outputs a string of "Generated 1,2,..."), then you are in a good place! Now make your onnx/tensorrt code do the same :)



# Holoscan

*Visit the [SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/examples/byom.html) for step-by-step documentation of the bring-your-own-model example.*

