<!--- 0. Title -->
# RFCN inference

<!-- 10. Description -->

This document has instructions for running RFCN inference using
Intel-optimized TensorFlow.


<!--- 30. Datasets -->
## Dataset

The [COCO validation dataset](http://cocodataset.org) is used in these
RFCN quickstart scripts. The inference quickstart scripts use raw images,
and the accuracy quickstart scripts require the dataset to be converted
into the TF records format.
See the [COCO dataset](/datasets/coco/README.md) for instructions on
downloading and preprocessing the COCO validation dataset.


<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`inference.sh`](/quickstart/object_detection/tensorflow/rfcn/inference/cpu/inference.sh) | Runs inference on a directory of raw images for 500 steps and outputs performance metrics. |
| [`accuracy.sh`](/quickstart/object_detection/tensorflow/rfcn/inference/cpu/accuracy.sh) | Processes the TF records to run inference and check accuracy on the results. |

<!--- 50. AI Tools -->
## Run the model

Setup your environment using the instructions below, depending on if you are
using [AI Tools](/docs/general/tensorflow/AITools.md):

<table>
  <tr>
    <th>Setup using AI Tools on Linux</th>
    <th>Setup without AI Tools on Linux</th>
    <th>Setup without AI Tools on Windows</th>
  </tr>
  <tr>
    <td>
      <p>To run using AI Tools on Linux you will need:</p>
      <ul>
        <li>git
        <li>numactl
        <li>wget
        <li>build-essential
        <li>Cython
        <li>contextlib2
        <li>jupyter
        <li>lxml
        <li>matplotlib
        <li>pillow>=9.3.0
        <li>protobuf-compiler
        <li>pycocotools
        <li>Activate the `tensorflow` conda environment
        <pre>conda activate tensorflow</pre>
      </ul>
    </td>
    <td>
      <p>To run without AI Tools on Linux you will need:</p>
      <ul>
        <li>Python 3
        <li>git
        <li>numactl
        <li>wget
        <li><a href="https://pypi.org/project/intel-tensorflow/">intel-tensorflow>=2.5.0</a>
        <li>build-essential
        <li>Cython
        <li>contextlib2
        <li>jupyter
        <li>lxml
        <li>matplotlib
        <li>pillow>=9.3.0
        <li>protobuf-compiler
        <li>pycocotools
        <li>A clone of the AI Reference Models repo<br />
        <pre>git clone https://github.com/IntelAI/models.git</pre>
      </ul>
    </td>
    <td>
      <p>To run without AI Tools on Windows you will need:</p>
      <ul>
        <li><a href="/docs/general/Windows.md">Intel AI Reference Models on Windows Systems prerequisites</a>
        <li>build-essential
        <li>Cython
        <li>contextlib2
        <li>jupyter
        <li>lxml
        <li>matplotlib
        <li>pillow>=9.3.0
        <li>protobuf-compiler
        <li>pycocotools
        <li>A clone of the AI Reference Models repo<br />
        <pre>git clone https://github.com/IntelAI/models.git</pre>
      </ul>
    </td>
  </tr>
</table>

For more information on the required dependencies, see the documentation on [prerequisites](https://github.com/tensorflow/models/blob/6c21084503b27a9ab118e1db25f79957d5ef540b/research/object_detection/g3doc/installation.md#installation)
in the TensorFlow models repo.

Download and extract the pretrained model and set the `PRETRAINED_MODEL`
environment variable to point to the frozen graph file.
If you run on Windows, please use a browser to download the pretrained model using the link below.
For Linux, run:
```
# FP32 Pretrained Model
https://storage.googleapis.com/intel-optimized-tensorflow/models/2_10_0/rfcn_frozen_inference_graph.pb
export PRETRAINED_MODEL=$(pwd)/rfcn_frozen_inference_graph.pb

# Int8 Pretrained Model
https://storage.googleapis.com/intel-optimized-tensorflow/models/2_10_0/rfcn_final_fused_pad_and_conv.pb
export PRETRAINED_MODEL=$(pwd)/rfcn_final_fused_pad_and_conv.pb
```

RFCN uses the object detection code from the [TensorFlow Model Garden](https://github.com/tensorflow/models).
Clone this repo with the SHA specified below and apply the patch from the AI Reference Models directory.
Set the `TF_MODELS_DIR` environment variable to the path of your clone of the TF Model Garden.
```
# Clone the TensorFlow Model Garden
git clone https://github.com/tensorflow/models.git tensorflow-models
cd tensorflow-models
git checkout 6c21084503b27a9ab118e1db25f79957d5ef540b

# Apply the TF2 patch from the AI Reference Models repo directory
git apply --ignore-space-change --ignore-whitespace <AI Reference Models directory>/models/object_detection/tensorflow/rfcn/inference/tf-2.0.patch

# Set the TF_MODELS_DIR env var
export TF_MODELS_DIR=$(pwd)
```

### Run on Linux
Download and install [Google Protobuf version 3.3.0](https://github.com/google/protobuf/releases/download/v3.3.0/protoc-3.3.0-linux-x86_64.zip), and
run [protobuf compilation](https://github.com/tensorflow/models/blob/6c21084503b27a9ab118e1db25f79957d5ef540b/research/object_detection/g3doc/installation.md#protobuf-compilation).
```
cd TF_MODELS_DIR/research
protoc object_detection/protos/*.proto --python_out=.
cd ../..
```

Once your environment is setup, navigate back to your AI Reference Models directory. Ensure that
you have set environment variables pointing to the TensorFlow Model Garden repo, the dataset,
and output directories, and then run a quickstart script.

To run inference with performance metrics:
```
# cd to your AI Reference Models directory
cd models

export DATASET_DIR=<path to the coco val2017 raw image directory (ex: /home/user/coco_dataset/val2017)>
export PRECISION=<set the precision to "int8" or "fp32">
export OUTPUT_DIR=<path to the directory where log files will be written>
export TF_MODELS_DIR=<directory where TensorFlow Model Garden is cloned>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

./quickstart/object_detection/tensorflow/rfcn/inference/cpu/inference.sh
```

To get accuracy metrics:
```
# cd to your AI Reference Models directory
cd models

export DATASET_DIR=<path to TF record file (ex: /home/user/coco_output/coco_val.record)>
export PRECISION=<set the precision to "int8" or "fp32">
export OUTPUT_DIR=<path to the directory where log files will be written>
export TF_MODELS_DIR=<directory where TensorFlow Model Garden is cloned>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

./quickstart/object_detection/tensorflow/rfcn/inference/cpu/accuracy.sh
```

### Run on Windows
* Download and install [Google Protobuf version 3.4.0]((https://github.com/protocolbuffers/protobuf/releases/tag/v3.4.0)) for Windows in addition to the above listed dependencies.
Download and extract [protoc-3.4.0-win32.zip](https://github.com/protocolbuffers/protobuf/releases/download/v3.4.0/protoc-3.4.0-win32.zip)
Navigate to the `research` directory in `TF_MODELS_DIR` and install Google Protobuf:
```
set TF_MODELS_DIR=<directory where TensorFlow Model Garden is cloned>
cd %TF_MODELS_DIR%\research
“C:\<user>\protoc-3.4.0-win32\bin\protoc.exe” object_detection/protos/*.proto --python_out=.
```

After installing the prerequisites and cloning the TensorFlow models repo, and downloading the pretrained model,
set the environment variables for the paths to your `PRETRAINED_MODEL`, an `OUTPUT_DIR` where log files will be written,
TF_MODELS_DIR, and `DATASET_DIR` for COCO raw dataset directory or tf_records file based on whether you run inference or accuracy scripts.
Navigate to your AI Reference Models directory and then run a [quickstart script](#quick-start-scripts).
```
# cd to your AI Reference Models directory
cd models

set PRETRAINED_MODEL=<path to the frozen graph downloaded above>
set DATASET_DIR=<path to COCO raw dataset directory or tf_records file based on whether you run inference or accuracy scripts>
set PRECISION=<set the precision to "int8" or "fp32">
set OUTPUT_DIR=<directory where log files will be written>
set TF_MODELS_DIR=<directory where TensorFlow Model Garden is cloned>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
set BATCH_SIZE=<customized batch size value>

bash quickstart\object_detection\tensorflow\rfcn\inference\cpu\<script name>.sh
```
> Note: You may use `cygpath` to convert the Windows paths to Unix paths before setting the environment variables. 
As an example, if the dataset location on Windows is `D:\user\coco_dataset\val2017`, convert the Windows path to Unix as shown:
> ```
> cygpath D:\user\coco_dataset\val2017
> /d/user/coco_dataset/val2017
>```
>Then, set the `DATASET_DIR` environment variable `set DATASET_DIR=/d/user/coco_dataset/val2017`.

<!--- 90. Resource Links-->
## Additional Resources

* To run more advanced use cases, see the instructions for the available precisions [FP32](fp32/Advanced.md) [Int8](int8/Advanced.md) [<bfloat16 precision>](<bfloat16 advanced readme link>) for calling the `launch_benchmark.py` script directly.
* To run the model using docker, please see the [Intel® Developer Catalog](https://www.intel.com/content/www/us/en/developer/tools/software-catalog/containers.html)
  workload container:<br />
  [https://www.intel.com/content/www/us/en/developer/articles/containers/rfcn-fp32-inference-tensorflow-container.html](https://www.intel.com/content/www/us/en/developer/articles/containers/rfcn-fp32-inference-tensorflow-container.html).
