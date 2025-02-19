# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

MODEL_DIR=${MODEL_DIR-$PWD}
if [ ! -e "${MODEL_DIR}/models/recommendation/pytorch/dlrm/product/dlrm_s_pytorch.py"  ]; then
    echo "Could not find the script of dlrm_s_pytorch.py. Please set environment variable '\${MODEL_DIR}'."
    echo "From which the dlrm_s_pytorch.py exist at the: \${MODEL_DIR}/models/recommendation/pytorch/dlrm/product/dlrm_s_pytorch.py"
    exit 1
fi
MODEL_SCRIPT=${MODEL_DIR}/models/recommendation/pytorch/dlrm/product/dlrm_s_pytorch.py

echo "PRECISION: ${PRECISION}"
echo "DATASET_DIR: ${DATASET_DIR}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

if [ -z "${DATASET_DIR}" ]; then
  echo "The required environment variable DATASET_DIR has not been set"
  exit 1
fi

if [ -z "${NODE}" ]; then
  echo "The  environment variable NODE which is the number of node(machine) has not been set"
  exit 1
fi


if [ ! -d "${DATASET_DIR}" ]; then
  echo "The DATASET_DIR '${DATASET_DIR}' does not exist"
  exit 1
fi

if [[ "$NUM_BATCH" != "" ]]
then
    ARGS="$ARGS --num-batches=${NUM_BATCH}"
    echo "will early stop after ${NUM_BATCH} batches"
fi
 
# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}
LOG=${OUTPUT_DIR}/dlrm_distribute_training_log/${PRECISION}
rm -rf ${LOG}
mkdir -p ${LOG}

if [[ $PRECISION == "bf16" ]]; then
    ARGS="$ARGS --bf16"
    echo "running bf16 path"
elif [[ $PRECISION == "fp32" ]]; then
    echo "running fp32 path"
elif [[ $PRECISION == "bf32" ]]; then
    ARGS="$ARGS --bf32"
    echo "running bf32 path"
else
    echo "The specified PRECISION '${PRECISION}' is unsupported."
    echo "Supported PRECISIONs are: fp32, bf32, bf16"
    exit 1
fi

BATCHSIZE=${BATCHSIZE:-32768}
NUM_CCL_WORKER=${NUM_CCL_WORKER:-8}
HOSTFILE=${HOSTFILE:-hostfile1}

seed_num=1665468325 #1665462256 #$(date +%s)
oneccl_bindings_for_pytorch_path=$(python -c "import torch; import oneccl_bindings_for_pytorch; import os;  print(os.path.abspath(os.path.dirname(oneccl_bindings_for_pytorch.__file__)))")
source $oneccl_bindings_for_pytorch_path/env/setvars.sh
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
export CCL_LOG_LEVEL=info
# export CCL_MNIC=local
# export FI_PROVIDER=psm3
# export CCL_ATL_TRANSPORT=ofi

export CCL_ALLREDUCE=rabenseifner
# export PSM3_IDENTIFY=1
# export PSM3_ALLOW_ROUTERS=1
# export PSM3_RDMA=1 
# export PSM3_RV_MR_CACHE_SIZE=8192 
# export FI_PROVIDER_PATH=/usr/lib64/libfabric
# export CCL_MNIC_NAME=irdma-cvl01tf2,irdma-cvl02tf2,irdma-cvl11tf2,irdma-cvl12tf2
# export CCL_MNIC_COUNT=2

CORES=`lscpu | grep Core | awk '{print $4}'`
CORES_PER_INSTANCE=$CORES
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`

LOCAL_BATCH_SIZE=$((BATCHSIZE / NODE))
LOCAL_BATCH_SIZE=$((LOCAL_BATCH_SIZE / 2))

LOG_0="${LOG}/socket.log"
python -m intel_extension_for_pytorch.cpu.launch --enable_tcmalloc --logical_core_for_ccl --ccl_worker_count $NUM_CCL_WORKER --distributed --hostfile $HOSTFILE --nnodes $NODE \
$MODEL_SCRIPT \
  --raw-data-file=${DATASET_DIR}/day --processed-data-file=${DATASET_DIR}/terabyte_processed.npz \
  --data-set=terabyte \
  --memory-map --mlperf-bin-loader --mlperf-bin-shuffle --round-targets=True --learning-rate=18.0 \
  --arch-mlp-bot=13-512-256-128 --arch-mlp-top=1024-1024-512-256-1 \
  --arch-sparse-feature-size=128 --max-ind-range=40000000 \
  --numpy-rand-seed=${seed_num} --print-auc --mlperf-auc-threshold=0.8025 \
  --lr-num-warmup-steps=8000   --lr-decay-start-step=70000 --lr-num-decay-steps=30000\
  --local-batch-size=${LOCAL_BATCH_SIZE} --print-freq=100 --print-time --ipex-interaction \
  --test-mini-batch-size=65536 --ipex-merged-emb --should-test --test-freq 6400\
  $ARGS |tee $LOG_0
wait

throughput=$(grep 'Throughput:' ${LOG}/socket* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk '
BEGIN {
        sum = 0;
        i = 0;
      }
      {
        sum = sum + $1;
        i++;
      }
END   {
sum = sum / i;
        printf("%.3f", sum);
}')
echo ""dlrm";"training distributed throughput";${PRECISION};${BATCHSIZE};${throughput}" | tee -a ${OUTPUT_DIR}/summary.log
