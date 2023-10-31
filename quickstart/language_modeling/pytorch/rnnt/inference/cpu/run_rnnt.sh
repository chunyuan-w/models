export LD_PRELOAD="/home/sdp/chunyuan/jemalloc/lib/libjemalloc.so:/home/sdp/anaconda3/envs/chunyuan-pt2/lib/libiomp5.so"
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

export MODEL_DIR=/localdisk2/chunyuan/chunyuan-pt2/intel_model_zoo
export DATASET_DIR=/localdisk/chunyuan/
export CHECKPOINT_DIR=/localdisk/chunyuan/ckpt/
export OUTPUT_DIR=/localdisk2/chunyuan/chunyuan-pt2/intel_model_zoo/models

cd ${MODEL_DIR}

cd ${MODEL_DIR}/quickstart/language_modeling/pytorch/rnnt/inference/cpu

# ACC=fp32
ACC=bf16
# ACC=bf32

if [ "$1" == "acc" ]; then
    bash accuracy.sh $ACC $2 $3
elif [ "$1" == "thp" ]; then
    bash inference_throughput.sh $ACC $2 $3
elif [ "$1" == "rt" ]; then
    bash inference_realtime.sh $ACC $2 $3
fi

# For torch.compile
#  ./run_rnnt.sh rt torch_compile
#  ./run_rnnt.sh thp torch_compile
#  ./run_rnnt.sh thp torch_compile profiling


# For IPEX:
#  ./run_rnnt.sh thp
