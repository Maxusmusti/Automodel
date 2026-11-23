# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -xeuo pipefail # Exit immediately if a command exits with a non-zero status

export PYTHONPATH=${PYTHONPATH:-}:$(pwd)
export CUDA_VISIBLE_DEVICES="0,1"
# Deterministic kernels, so the resumed run's next update can be compared bitwise with the uninterrupted run.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export FLASH_ATTENTION_DETERMINISTIC=1

PYTEST_S_FLAG=""
if [ "${PYTEST_PROPAGATE_S:-}" = "1" ]; then
    PYTEST_S_FLAG="-s"
fi

# OSFT (fp32 master) on a 2-layer Mixtral: attention projections are decomposed, MoE experts stay dense.
TRANSFORMERS_OFFLINE=1 python -m torch.distributed.run \
--master-port=29511 --nproc_per_node=2 --nnodes=1 -m coverage run \
-m pytest $PYTEST_S_FLAG tests/functional_tests/checkpoint/test_osft.py::test_osft_checkpoint \
    --config examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml \
    --model.pretrained_model_name_or_path $TEST_DATA_DIR/hf_mixtral_2l/ \
    --model.torch_dtype float32 \
    --step_scheduler.max_steps 6 \
    --step_scheduler.global_batch_size 16 \
    --step_scheduler.local_batch_size 4 \
    --step_scheduler.ckpt_every_steps 6 \
    --step_scheduler.val_every_steps 100 \
    --dataset.tokenizer.pretrained_model_name_or_path $TEST_DATA_DIR/hf_mixtral_2l/ \
    --validation_dataset.tokenizer.pretrained_model_name_or_path $TEST_DATA_DIR/hf_mixtral_2l/ \
    --dataset.dataset_name $HF_CACHE/squad/ \
    --validation_dataset.dataset_name $HF_CACHE/squad/ \
    --dataset.limit_dataset_samples 512 \
    --checkpoint.enabled true \
    --checkpoint.checkpoint_dir checkpoints_osft/ \
    --checkpoint.save_consolidated true \
    --osft._target_ nemo_automodel.components._peft.osft.OSFTConfig \
    --osft.unfreeze_rank_ratio 0.25 \
    --osft.target_modules "[\"*.q_proj\", \"*.k_proj\", \"*.v_proj\", \"*.o_proj\"]" \
    --distributed.dp_size none \
    --distributed.tp_size 1 \
    --distributed.cp_size 1 \
    --distributed.sequence_parallel false
