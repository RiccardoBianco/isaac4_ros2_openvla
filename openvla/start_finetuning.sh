
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir "/cluster/scratch/${USER}/tensorflow_datasets" \
  --dataset_name sim_data_custom_v0 \
  --run_root_dir "/cluster/scratch/${USER}/finetuning_openvla/logs" \
  --adapter_tmp_dir "/cluster/scratch/${USER}/finetuning_openvla/adapter_tmp" \
  --lora_rank 32 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug False \
  --wandb_project openvla \
  --wandb_entity rbianco-eth-z-rich \
  --save_steps 2500