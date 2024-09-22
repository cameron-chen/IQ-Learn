#!/bin/bash

cd ..
# Command template with placeholder for the seed
cmd="python train_iq.py additional_loss=currentQ_expertAndPolicy cql_coef=2 num_random=5 env.learn_steps=1000000 cond_dim=10 method.kld_alpha=1 agent.actor_lr=2e-4 agent.init_temp=5e-3 wandb=True env=walker agent=sac expert.demos=30 method.enable_bc_actor_update=False method.bc_init=False method.bc_alpha=0.1 env.eval_interval=1e4 cond_type=debug env.demo=walker/Walker2d-v2_30_3074r+4329r+5898r.pkl env.cond=walker/240825_162656/Walker2d-v2_30_3074r+4329r+5898r_meanAsEmb_prob-encoder_dim10_kld_alpha1_betaB_step_10.pkl method.loss=v0 method.regularize=True exp_dir=/common/home/users/z/zichang.ge.2023/proj/IQ-Learn/iq_learn/encoder/experiments/walker/240825_162656/ encoder=prob-encoder_dim10_kld_alpha1_betaB_step_10.ckpt num_levels=3 save_last=True seed=SEED"

# Array of seeds
seeds=(1 2 3)

# Loop over each seed
for seed in "${seeds[@]}"
do
  # Replace the placeholder "SEED" with the actual seed
  cmd_with_seed=${cmd//SEED/$seed}

  # Run the command in the background
  echo "Running command with seed=$seed"
  $cmd_with_seed &

  # Sleep for a few seconds between each run
  sleep 300
done

# Wait for all background jobs to complete
wait

echo "All jobs finished."
