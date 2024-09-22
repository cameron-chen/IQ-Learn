#!/bin/bash
cd ..
# Command template with placeholder for the seed
cmd="python train_iq.py env.learn_steps=1e6 cond_dim=10 method.kld_alpha=1 agent.actor_lr=3e-05 agent.init_temp=1e-12 wandb=True env=hopper agent=sac expert.demos=30 method.enable_bc_actor_update=False method.bc_init=False method.bc_alpha=0.5 env.eval_interval=1e4 cond_type=debug env.demo=hopper/Hopper-v2_30_409r+876r+3213r.pkl env.cond=hopper/no_id/Hopper-v2_30_409r+876r+3213r_meanAsEmb_prob-encoder_dim10_kld_alpha1_betaB_step_70.pkl method.loss=v0 method.regularize=True exp_dir=/common/home/users/z/zichang.ge.2023/proj/IQ-Learn/iq_learn/encoder/experiments/hopper/ encoder=prob-encoder_dim10_kld_alpha1_betaB_step_70.ckpt num_levels=3 additional_loss=none seed=SEED"

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
