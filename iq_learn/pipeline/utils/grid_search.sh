#!/bin/bash

# Define the ranges for hyperparameters
SHORT_NAME=ant
ACTOR_LR_VALUES=(5e-5 2e-4 5e-4)
CRITIC_LR_VALUES=(1e-5)
# CRITIC_LR_VALUES=(1e-5 5e-5 2e-4 5e-4 1e-3)

# # Log file to record the results
HOME_DIR=/home/zichang/proj/IQ-Learn/iq_learn
cd $HOME_DIR

# LOG_DIR="$HOME_DIR/pipeline/logs/$SHORT_NAME/grid_search"
# mkdir -p "$LOG_DIR"
# LOGFILE="$LOG_DIR/grid_search_results.txt"
# echo "Grid Search Results" > $LOGFILE
# echo "===================" >> $LOGFILE


# source activate base
# conda activate hil
# Loop over each combination of hyperparameters
for actor_lr in "${ACTOR_LR_VALUES[@]}"; do
    for critic_lr in "${CRITIC_LR_VALUES[@]}"; do
        
        # Define the command
        CMD="python train_iq.py additional_loss=combined_expertAndPolicy cql_coef=1 env.learn_steps=15e5 cond_dim=10 method.kld_alpha=1 agent.actor_lr=$actor_lr agent.critic_lr=$critic_lr agent.init_temp=1e-2 seed=0 wandb=True env=ant agent=sac expert.demos=20 method.enable_bc_actor_update=False method.bc_init=False method.bc_alpha=0.5 env.eval_interval=1e4 cond_type=debug env.demo=ant/Ant-v2_20_2990r+5688r.pkl env.cond=ant/240717_175438/Ant-v2_20_2990r+5688r_meanAsEmb_prob-encoder_dim10_kld_alpha0.5_betaB_step_190.pkl method.loss=v0 method.regularize=True exp_dir=/home/zichang/proj/IQ-Learn/iq_learn/encoder/experiments/ant/240717_175438/ encoder=prob-encoder_dim10_kld_alpha0.5_betaB_step_190.ckpt save_last=True save_interval=500"

        # Print the command being run
        echo "Running: $CMD"
        
        # Run the command in the background
        $CMD &

        # sleep 5 mins
        sleep 300

    done
done

# Wait for all background processes to complete
wait

# echo "Grid search completed. Results saved in $LOGFILE."
echo "Completed

