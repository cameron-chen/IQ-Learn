#!/bin/bash -i

#!/bin/bash

GREEN='\033[0;32m'
NC='\033[0m' # No Color
echo -e \
"$GREEN
██████╗ ██╗██████╗ ███████╗██╗     ██╗███╗   ██╗███████╗    ██╗   ██╗ ██████╗ 
██╔══██╗██║██╔══██╗██╔════╝██║     ██║████╗  ██║██╔════╝    ██║   ██║██╔═████╗
██████╔╝██║██████╔╝█████╗  ██║     ██║██╔██╗ ██║█████╗      ██║   ██║██║██╔██║
██╔═══╝ ██║██╔═══╝ ██╔══╝  ██║     ██║██║╚██╗██║██╔══╝      ╚██╗ ██╔╝████╔╝██║
██║     ██║██║     ███████╗███████╗██║██║ ╚████║███████╗     ╚████╔╝ ╚██████╔╝
╚═╝     ╚═╝╚═╝     ╚══════╝╚══════╝╚═╝╚═╝  ╚═══╝╚══════╝      ╚═══╝   ╚═════╝ $NC"


# Rules: 
#  - all `python` should be befind with `CUDA_VISIBLE_DEVICES=`
#  - all paths should be absolute
#  - make sure hyperparams are correct
#  - check space of the nfs-share
#  - notification of the critical stages

# 0. setup
## 0.1. environment
HOME_DIR=/home/zichang/proj/IQ-Learn/iq_learn
SCRIPT_DIR=$HOME_DIR/scripts/pipeline
SHORT_NAME=hopper # used as directory name
ENV_NAME=Hopper-v2 # used as file name

## 0.2. variables
LOG_DIR="$HOME_DIR/pipeline/logs"
MODEL_ID=30

## 0.3. resuming
#   - write a record when one step is finished
#   - if the script is interrupted, check the record and resume from the last step
TRACKING_FILE="${LOG_DIR}/tracking.txt"
BG_LOG_FILE="${LOG_DIR}/background.log"
if [ -f $TRACKING_FILE ]; then
    echo "Resuming from the last step"
    last_step=$(tail -n 1 $TRACKING_FILE)
    if [ $last_step -eq 11 ]; then
        echo "Step 1.1 is finished"
    elif [ $last_step -eq 12 ]; then
        echo "Step 1.2 is finished"
    elif [ $last_step -eq 13 ]; then
        echo "Step 1.3 is finished"
    elif [ $last_step -eq 14 ]; then
        echo "Step 1.3 is finished"
    elif [ $last_step -eq 21 ]; then
        echo "Step 2.1 is finished"
    elif [ $last_step -eq 22 ]; then
        echo "Step 2.2 is finished"
    elif [ $last_step -eq 23 ]; then
        echo "Step 2.3 is finished"
    elif [ $last_step -eq 31 ]; then
        echo "Step 3.1 is finished"
    elif [ $last_step -eq 32 ]; then
        echo "Step 3.2 is finished"
    elif [ $last_step -eq 33 ]; then
        echo "Step 3.3 is finished"
    fi
else
    echo "Starting from the beginning"
    echo -e "0" > $TRACKING_FILE
    last_step=0
fi


## 0.4. activate environment

cd $HOME_DIR
source activate base
conda activate hil

echo \
"
███████╗████████╗ █████╗  ██████╗ ███████╗     ██╗
██╔════╝╚══██╔══╝██╔══██╗██╔════╝ ██╔════╝    ███║
███████╗   ██║   ███████║██║  ███╗█████╗      ╚██║
╚════██║   ██║   ██╔══██║██║   ██║██╔══╝       ██║
███████║   ██║   ██║  ██║╚██████╔╝███████╗     ██║
╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝     ╚═╝"
# Stage 1. Deterministic encoder training
## 1.1. expert training
#   - generate 100 expert trajs for each skill level (3)
EXPERT_GEN_STEPS=100000
EXPERT_DEMOS=100
threshold=0
if [ $last_step -lt 11 ]; then
    echo "Stage 1: Deterministic encoder training"
    echo "Step 1.1: Expert training"
    python expert_generation.py env=$SHORT_NAME expert_gen_steps=$EXPERT_GEN_STEPS agent=sac eval.use_baselines=True expert.demos=$EXPERT_DEMOS eval.threshold=$threshold method.loss=v0 method.regularize=True agent.actor_lr=3e-5 seed=0 > $BG_LOG_FILE &
    wait
    
    if [ $? -eq 0 ]; then
        echo -e "11" >> $TRACKING_FILE
    else
        echo "Step 1.1 failed"
        exit 1
    fi
fi

## 1.2. expert generation
if [ $last_step -lt 12 ]; then
    echo "Step 1.2: Expert generation"
    # for each skill level, generate 100 expert trajs. used for stage 1 training
    EXPERT_DEMOS=100
    for i in 1 2 3; do
        res=$((i * EXPERT_GEN_STEPS))
        policy_name=$SHORT_NAME/$res.zip
        if [ ! -f "trained_policies/$policy_name" ]; then
            echo "Policy $policy_name does not exist"
            exit 1
        fi
        if [ ! -d "experts/$SHORT_NAME/" ]; then
            mkdir "experts/$SHORT_NAME/"
        fi
        sleep 10
        python expert_generation.py env=$SHORT_NAME agent=sac eval.use_baselines=False eval.policy=trained_policies/$policy_name expert.demos=$EXPERT_DEMOS eval.threshold=$threshold method.loss=v0 method.regularize=True agent.actor_lr=3e-5 seed=0 > $BG_LOG_FILE &
    done
    wait

    # for each skill level, generate 100 expert trajs. used for stage 1 evaluation
    EXPERT_DEMOS=10
    for i in 1 2 3; do
        res=$((i * EXPERT_GEN_STEPS))
        policy_name=$SHORT_NAME/$res.zip
        if [ ! -f "trained_policies/$policy_name" ]; then
            echo "Policy $policy_name does not exist"
            exit 1
        fi
        if [ ! -d "experts/$SHORT_NAME/" ]; then
            mkdir "experts/$SHORT_NAME/"
        fi
        sleep 10
        python expert_generation.py env=$SHORT_NAME agent=sac eval.use_baselines=False eval.policy=trained_policies/$policy_name expert.demos=$EXPERT_DEMOS eval.threshold=$threshold method.loss=v0 method.regularize=True agent.actor_lr=3e-5 seed=0 > $BG_LOG_FILE &
    done
    wait
    if [ $? -eq 0 ]; then
        echo -e "12" >> $TRACKING_FILE
    else
        echo "Step 1.2 failed"
        exit 1
    fi
fi

## 1.3. Deterministic encoder training
#   - load 300 trajs to encoder
if [ $last_step -lt 13 ]; then
    echo "Step 1.3: Deterministic encoder training"
    filtered_files=$(ls experts/$SHORT_NAME/ | grep "${ENV_NAME}_100_")
    sorted_files=$(echo "$filtered_files" | sort -t '_' -k3n)
    if [ $(echo "$sorted_files" | wc -l) -lt 3 ]; then
        echo "Error: There are fewer than 3 files in experts/$SHORT_NAME/."
        exit 1
    fi
    first_three_files=$(echo "$sorted_files" | awk 'BEGIN {sep=""} NR<=3 { printf "%s../experts/'$SHORT_NAME'/%s",sep,$0; sep="," }')
    echo "Training started with 3 expert files: $first_three_files"
    cd $HOME_DIR/encoder
    PYOPENGL_PLATFORM=egl python train_rl.py --expert_file=$first_three_files --belief-size=256 --name=$SHORT_NAME --coding_len_coeff=0.0001 --max_coding_len_coeff=0.0 --hil-seq-size=1000 --kl_coeff=0.0 --rec_coeff=1.0 --use_abs_pos_kl=1.0 --batch-size=16 --dataset-path=hopper --max-iters=30 --save_interval=500 --state-size=8 --use_min_length_boundary_mask --latent-n=10 > $BG_LOG_FILE &
    wait
    if [ $? -eq 0 ]; then
        echo -e "13" >> $TRACKING_FILE
    else
        echo "Step 1.3 failed"
        exit 1
    fi
fi

## 1.4. Evaluate the deterministic encoder
#   - save dir is in the model training path
if [ $last_step -lt 14 ]; then
    echo "Step 1.4: Evaluate the deterministic encoder"
    cd $HOME_DIR
    filtered_files=$(ls experts/$SHORT_NAME/ | grep "${ENV_NAME}_10_")
    sorted_files=$(echo "$filtered_files" | sort -t '_' -k3n)
    if [ $(echo "$sorted_files" | wc -l) -lt 3 ]; then
        echo "Error: There are fewer than 3 files in experts/$SHORT_NAME/."
        exit 1
    else
        first_three_files=$(echo "$sorted_files" | awk 'BEGIN {sep=""} NR<=3 { printf "%s../experts/'$SHORT_NAME'/%s",sep,$0; sep="," }')
        echo "Evaluation of deterministic encoder started with 3 expert files: $first_three_files. Plot can be found at encoder/plot/$SHORTNAME/test_${SHORT_NAME}_PCA.png later."
    fi
    cd $HOME_DIR/encoder
    python embed_hil.py test_$SHORT_NAME $first_three_files -b env=\"$SHORT_NAME\" -b checkpoint=\"experiments/$SHORT_NAME/model-$MODEL_ID.ckpt\" > $BG_LOG_FILE &
    wait
    if [ $? -eq 0 ]; then
        echo -e "14" >> $TRACKING_FILE
    else
        echo "Step 1.4 failed"
        exit 1
    fi
fi

echo \
"
███████╗████████╗ █████╗  ██████╗ ███████╗    ██████╗ 
██╔════╝╚══██╔══╝██╔══██╗██╔════╝ ██╔════╝    ╚════██╗
███████╗   ██║   ███████║██║  ███╗█████╗       █████╔╝
╚════██║   ██║   ██╔══██║██║   ██║██╔══╝      ██╔═══╝ 
███████║   ██║   ██║  ██║╚██████╔╝███████╗    ███████╗
╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝    ╚══════╝"
# Stage 2. Probabilistic encoder training
## 2.1. condition generation
#   - generate 20 expert trajs by combining 2 skill levels
#   - embed the expert trajs to get the condition
if [ $last_step -lt 21 ]; then
    echo "Step 2.1. condition generation" 
    cd $HOME_DIR
    filtered_files=$(ls experts/$SHORT_NAME/ | grep "${ENV_NAME}_10_")
    sorted_files=$(echo "$filtered_files" | sort -t '_' -k3n)
    file_count=$(echo "$sorted_files" | wc -l)
    if [ $file_count -lt 2 ]; then
        echo "Error: There are fewer than 2 files in experts/$SHORT_NAME/."
        exit 1
    else
        last_two_files=$(echo "$sorted_files" | tail -n 2 | awk 'BEGIN {sep=""} { printf "%s../experts/'$SHORT_NAME'/%s",sep,$0; sep="," }')
        # Extracting last words from the last two files
        last_words=$(echo "$sorted_files" | tail -n 2 | awk -F'_' '{gsub(/\..+$/, "", $NF); print $NF}')

        # Combining last words with '+'
        combined_last_word=$(echo "$last_words" | paste -sd '+')

        echo "Combining 2 skill levels: $last_two_files"
        echo "Combined last word: $combined_last_word"
    fi
    traj_name=${ENV_NAME}_20_${combined_last_word}

    cd $HOME_DIR/utils
    python combine_data.py $last_two_files 10 ../experts/$SHORT_NAME/${traj_name}.pkl > $BG_LOG_FILE &
    wait
    echo "Combined 2 skill levels to experts/$SHORT_NAME/$traj_name.pkl"

    
    cd $HOME_DIR/encoder
    python embed_hil.py $traj_name ../experts/$SHORT_NAME/${traj_name}.pkl -b env=\"$SHORT_NAME\" -b checkpoint=\"experiments/$SHORT_NAME/model-$MODEL_ID.ckpt\" --embed_mode=det &
    wait

    if [ $? -eq 0 ]; then
        echo -e "21" >> $TRACKING_FILE
        echo "Embedded the combined traj to condition cond/${traj_name}.pkl"
    else
        echo "Step 2.1 failed"
        exit 1
    fi
fi

## 2.2. Probabilistic encoder training
#   - load 20 trajs to encoder
if [ $last_step -lt 22 ]; then
    echo "Step 2.2. Probabilistic encoder training"
    cd $HOME_DIR
    demo=$(ls experts/$SHORT_NAME/ | grep "${ENV_NAME}_20_" | head -n 1)
    cond=$(ls cond/$SHORT_NAME/ | grep "${ENV_NAME}_20_" | head -n 1)
    echo "Training started with 20 expert trajs: $demo and initial condition: $cond"
    python train_iq.py bc_steps=60 cond_dim=10 method.kld_alpha=1 agent.actor_lr=3e-05 agent.init_temp=1e-12 seed=0 wandb=True env=$SHORT_NAME agent=sac expert.demos=20 env.learn_steps=2e5 method.enable_bc_actor_update=False method.bc_init=True method.bc_alpha=0.5 env.eval_interval=1e4 cond_type=debug env.demo=$SHORT_NAME/$demo env.cond=$SHORT_NAME/$cond method.loss=value method.regularize=True exp_dir=$HOME_DIR/encoder/experiments/$SHORT_NAME/ encoder=model-$MODEL_ID.ckpt > $BG_LOG_FILE &
    wait
    if [ $? -eq 0 ]; then
        echo -e "22" >> $TRACKING_FILE
    else
        echo "Step 2.2 failed"
        exit 1
    fi
fi

## 2.3. Probabilistic encoder evaluation
if [ $last_step -lt 23 ]; then
    echo "Step 2.3. Probabilistic encoder evaluation"
    cd $HOME_DIR/encoder
    prob_encoder=$(find "experiments/$SHORT_NAME/" -type f -name "*prob-encoder*" | sort -t_ -k7n | tail -n 1)
    if [ -z "$prob_encoder" ]; then
        echo "Error: No file containing 'prob-encoder' found in directory encoder/experiments/$SHORT_NAME/."
        exit 1
    else
        echo "Probabilistic encoder found: $prob_encoder"
    fi

    cd $HOME_DIR
    filtered_files=$(ls experts/$SHORT_NAME/ | grep "${ENV_NAME}_10_")
    sorted_files=$(echo "$filtered_files" | sort -t '_' -k3n)
    if [ $(echo "$sorted_files" | wc -l) -lt 3 ]; then
        echo "Error: There are fewer than 3 files in experts/$SHORT_NAME/."
        exit 1
    else
        first_three_files=$(echo "$sorted_files" | awk 'BEGIN {sep=""} NR<=3 { printf "%s../experts/'$SHORT_NAME'/%s",sep,$0; sep="," }')
        echo "Evaluation of mean as embeddings started with 3 expert files: $first_three_files"
    fi

    cd $HOME_DIR/encoder
    prob_encoder_purename=$(basename "$prob_encoder" .ckpt)
    python embed_hil.py test_meanAsEmb_$prob_encoder_purename $first_three_files -b env=\"$SHORT_NAME\" -b checkpoint=\"$prob_encoder\" --embed_mode=mean &
    wait
    if [ $? -eq 0 ]; then
        echo -e "23" >> $TRACKING_FILE
        echo "Evaluated using mean as embedding. Plot can be found at encoder/plot/$SHORTNAME/test_meanAsEmb_${prob_encoder_purename}_PCA.png"
    else
        echo "Step 2.3 failed"
        exit 1
    fi
fi

echo \
"
███████╗████████╗ █████╗  ██████╗ ███████╗    ██████╗ 
██╔════╝╚══██╔══╝██╔══██╗██╔════╝ ██╔════╝    ╚════██╗
███████╗   ██║   ███████║██║  ███╗█████╗       █████╔╝
╚════██║   ██║   ██╔══██║██║   ██║██╔══╝       ╚═══██╗
███████║   ██║   ██║  ██║╚██████╔╝███████╗    ██████╔╝
╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝    ╚═════╝ "
# Stage 3. Decoder training
## 3.1. Condition generation
if [ $last_step -lt 31 ]; then
    echo "Step 3.1. Condition generation"
    cd $HOME_DIR/encoder
    demo=$(find "../experts/$SHORT_NAME/" -type f -name "${ENV_NAME}_20_*" | head -n 1)
    demo_purename=$(basename "$demo" .pkl)
    prob_encoder=$(find "experiments/$SHORT_NAME/" -type f -name "*prob-encoder*" | sort -t_ -k7n | tail -n 1)
    if [ -z "$demo" ]; then
        echo "Error: No file containing '20' found in directory experts/$SHORT_NAME/."
        exit 1
    else
        echo "Demo found: $demo"
    fi
    if [ -z "$prob_encoder" ]; then
        echo "Error: No file containing 'prob-encoder' found in directory encoder/experiments/$SHORT_NAME/."
        exit 1
    else
        prob_encoder_purename=$(basename "$prob_encoder" .ckpt)
        echo "Probabilistic encoder found: $prob_encoder"
    fi
    cond_name=${demo_purename}_meanAsEmb_${prob_encoder_purename}
    cd $HOME_DIR/encoder
    python embed_hil.py $cond_name $demo -b env=\"$SHORT_NAME\" -b checkpoint=\"$prob_encoder\" --embed_mode=mean &
    wait
    if [ $? -eq 0 ]; then
        echo -e "31" >> $TRACKING_FILE
        echo "Condition generated using mean as embedding. Plot can be found at encoder/plot/$SHORTNAME/${cond_name}_PCA.png"
    else
        echo "Step 3.1 failed"
        exit 1
    fi
fi

## 3.2. Decoder training
if [ $last_step -lt 32 ]; then
    echo "Step 3.2. Decoder training"
    cd $HOME_DIR
    demo=$(ls experts/$SHORT_NAME/ | grep "${ENV_NAME}_20_" | head -n 1)
    demo_name=$(basename "$demo" .pkl)
    prob_encoder=$(find "encoder/experiments/$SHORT_NAME/" -type f -name "*prob-encoder*" | sort -t_ -k7n | tail -n 1)
    prob_encoder=$(basename "$prob_encoder")
    prob_encoder_name=$(basename "$prob_encoder" .ckpt)
    cond=$(ls cond/$SHORT_NAME/ | grep "${demo_name}_meanAsEmb_${prob_encoder_name}" | head -n 1)
    if [ -z "$cond" ]; then
        echo "Error: No condition containing '${demo_name}_meanAsEmb_${prob_encoder_name}' found in directory cond/$SHORT_NAME/."
        exit 1
    else
        echo -e "Cond found: $cond.\nDecoder training will start soon. You can find the training log at $BG_LOG_FILE."
    fi

    python train_iq.py bc_steps=30 cond_dim=10 method.kld_alpha=1 agent.actor_lr=3e-05 agent.init_temp=1e-12 seed=0 wandb=True env=$SHORT_NAME agent=sac expert.demos=20 env.learn_steps=1e5 method.enable_bc_actor_update=False method.bc_init=False method.bc_alpha=0.5 env.eval_interval=1e4 cond_type=debug env.demo=$SHORT_NAME/$demo env.cond=$SHORT_NAME/$cond method.loss=value method.regularize=True exp_dir=$HOME_DIR/encoder/experiments/$SHORT_NAME/ encoder=$prob_encoder &> "$BG_LOG_FILE" &
    python_pid=$!

    # Echo the PID of the Python process
    echo "Decoder training started with PID: $python_pid"

    # Wait for the Python process to finish
    wait

    # Check the exit status of the Python process
    if [ $? -eq 0 ]; then
        echo -e "32" >> $TRACKING_FILE
    else
        echo "Step 3.2 failed"
        exit 1
    fi
fi

## 3.3 Decoder evaluation
if [ $last_step -lt 33 ]; then
    echo "TODO: Evaluate the decoder" # TODO: Evaluate the decoder
    exit 0
    if [ $? -eq 0 ]; then
        echo -e "33" >> $TRACKING_FILE
    else
        echo "Step 3.3 failed"
        exit 1
    fi
fi