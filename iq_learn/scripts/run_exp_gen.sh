cd ..
python expert_generation.py env=cheetah_long agent=sac method.loss=value method.regularize=True agent.actor_lr=3e-05 seed=0 agent.init_temp=0.01