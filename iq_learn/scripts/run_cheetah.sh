cd ..
python train_iq.py env=cheetah agent=sac expert.demos=3 cond_dim=10 random_index=1 method.loss=value method.regularize=True agent.actor_lr=3e-05 seed=0 agent.init_temp=0.01
