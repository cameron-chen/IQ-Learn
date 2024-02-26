cd ..
python train_iq.py env=cheetah_long agent=sac expert.demos=10 cond_dim=1 random_index=-1 method.loss=value method.regularize=True agent.actor_lr=3e-05 seed=0 agent.init_temp=1e-6 