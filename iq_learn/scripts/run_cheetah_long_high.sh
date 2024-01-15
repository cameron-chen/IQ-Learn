cd ..
python train_iq.py env=cheetah_long_high agent=sac expert.demos=3 method.loss=value method.regularize=True agent.actor_lr=3e-05 seed=0