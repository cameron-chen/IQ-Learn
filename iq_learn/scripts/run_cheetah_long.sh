cd ..
python train_iq.py env=cheetah_long agent=sac expert.demos=6 method.loss=value method.regularize=True agent.actor_lr=3e-05 seed=0