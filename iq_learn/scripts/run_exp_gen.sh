cd ..
python expert_generation.py env=hopper agent=sac eval.use_baselines=True expert.demos=100 eval.threshold=50 method.loss=v0 method.regularize=True agent.actor_lr=3e-5 seed=0