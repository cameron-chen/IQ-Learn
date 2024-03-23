from typing import Dict, Union

import numpy as np
import torch


class BehaviorCloningLossCalculator:
    """Functor to compute the loss used in Behavior Cloning."""

    def __init__(self, ent_weight: float, l2_weight: float) -> None:
        self.ent_weight = ent_weight
        self.l2_weight = l2_weight

    def __call__(
        self,
        agent,
        obs,
        acts: Union[torch.Tensor, np.ndarray],
    ) -> Dict:
        """Calculate the supervised learning loss used to train the behavioral clone.

        Args:
            policy: The actor-critic policy whose loss is being computed.
            obs: The observations seen by the expert.
            acts: The actions taken by the expert.

        Returns:
            A BCTrainingMetrics object with the loss and all the components it
            consists of.
        """
        (_, log_prob, entropy) = agent.evaluate_actions(
            obs,  # type: ignore[arg-type]
            acts,
        )
        prob_true_act = torch.exp(log_prob).mean()
        log_prob = log_prob.mean()
        entropy = entropy.mean() if entropy is not None else None

        l2_norms = [torch.sum(torch.square(w)) for w in agent.actor.parameters()]
        l2_norm = sum(l2_norms) / 2  # divide by 2 to cancel with gradient of square
        # sum of list defaults to float(0) if len == 0.
        assert isinstance(l2_norm, torch.Tensor)

        ent_loss = -self.ent_weight * (
            entropy if entropy is not None else torch.zeros(1)
        )
        neglogp = -log_prob
        l2_loss = self.l2_weight * l2_norm
        loss = neglogp + ent_loss + l2_loss

        return {
            "bc_actor_loss/neglogp": neglogp,
            "bc_actor_loss/entropy": entropy,
            "bc_actor_loss/ent_loss": ent_loss,
            "bc_actor_loss/prob_true_act": prob_true_act,
            "bc_actor_loss/l2_norm": l2_norm,
            "bc_actor_loss/l2_loss": l2_loss,
            "loss/bc_actor": loss,
        }
