from typing import Any, Optional, Union

import eagerpy as ep
import librosa
import numpy as np
import torch
from foolbox.attacks.base import T, get_criterion, raise_if_kwargs, verify_input_bounds
from foolbox.attacks.gradient_descent_base import LinfBaseGradientDescent
from foolbox.criteria import Misclassification, TargetedMisclassification
from foolbox.models.base import Model
from tqdm import tqdm

class PGD_freq(LinfBaseGradientDescent):
    def __init__(
            self,
            *,
            rel_stepsize: float = 0.01 / 0.3,
            abs_stepsize: Optional[float] = None,
            steps: int = 40,
            random_start: bool = True,
    ):
        """
        Initialize the PGD_freq attack.

        Parameters:
        - rel_stepsize: The relative step size for each iteration.
        - abs_stepsize: The absolute step size for each iteration.
        - steps: The number of optimization steps.
        - random_start: Whether to start the optimization from a random point.
        """
        super().__init__(
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start,
        )

    def run(
            self,
            model: Model,
            inputs: T,
            criterion: Union[Misclassification, TargetedMisclassification, T],
            *,
            epsilon: float,
            lamb: float = None,
            **kwargs: Any,
    ) -> T:
        """
        Run the PGD_freq attack.

        Parameters:
        - model: The model to attack.
        - inputs: The input samples to attack.
        - criterion: The criterion to maximize or minimize.
        - epsilon: The maximum perturbation allowed.
        - lamb: Not used in this implementation.
        - kwargs: Additional keyword arguments.

        Returns:
        - T: The perturbed input samples.
        """
        raise_if_kwargs(kwargs)
        x0, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion, kwargs

        verify_input_bounds(x0, model)

        if isinstance(criterion_, Misclassification):
            gradient_step_sign = 1.0
            classes = criterion_.labels
        elif hasattr(criterion_, "target_classes"):
            gradient_step_sign = -1.0
            classes = criterion_.target_classes  # type: ignore
        else:
            raise ValueError("unsupported criterion")

        loss_fn = self.get_loss_fn(model, classes)
        if self.abs_stepsize is None:
            stepsize = self.rel_stepsize * epsilon
        else:
            stepsize = self.abs_stepsize

        if self.random_start:
            x = self.get_random_start(x0, epsilon)
            x = ep.clip(x, *model.bounds)
        else:
            x = x0
        x0 = x0.raw

        freqs = librosa.fft_frequencies(sr=22050, n_fft=2048)
        weights = librosa.A_weighting(freqs)
        normalized_weights = (weights - min(weights)) / (max(weights) - min(weights))
        scaling_factor = 0.2
        eps = epsilon + (1 - normalized_weights) * scaling_factor - np.mean((1 - normalized_weights) * scaling_factor)
        eps = torch.from_numpy(eps / 40)
        eps = eps.unsqueeze(1)
        normalized_weights = eps.to('cuda:0')

        for _ in tqdm(range(self.steps)):
            _, gradients = self.value_and_grad(loss_fn, x)
            gradients = self.normalize(gradients, x=x, bounds=model.bounds)
            x = x.raw + gradient_step_sign * gradients.raw * normalized_weights
            x = x0.raw + (x - x0.raw).clip(-normalized_weights * 40.0, normalized_weights * 40.0)
            x = ep.astensor(x)
            x = ep.clip(x, *model.bounds)
        return restore_type(x)
