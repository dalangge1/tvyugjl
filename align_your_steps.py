import numpy as np
from diffusers import DPMSolverMultistepScheduler as DefaultDPMSolver
import torch


def loglinear_interp(t_steps, num_steps):
    """
    Performs log-linear interpolation of a given array of decreasing numbers.
    """
    xs = np.linspace(0, 1, len(t_steps))
    ys = np.log(t_steps[::-1])

    new_xs = np.linspace(0, 1, num_steps)
    new_ys = np.interp(new_xs, xs, ys)

    interped_ys = np.exp(new_ys)[::-1].copy()
    return interped_ys


# Add support for setting custom timesteps
class DPMSolverMultistepScheduler(DefaultDPMSolver):
    def set_timesteps(
            self, num_inference_steps=None, device=None,
            timesteps=None
    ):
        if timesteps is None:
            super().set_timesteps(num_inference_steps, device)
            return

        all_sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        self.sigmas = torch.from_numpy(all_sigmas[timesteps])
        self.timesteps = torch.tensor(timesteps[:-1]).to(device=device, dtype=torch.int64)  # Ignore the last 0

        self.num_inference_steps = len(timesteps)

        self.model_outputs = [
                                 None,
                             ] * self.config.solver_order
        self.lower_order_nums = 0

        # add an index counter for schedulers that allow duplicated timesteps
        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication