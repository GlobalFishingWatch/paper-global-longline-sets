import numpy as np


class CosineAnnealingScheduler:
    """Scheduler for cosine annealing

    Learning rate ramps up to `max_lr` from zero over
    `warmup` epochs. The learning rate then decays
    to `min_lr` following a cosine trajectory over
    `n0` epochs. The learning rate then jumps back
    to `mak_lr` and this repeats except that the period
    doubles with reach repetition.
    """

    def __init__(
        self,
        warmup: int = 10,
        n0: int = 50,
        max_lr: float = 1e-2,
        min_lr: float = 0.0,
        length_scale: float = 1.0,
    ):
        assert n0 > 1
        self.n0 = n0
        self.n = n0
        self.i0 = warmup + 1
        self.warmup = warmup
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.length_scale = length_scale

    def __call__(self, epoch: int, lr: float) -> float:
        n = epoch + 1
        if n <= self.warmup:
            # Reset n0 in case we've restarted
            self.n = self.n0
            return n / (self.warmup + 1) * (self.max_lr - self.min_lr) + self.min_lr
        if n - self.i0 >= self.n:
            self.i0 = n
            self.n = int(round(self.n * self.length_scale))
        arg = np.pi * (n - self.i0) / (self.n - 1)
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(arg))

    def epochs_for_cycles(self, cycles: int) -> int:
        """How many epochs are needed to complete cycles annealings

        Args:
            cycles

        Returns
        -------
            number of epochs needed to complete `cycles` cycles
        """
        cnt = self.warmup
        n = self.n0
        for _ in range(cycles):
            cnt += n
            n = int(round(n * self.length_scale))
        return cnt
