import torch

class OptimisticGD(torch.optim.Optimizer):
    """
    Simple Optimistic Gradient Descent (OGD) optimizer.

    Update:
        g_t = grad at step t
        θ_{t+1} = θ_t - lr * (2 * g_t - g_{t-1})
    On the first step (no previous grad), falls back to vanilla GD.
    """

    def __init__(self, params, lr=1e-2):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        If closure is provided, it should re-compute the loss and gradients.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]
                prev_grad = state.get("prev_grad", None)

                # If no previous gradient, just do plain GD
                if prev_grad is None:
                    update = grad
                else:
                    # Optimistic update: 2*g_t - g_{t-1}
                    update = 2 * grad - prev_grad

                # θ <- θ - lr * update
                p.add_(update, alpha=-lr)

                # Store current gradient for next step
                state["prev_grad"] = grad.clone()

        return loss
