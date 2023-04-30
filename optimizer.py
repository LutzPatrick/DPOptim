import torch

class SGD_clipped(torch.optim.Optimizer):

    def __init__(self, params, lr, momentum):
        defaults = {'lr': lr, 'momentum': momentum}
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['grad_prev'] = torch.zeros(p.shape, device=p.device)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if momentum > 0:
                    grad = momentum * state['grad_prev'] + p.grad
                else:
                    grad = p.grad
                state['grad_prev'] = grad
                p.add_(grad, alpha=-lr)                

        return loss