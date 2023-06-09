import torch    

class ONC_Descent(torch.optim.Optimizer):

    def __init__(self, params, lr, D, batch_size, grad_checkpoint=None, momentum=None):
        defaults = {'lr': lr, 'D': D, 'momentum': momentum}
        super().__init__(params, defaults)

        if grad_checkpoint is None:
            # create internal state
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]
                    state['grad_prev'] = None
                    state['x_prev'] = p
                    state['batch_size'] = batch_size
                    state['cpt'] = 0
        else: 
            # couple internal states of this and the checkpoint optimizer
            for group1, group2 in zip(self.param_groups, grad_checkpoint.param_groups):
                for p1, p2 in zip(group1['params'], group2['params']):
                    self.state[p1] = grad_checkpoint.state[p2]

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
            D = group['D']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                #if momentum > 0:
                #    grad = momentum * state['grad_prev'] + p.grad
                #else:
                #    grad = p.grad
                if state['cpt'] % state['batch_size'] == 0:
                    grad = p.grad
                else:
                    grad = state['grad_prev'] + p.grad
                
                
                if state['cpt'] % state['batch_size'] == state['batch_size'] - 1:
                    p.data = state['x_prev'].data
                else:
                    delta = -lr*torch.clip(grad, -D, D)
                    s = torch.rand(1, device=p.get_device())
                    p.data = state['x_prev'].data + delta * s
                    state['grad_prev'] = grad
                    state['x_prev'] += delta 

                state['cpt'] += 1

        return loss