# from munch import munchify
import torch

# from corruption import build_corruption

__couplings__ = {}

def register_coupling(name):
    def decorator(cls):
        if name in __couplings__:
            raise ValueError('Cannot register duplicate coupling ({})'.format(name))
        __couplings__[name] = cls
        return cls
    return decorator

def get_coupling(coupling, batch_size, **kwargs):
    if 'inverse' in coupling:
        coupling, task = coupling.split('_')
        return __couplings__[coupling](batch_size, task, **kwargs)
    if __couplings__.get(coupling) is None:
        raise ValueError('Coupling {} not found'.format(coupling))
    return __couplings__[coupling](batch_size, **kwargs)

@register_coupling('independent')
class IndependentCoupling:
    def __init__(self, batch_size, **kwargs):
        self.batch_size = batch_size
    
    def __call__(self, X0, X1):
        X0, X1 = X0[0].cuda(), X1[0].cuda() # remove labels
        idx_X0 = torch.randperm(X0.shape[0])[:self.batch_size]
        idx_X1 = torch.randperm(X1.shape[0])[:self.batch_size]
        return X0[idx_X0], X1[idx_X1]

@register_coupling('pix2pix')
class Pix2PixCoupling:
    def __init__(self, batch_size, **kwargs):
        self.batch_size = batch_size
    
    def __call__(self, X0, X1):
        X0, X1 = X0[0].cuda(), X1[0].cuda() # remove labels
        size = X0.shape[2]
        X0, X1 = X0[...,size:], X0[...,:size]
        if X0.shape[0] >= self.batch_size:
            idx = torch.randperm(X0.shape[0])[:self.batch_size]
        else:
            idx = torch.randint(high=X0.shape[0], size=[self.batch_size])
        return X0[idx], X1[idx]

@register_coupling('ot')
class OTCoupling:
    def __init__(self, batch_size, reg=0.01, maxiter=30, **kwargs):
        self.batch_size = batch_size
        self.reg = reg
        self.maxiter = maxiter
    
    def __sinkhorn__(self, cost_matrix, reg=1e-1, maxiter=30, momentum=0.):
        """Log domain version on sinkhorn distance algorithm (https://arxiv.org/abs/1306.0895).
        Inspired by https://github.com/gpeyre/SinkhornAutoDiff/blob/master/sinkhorn_pointcloud.py ."""
        m, n = cost_matrix.size()

        mu = torch.FloatTensor(m).fill_(1./m)
        nu = torch.FloatTensor(n).fill_(1./n)

        if torch.cuda.is_available():
            mu, nu = mu.cuda(), nu.cuda()

        def M(u, v):
            "Modified cost for logarithmic updates"
            "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
            return (-cost_matrix + u.unsqueeze(1) + v.unsqueeze(0)) / reg

        u, v = 0. * mu, 0. * nu        

        # Actual Sinkhorn loop
        for i in range(maxiter):
            u1, v1 = u, v
            u = reg * (torch.log(mu) - torch.logsumexp(M(u, v), dim=1)) + u
            v = reg * (torch.log(nu) - torch.logsumexp(M(u, v).t(), dim=1)) + v
            if momentum > 0.:
                u = -momentum * u1 + (1+momentum) * u
                v = -momentum * v1 + (1+momentum) * v

        pi = torch.exp(M(u, v))  # Transport plan pi = diag(a)*K*diag(b)
        cost = torch.sum(pi * cost_matrix)  # Sinkhorn cost
        return pi
    
    def __call__(self, x_0, x_T):
        x_0, x_T = x_0[0].cuda(), x_T[0].cuda() # remove labels
        cost_matrix = (x_T[:,None] - x_0[None]).flatten(start_dim=2).norm(dim=2)
        pi = self.__sinkhorn__(cost_matrix, self.reg, self.maxiter)
        idx_X1 = torch.randperm(x_T.shape[0])[:self.batch_size]
        idx_X0 = torch.multinomial(pi,1).reshape(-1)[idx_X1]
        return x_0[idx_X0], x_T[idx_X1]

# @register_coupling('inverse')
# class InverseCoupling:
#         def __init__(self, batch_size, task, **kwargs):
#             self.batch_size = batch_size
#             opt = munchify({'device':kwargs.get('device', 'cuda'),
#                             'image_size':kwargs.get('image_size', 256)})
#             self.task = task
#             self.operator = build_corruption(opt=opt, corrupt_type=task)
        
#         def __call__(self, X0, _):
#             X0 = X0[0].cuda() # remove labels
#             if 'inpaint' in self.task:
#                 with torch.no_grad():
#                     X1, mask = self.operator(X0)
#                     if mask is not None:
#                         mask = mask.detach().to(X0.device)
#                         X1 = (1. - mask) * X0 + mask * torch.randn_like(X1)
#             else:
#                 with torch.no_grad():
#                     X1 = self.operator(X0)
#             return X0, X1

@register_coupling('latent')
class LatentCoupling:
    def __init__(self, batch_size, **kwargs):
        self.batch_size = batch_size
        data_stats = torch.load('data_stats/cifar10.pt')
        self.mus = data_stats['mus'].cuda()
        self.stds = data_stats['stds'].cuda()
    
    def __call__(self, X0, X1):
        X0, y0 = X0[0].cuda(), X0[1].cuda()
        X1 = self.mus[y0] + torch.randn_like(X0) * self.stds[y0]
        idx = torch.randperm(X0.shape[0])[:self.batch_size]
        return X0[idx], X1[idx]