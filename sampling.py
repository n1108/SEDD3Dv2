import abc
import torch
import torch.nn.functional as F
from catsample import sample_categorical

from model import utils as mutils

_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(
                f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)

    
def get_predictor(name):
    return _PREDICTORS[name]



class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, graph, noise):
        super().__init__()
        self.graph = graph
        self.noise = noise

    @abc.abstractmethod
    def update_fn(self, score_fn, x, t, step_size, current_image_size):
        """One update of the predictor.

        Args:
            score_fn: score function
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
        """
        pass


@register_predictor(name="euler")
class EulerPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size, cond=None, cond_expanded=None, current_image_size=None):
        sigma, dsigma = self.noise(t)
        score = score_fn(x, cond, sigma, current_image_size=current_image_size)

        if cond is not None:
            cond = cond_expanded
        rev_rate = step_size * dsigma[..., None] * self.graph.reverse_rate(x, score, cond)
        x = self.graph.sample_rate(x, rev_rate)
        return x

@register_predictor(name="none")
class NonePredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size, current_image_size):
        return x


@register_predictor(name="analytic")
class AnalyticPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size, cond=None, cond_expanded=None, current_image_size=None):
        curr_sigma = self.noise(t)[0]
        next_sigma = self.noise(t - step_size)[0]
        dsigma = curr_sigma - next_sigma

        score = score_fn(x, cond, curr_sigma, current_image_size=current_image_size)
        if cond is not None:
            cond = cond_expanded

        stag_score = self.graph.staggered_score(score, dsigma, cond)
        probs = stag_score * self.graph.transp_transition(x, dsigma, cond)
        return sample_categorical(probs)

    
class Denoiser:
    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise

    def update_fn(self, score_fn, x, t, cond=None, cond_expanded=None, current_image_size=None):
        sigma = self.noise(t)[0]

        score = score_fn(x, cond, sigma, current_image_size=current_image_size)
        if cond is not None:
            cond = cond_expanded
        stag_score = self.graph.staggered_score(score, sigma, cond)
        probs = stag_score * self.graph.transp_transition(x, sigma, cond)
        # truncate probabilities
        if self.graph.absorb:
            probs = probs[..., :-1]
        
        #return probs.argmax(dim=-1)
        return sample_categorical(probs)
                       

def get_sampling_fn(config, graph, noise, batch_dims, eps, device, cond=None):
    
    sampling_fn = get_pc_sampler(graph=graph,
                                 noise=noise,
                                 batch_dims=batch_dims,
                                 predictor=config.sampling.predictor,
                                 steps=config.sampling.steps,
                                 denoise=config.sampling.noise_removal,
                                 eps=eps,
                                 device=device,
                                 cond=cond)
    
    return sampling_fn
    

def get_pc_sampler(graph, noise, batch_dims, predictor, steps, denoise=True, eps=1e-5, device=torch.device('cpu'), proj_fun=lambda x: x, cond=None, current_image_size=None): # Added current_image_size
    predictor = get_predictor(predictor)(graph, noise)
    projector = proj_fun
    denoiser = Denoiser(graph, noise)
    @torch.no_grad()
    def pc_sampler(model):
        if cond is not None:
            sr = round((batch_dims[1] // cond.shape[1] // cond.shape[2] // cond.shape[3]) ** (1/3))
            cond_expanded = cond.repeat_interleave(sr, dim=1) \
                                .repeat_interleave(sr, dim=2) \
                                .repeat_interleave(sr, dim=3).reshape(cond.shape[0], -1)
        else:
            cond_expanded = None
        sampling_score_fn = mutils.get_score_fn(model, train=False, sampling=True)
        x = graph.sample_limit(*batch_dims, cond=cond_expanded).to(device)
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps

        for i in range(steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = projector(x)
            x = predictor.update_fn(sampling_score_fn, x, t, dt, cond=cond, cond_expanded=cond_expanded, current_image_size=current_image_size)
            

        if denoise:
            # denoising step
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            x = denoiser.update_fn(sampling_score_fn, x, t, cond=cond, cond_expanded=cond_expanded, current_image_size=current_image_size)
            
        return x
    
    return pc_sampler

