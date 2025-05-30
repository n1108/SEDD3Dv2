import torch
import torch.nn.functional as F


def get_model_fn(model, train=False):
    """Create a function to give the output of the score-based model.

    Args:
        model: The score model.
        train: `True` for training and `False` for evaluation.
        mlm: If the input model is a mlm and models the base probability 

    Returns:
        A model function.
    """

    def model_fn(x, cond, sigma, current_image_size=None): # <--- 新增 current_image_size
        """Compute the output of the score-based model.

        Args:
            x: A mini-batch of input data.
            labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
              for different models.

        Returns:
            A tuple of (model output, new mutable states)
        """
        if train:
            model.train()
        else:
            model.eval()
        
            # otherwise output the raw values (we handle mlm training in losses.py)
        return model(x, cond, sigma, current_image_size=current_image_size) # <--- 修改调用

    return model_fn


def get_score_fn(model, train=False, sampling=False):
    if sampling:
        assert not train, "Must sample in eval mode"
    model_fn = get_model_fn(model, train=train)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        def score_fn(x, cond, sigma, current_image_size=None): # <--- 新增 current_image_size
            sigma = sigma.reshape(-1)
            score = model_fn(x, cond, sigma, current_image_size=current_image_size) # <--- 修改调用
            
            if sampling:
                # when sampling return true score (not log used for training)
                return score.exp()
                
            return score

    return score_fn
