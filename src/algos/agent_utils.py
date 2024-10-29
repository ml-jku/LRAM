import zipfile
import torch
import numpy as np
from torch.nn.parallel import DistributedDataParallel
from stable_baselines3.common.save_util import data_to_json, open_path, get_system_info


def make_random_proj_matrix(in_dim, proj_dim, seed=42, norm=False, device=None, batch_size=None):
    # to deterministically get the same projection matrix (for every size), fix the rng seed
    rng = np.random.RandomState(seed)
    shape = (proj_dim, in_dim) if batch_size is None else (batch_size, proj_dim, in_dim)
    # scale = np.sqrt(in_dim / proj_dim)
    scale = np.sqrt(1 / proj_dim)
    rand_matrix = rng.normal(loc=0, scale=scale, size=shape).astype(dtype=np.float32)
    if norm: 
        norms = np.linalg.norm(rand_matrix, axis=0) + 1e-8
        rand_matrix = rand_matrix / norms
    if device is not None: 
        rand_matrix = torch.from_numpy(rand_matrix).to(device)
    return rand_matrix


def dropout_dims(x, p=0.5, dim=None):
    """
    Drops out dimnensions of given vector. E.g., useful for continous state/actions

    Args:
        x (Tensor): Input tensor.
        p (float, optional): Dropout probability. Default is 0.5.
    Returns:
        Tensor: Input tensor with dropped out dims
    """
    if dim is not None: 
        shape = list(x.shape)
        shape[dim] = 1
        mask = torch.bernoulli(torch.full(shape, 1-p, device=x.device)).long()
    else: 
        mask = torch.bernoulli(torch.full_like(x, 1-p)).long()
    return x * mask


def make_gaussian_noise(x, mean=0.0, std=0.1, nonzero=True, constant=True):
    """
    Makes Gaussian noise for a tensor input.

    Args:
        x (Tensor): Input tensor with shape [batch_size, seq_len, dim].
        mean (float, optional): Mean of the Gaussian distribution. Default is 0.0.
        std (float, optional): Standard deviation of the Gaussian distribution. Default is 1.0.
    Returns:
        Tensor: Noise.
    """
    if std is None: 
        std = 0.1
    if len(x.shape) == 1: 
        noise = torch.normal(mean=mean, std=std, size=(x.shape[0],), device=x.device)
    else: 
        if constant: 
            batch_size, seq_len, dim = x.shape
            # constant noise along seq_len
            noise = torch.normal(mean=mean, std=std, size=(batch_size, 1, dim), device=x.device)
        else: 
            noise = torch.normal(mean=mean, std=std, size=x.shape, device=x.device)
    if nonzero: 
        # handles padding + 0-dims in metaworld/dmc
        noise = noise * (x != 0)
    return noise


def add_gaussian_noise(x, mean=0.0, std=0.1, nonzero=True, constant=True):
    return x + make_gaussian_noise(x, mean=mean, std=std, nonzero=nonzero, constant=constant)


class HLGaussLoss(torch.nn.Module):
    
    def __init__(self, min_value=-1, max_value=1, num_bins=64, sigma=0.01, bin_std_ratio=0.75, reduction="mean"):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.num_bins = num_bins
        self.bin_width = (max_value - min_value) / num_bins
        self.sigma = sigma
        self.bin_std_ratio = bin_std_ratio
        self.reduction = reduction
        if bin_std_ratio is not None: 
            # set as as proposed by: https://arxiv.org/abs/2403.03950
            # distributes probability mass to ~6 locations. 
            self.sigma = self.bin_width * bin_std_ratio
        self.register_buffer('support', torch.linspace(min_value, max_value, num_bins + 1, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(logits, self.transform_to_probs(target), reduction=self.reduction)
    
    def transform_to_probs(self, target: torch.Tensor) -> torch.Tensor:
        target = torch.clamp(target, self.min_value, self.max_value)
        cdf_evals = torch.special.erf((self.support - target.unsqueeze(-1)) / (torch.sqrt(torch.tensor(2.0)) * self.sigma))
        z = cdf_evals[..., -1] - cdf_evals[..., 0]
        bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
        return bin_probs / z.unsqueeze(-1)

    def transform_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        centers = (self.support[:-1] + self.support[1:]) / 2
        return torch.sum(probs * centers, dim=-1)
    

def make_loss_fn(kind, reduction="mean", label_smoothing=0.0, loss_fn_kwargs=None):
    reduction = loss_fn_kwargs.get("reduction", reduction) if loss_fn_kwargs is not None else reduction
    if kind in ["mse", "td3+bc"]:
        loss_fn = torch.nn.MSELoss(reduction=reduction)
    elif kind in ["smooth_l1", "dqn"]:
        loss_fn = torch.nn.SmoothL1Loss(reduction=reduction)
    elif kind == "huber":
        loss_fn = torch.nn.HuberLoss(reduction=reduction)
    elif kind == "nll":
        loss_fn = torch.nn.NLLLoss(reduction=reduction)
    elif kind == "ce":
        loss_fn = torch.nn.CrossEntropyLoss(reduction=reduction, label_smoothing=label_smoothing)
    elif kind in ["td3", "ddpg", "sac"]:
        loss_fn = None
    elif kind == "hl_gauss": 
        loss_fn_kwargs = {} if loss_fn_kwargs is None else loss_fn_kwargs
        loss_fn = HLGaussLoss(**loss_fn_kwargs)
    else:
        raise ValueError(f"Unknown loss kind: {kind}")
    return loss_fn


class CustomDDP(DistributedDataParallel):
    """
    The default DistributedDataParallel enforces access to class the module attributes via self.module. 
    This is impractical for our use case, as we need to access certain module access throughout. 
    We override the __getattr__ method to allow access to the module attributes directly.
    
    For example: 
    ```
        # default behaviour
        model = OnlineDecisionTransformerModel()
        model = DistributedDataParallel(model)
        model.module.some_attribute
        
        # custom behaviour using this class
        model = OnlineDecisionTransformerModel()
        model = CustomDDP(model)
        model.some_attribute
        
    ```        
    Shoudl not cause any inconsistencies: 
    https://discuss.pytorch.org/t/access-to-attributes-of-model-wrapped-in-ddp/130572
    
    """
    
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def get_param_count(model, prefix="model"):    
    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {f"{prefix}_total": params, f"{prefix}_trainable": trainable_params}


def save_to_zip_file_fixed(
    save_path,
    data=None,
    params=None,
    pytorch_variables=None,
    verbose: int = 0,
) -> None:
    """
    Save model data to a zip archive.

    :param save_path: Where to store the model.
        if save_path is a str or pathlib.Path ensures that the path actually exists.
    :param data: Class parameters being stored (non-PyTorch variables)
    :param params: Model parameters being stored expected to contain an entry for every
                   state_dict with its name and the state_dict.
    :param pytorch_variables: Other PyTorch variables expected to contain name and value of the variable.
    :param verbose: Verbosity level, 0 means only warnings, 2 means debug information
    """
    save_path = open_path(save_path, "w", verbose=0, suffix="zip")
    # data/params can be None, so do not
    # try to serialize them blindly
    if data is not None:
        serialized_data = data_to_json(data)

    # Create a zip-archive and write our objects there.
    with zipfile.ZipFile(save_path, mode="w") as archive:
        # Do not try to save "None" elements
        if data is not None:
            archive.writestr("data", serialized_data)
        if pytorch_variables is not None:
            with archive.open("pytorch_variables.pth", mode="w", force_zip64=True) as pytorch_variables_file:
                torch.save(pytorch_variables, pytorch_variables_file)
        if params is not None:
            for file_name, dict_ in params.items():
                with archive.open(file_name + ".pth", mode="w", force_zip64=True) as param_file:
                    torch.save(dict_, param_file)
        # Save system info about the current python env
        archive.writestr("system_info.txt", get_system_info(print_info=False)[1])
