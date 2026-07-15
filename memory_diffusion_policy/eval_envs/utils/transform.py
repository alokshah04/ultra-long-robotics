import logging
from typing import Sequence
import numpy as np
from copy import deepcopy
from eval_envs.utils.normalize import Normalizer, Unnormalizer, NormStats


class DeltaActionCalculator:
    """Repacks absolute actions into delta action space."""
    
    def __init__(self, mask: Sequence[bool] | None): # grasp should be False
        self.mask = mask


    def __call__(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if "action" not in data:
            logging.warning("No action in data to transform")
            return data

        data_ = deepcopy(data)
        state, actions = data_["state"], data_["action"]
        if state.ndim == actions.ndim: # obs_horizon >1, take the latest state
            state = state[..., -1, :]
        if self.mask is None:
            self.mask = [True] * state.shape[-1]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data_["action"] = actions

        return data_


class AbsoluteActionCalculator:
    """Repacks delta actions into absolute action space."""

    def __init__(self, mask: Sequence[bool] | None): # grasp should be False
        self.mask = mask

    def __call__(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if "action" not in data:
            logging.warning("No action in data to transform")
            return data
        
        data_ = deepcopy(data)
        state, actions = data_["state"], data_["action"]
        if state.ndim == actions.ndim: # obs_horizon >1, take the latest state
            state = state[..., -1, :]
        if self.mask is None:
            self.mask = [True] * state.shape[-1]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data_["action"] = actions

        return data_


class DataTransform:
    def __init__(self, norm_stats: dict[str, NormStats], norm_type: str = "quantile", mask: Sequence[bool] | None = None, use_delta_action: bool = False, training: bool = True, eps: float = 1e-6, chunk_norm_power: float | None = None, chunk_norm_zero_start: bool | None = None):
        print(f"[Init DataTransform] norm_type: {norm_type}, mask: {mask}, use_delta_action: {use_delta_action}")
        self.norm_stats = norm_stats
        self.norm_type = norm_type
        self.mask = mask
                
        if use_delta_action:
            self.delta_action_calculator = DeltaActionCalculator(mask=mask)
            self.absolute_action_calculator = AbsoluteActionCalculator(mask=mask)
        else:
            self.delta_action_calculator = lambda x: x
            self.absolute_action_calculator = lambda x: x
            
        self.normalizer = Normalizer(norm_stats=norm_stats, norm_type=norm_type, eps=eps, chunk_norm_power=chunk_norm_power, chunk_norm_zero_start=chunk_norm_zero_start)
        self.unnormalizer = Unnormalizer(norm_stats=norm_stats, norm_type=norm_type, eps=eps, chunk_norm_power=chunk_norm_power, chunk_norm_zero_start=chunk_norm_zero_start)
        
        self.training = training
        
        
    def transform_in(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.training: # only apply delta action in training, in eval the input do not have action
            data = self.delta_action_calculator(data)
        data = self.normalizer(data)
        return data
    
    def transform_out(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        data = self.unnormalizer(data)
        data = self.absolute_action_calculator(data)
        return data
    
        
        

if __name__ == "__main__":
    from copy import deepcopy
    data = {
        "state": np.arange(1, 64+1).reshape(64, 1).astype(np.float32).repeat(8, axis=-1)*10,
        "action": np.arange(64*16*8, dtype=np.float32).reshape(64, 16, 8)
    }
    delta_action_calculator = DeltaActionCalculator(mask=[1,1,1,1,1,1,1,0])
    absolute_action_calculator = AbsoluteActionCalculator(mask=[1,1,1,1,1,1,1,0])
    da = delta_action_calculator(data)
    aa = absolute_action_calculator(da)
    assert np.allclose(data["action"], aa["action"])
    for i in range(64):
        print(da["action"][i])
        print(aa["action"][i])
        input()
    