import torch
import torch.nn as nn
class BaseModel(torch.nn.Module):
    
    def __init__(self, device: str = "cuda:0", *args, **kwargs):
        if len(kwargs) > 0:
            print("some kwargs are not used", kwargs)
            kwargs = {}
        super().__init__(*args, **kwargs)
        self.device = device
    
    def predict_action(self, obs_dict, policy_info) -> tuple[dict, dict]:
        # obs_dict: {'image': torch.Tensor, 'state': torch.Tensor}
        # return action_dict, policy_info
        # action_dict: {'action': torch.Tensor, ...}, [B, action_exec_horizon, action_dim]
        raise NotImplementedError
    
    def compute_loss(self, batch_dict) -> tuple[torch.Tensor, dict]:
        # batch_dict: {'image': torch.Tensor, 'state': torch.Tensor, 'action': torch.Tensor}
        # return loss and info for logging
        raise NotImplementedError