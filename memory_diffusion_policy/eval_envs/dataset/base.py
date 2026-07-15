from dataclasses import dataclass
from itertools import chain
import pathlib
from typing import List, Sequence
import numpy as np
from eval_envs.utils.normalize import NormStats, load, save, RunningStats
from eval_envs.utils.normalize import Normalizer, Unnormalizer
from tqdm import tqdm
from eval_envs.utils.transform import DeltaActionCalculator, AbsoluteActionCalculator


@dataclass
class IndexSample:
    buffer_start_idx: int
    buffer_end_idx: int
    sample_start_idx: int
    sample_end_idx: int
    episode_idx: int  # index of the episode
    frame_idx: int  # index of the frame in the current episode
    episode_len: int  # length of the current episode

    def __post_init__(self):
        for field in self.__annotations__:
            value = getattr(self, field)
            setattr(self, field, int(value))
    
    def __str__(self):
        return f"IndexSample(buffer_start_idx={self.buffer_start_idx}, buffer_end_idx={self.buffer_end_idx}, sample_start_idx={self.sample_start_idx}, sample_end_idx={self.sample_end_idx}, episode_idx={self.episode_idx}, frame_idx={self.frame_idx}, episode_len={self.episode_len})"
    
    def __repr__(self):
        return self.__str__()

class EpisodeIndexSample:
    def __init__(self):
        self.episode: List[IndexSample] = []
    
    def append(self, index_sample: IndexSample):
        self.episode.append(index_sample)
    
    def __len__(self):
        return len(self.episode)
    
    def __getitem__(self, idx: int | slice):
        return self.episode[idx]


class DatasetIndexSample:
    def __init__(self):
        self.episode_index_samples: List[EpisodeIndexSample] = []
    
    def append(self, episode_index_sample: EpisodeIndexSample):
        self.episode_index_samples.append(episode_index_sample)
    
    def __len__(self):
        return len(self.episode_index_samples)
    
    def __getitem__(self, idx: int | slice):
        return self.episode_index_samples[idx]
    
    def max_steps(self) -> int:
        return max(len(epi) for epi in self.episode_index_samples)
    
    def min_steps(self) -> int:
        return min(len(epi) for epi in self.episode_index_samples)
    
    def mean_steps(self) -> float:
        return sum(len(epi) for epi in self.episode_index_samples) / len(self.episode_index_samples)
    
    def to_flatten_list(self) -> List[IndexSample]:
        return list(chain(*[epi.episode for epi in self.episode_index_samples]))
    
    def __str__(self):
        return f"[Dataset Statistics] TotalEpisodes={len(self.episode_index_samples)}, " \
            f"TotalSamples={len(self.to_flatten_list())}, " \
            f"MaxLenPerEp={self.max_steps()}, " \
            f"MinLenPerEp={self.min_steps()}, " \
            f"MeanLenPerEp={self.mean_steps()}"
    

class BaseDataset:
    def __init__(
        self,
        data,
        episode_ends,
        obs_horizon=2,
        action_exec_horizon=8,
        action_pred_horizon=16,
        use_delta_action=False,
        mask: Sequence[bool] | None = None
    ):
        """
        data: list of dict or dict of list
        episode_ends: indices of the each episode ends, assume we have a flatten indices for all samples [0, Nsample-1]
        obs_horizon: number of observations to include in the state
        action_exec_horizon: number of actions to be executed during inference
        action_pred_horizon: number of actions to predict by the model
        use_delta_action: whether to use delta action or absolute action
        mask: usually the grasp action is masked out, e.g, [1,1,1,1,1,1,1,0] for 7-DoF action
        
        Follow Diffusion Policy
        #|o|o|                             obs_horizon: 2
        #| |a|a|a|a|a|a|a|a|               actions executed: 8
        #|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
        
        """
        self.data = data  # list of dict or dict of list
        self.episode_ends = episode_ends
        self.obs_horizon = obs_horizon
        self.action_exec_horizon = action_exec_horizon
        self.action_pred_horizon = action_pred_horizon
        self.obs_keys = ("image", "state")  # fixed keys
        self.action_keys = ("action",)  # fixed keys

        if use_delta_action:
            self.abs2delta = DeltaActionCalculator(mask=mask)
            self.delta2abs = AbsoluteActionCalculator(mask=mask)
        else:
            self.abs2delta = lambda x: x
            self.delta2abs = lambda x: x

        self.stats: dict[str, NormStats] | None = None

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def load_stats(self, path: pathlib.Path | str):
        self.stats = load(path)

    def save_stats(self, path: pathlib.Path | str):
        save(path, self.stats)

    def set_normalizer(self, norm_type: str = "quantile"):
        self.normalizer = Normalizer(
            norm_stats=self.stats, norm_type=norm_type)
        self.unnormalizer = Unnormalizer(
            norm_stats=self.stats, norm_type=norm_type)

    def compute_stats(self):
        stats_runners = {"state": RunningStats(), "action": RunningStats()}
        for i in tqdm(range(len(self)), desc="Computing Stats for Raw Data"):
            sample = self[i]
            sample = self.abs2delta(sample)
            for key in ["state", "action"]:
                stats_runners[key].update(
                    sample[key].reshape(-1, sample[key].shape[-1]))

        self.stats = dict()
        for key, runner in stats_runners.items():
            self.stats[key] = runner.get_statistics()
        return self.stats
    

        

class SampleLevelDataset(BaseDataset):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepare_indices()

    def prepare_indices(self):
        dataset_indices = self.create_sample_indices(
            episode_ends=self.episode_ends,
            sequence_length=self.action_pred_horizon,
            pad_before=self.obs_horizon-1,
            pad_after=max(self.action_pred_horizon-self.action_exec_horizon-self.obs_horizon+1, 0),
            # pad_after= self.action_exec_horizon-1, # original DP paper
        )
        print(dataset_indices)
        self.indices = dataset_indices.to_flatten_list()
    
    @staticmethod
    def create_sample_indices(
            episode_ends,
            sequence_length,
            pad_before=0,
            pad_after=0,
    ) -> DatasetIndexSample:
        indices = DatasetIndexSample()
        for i in range(len(episode_ends)):
            # for current episode
            cur_epi_list = EpisodeIndexSample()

            start_idx = 0
            if i > 0:
                start_idx = episode_ends[i-1]

            end_idx = episode_ends[i]
            episode_length = end_idx - start_idx

            min_start = -pad_before
            max_start = episode_length - sequence_length + pad_after

            # range stops one idx before end
            for idx in range(min_start, max_start+1):
                buffer_start_idx = max(idx, 0) + start_idx
                buffer_end_idx = min(idx+sequence_length,
                                     episode_length) + start_idx
                start_offset = buffer_start_idx - (idx+start_idx)
                end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
                sample_start_idx = 0 + start_offset
                sample_end_idx = sequence_length - end_offset
                cur_epi_list.append(
                    IndexSample(
                        buffer_start_idx=buffer_start_idx,
                        buffer_end_idx=buffer_end_idx,
                        sample_start_idx=sample_start_idx,
                        sample_end_idx=sample_end_idx,
                        episode_idx=i,
                        frame_idx=idx,
                        episode_len=int(episode_length),
                    ))
            indices.append(cur_epi_list)

        return indices

    @staticmethod
    def sample_sequence_from_dict_data(
        train_data, 
        sequence_length: int,
        index_sample: IndexSample
    ):
        # data is a dict of numpy arrays
        # e.g. {"image": np.array(N, H, W, 3), "state": np.array(N, 2), 
        #       "action": np.array(N, 2), "episode_idx": np.array(N), 
        #       "frame_idx": np.array(N), "episode_len": np.array(N)}
        # N is the number of samples in the buffer
        result = dict()
        for key, input_arr in train_data.items():
            sample = input_arr[index_sample.buffer_start_idx:index_sample.buffer_end_idx]
            data = sample
            if (index_sample.sample_start_idx > 0) or (index_sample.sample_end_idx < sequence_length):
                data = np.zeros(
                    shape=(sequence_length,) + input_arr.shape[1:],
                    dtype=input_arr.dtype)
                if index_sample.sample_start_idx > 0:
                    data[:index_sample.sample_start_idx] = sample[0]
                if index_sample.sample_end_idx < sequence_length:
                    data[index_sample.sample_end_idx:] = sample[-1]
                data[index_sample.sample_start_idx:index_sample.sample_end_idx] = sample
            result[key] = data
        return result

    @staticmethod
    def sample_sequence_from_seq_data(
        train_data, 
        sequence_length: int,
        index_sample: IndexSample):
        # TODO: didnt test this yet
        result = dict()

        samples = train_data[index_sample.buffer_start_idx:index_sample.buffer_end_idx]
        sample_keys = samples[0].keys()

        # Initialize result dict with empty lists
        for key in sample_keys:
            result[key] = []

        # Handle padding at the beginning (repeat first sample)
        if index_sample.sample_start_idx > 0:
            first_sample = samples[0]
            for _ in range(index_sample.sample_start_idx):
                for key in sample_keys:
                    result[key].append(first_sample[key])

        # Add the actual samples
        for sample in samples:
            for key in sample_keys:
                result[key].append(sample[key])

        # Handle padding at the end (repeat last sample)
        if index_sample.sample_end_idx < sequence_length:
            last_sample = samples[-1]
            padding_length = sequence_length - index_sample.sample_end_idx
            for _ in range(padding_length):
                for key in sample_keys:
                    result[key].append(last_sample[key])

        # Convert lists to numpy arrays
        for key in sample_keys:
            result[key] = np.array(result[key])

        return result

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int | slice):
        # Final output dict
        # state: [BatchSize, ObsHorizon, StateDim]
        # action: [BatchSize, ActionPredHorizon, ActionDim]
        # image: [BatchSize, ObsHorizon, H, W, 3]
        # episode_idx: [BatchSize]
        # frame_idx: [BatchSize]
        # episode_len: [BatchSize]

        # only return the original sample, not the normalized sample
        index_sample = self.indices[idx]

        if isinstance(self.data, list):
            sample = self.sample_sequence_from_seq_data(
                self.data,
                self.action_pred_horizon,
                index_sample)
        else:
            sample = self.sample_sequence_from_dict_data(
                self.data,
                self.action_pred_horizon,
                index_sample)

        # discard unused observations
        for key in self.obs_keys:
            sample[key] = sample[key][:self.obs_horizon, :]
        for key in self.action_keys:
            sample[key] = sample[key][:self.action_pred_horizon, :]

        sample["episode_idx"] = index_sample.episode_idx
        sample["frame_idx"] = index_sample.frame_idx
        sample["episode_len"] = index_sample.episode_len

        return sample


class EpisodeLevelDataset(BaseDataset):
    def __init__(self, stride=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # if stride > 1, the data is sampled every stride steps
        # rather than dense data

        self.stride = stride

    def prepare_data(self):
        indices = self.create_sample_indices(
            episode_ends=self.episode_ends,
            sequence_length=self.action_pred_horizon,
            pad_before=self.obs_horizon-1,
            pad_after=self.action_exec_horizon-1,
        )
        self.indices = []
        for epi_indices in indices:
            for ii in range(self.stride):
                self.indices.append(epi_indices[ii::self.stride])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int | slice):
        # Final output dict
        # state: [BatchSize, SeqLen, ObsHorizon, StateDim]
        # action: [BatchSize, SeqLen, ActionPredHorizon, ActionDim]
        # image: [BatchSize, SeqLen, ObsHorizon, H, W, 3]
        # episode_idx: [BatchSize]
        # frame_idx: [BatchSize, SeqLen]
        # episode_len: [BatchSize]
        raise NotImplementedError