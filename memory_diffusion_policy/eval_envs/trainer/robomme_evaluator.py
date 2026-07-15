# eval_envs/trainer/robomme_evaluator.py
"""
Standalone evaluator for the RoboMME benchmark.

Requires PYTHONPATH to include the robomme_policy_learning examples directory:
    PYTHONPATH=<robomme_policy_learning>/examples/robomme:$PYTHONPATH

EnvRunner API (from robomme_policy_learning/examples/robomme/env_runner.py):
  - __init__(env_id, video_save_dir, max_steps)
  - make_env(episode_id)              -> None  (difficulty via self.env.unwrapped.difficulty)
  - get_init_obs()                    -> {"images", "wrist_images", "states", "task_goal"}
  - step(action)                      -> ((img, wrist_img, state), stop_flag, outcome_str)
  - close_env()
"""
import os
import numpy as np
import torch
from typing import Optional, List, Dict, Any, Tuple
from collections import deque

from eval_envs.model.base import BaseModel
from eval_envs.utils.transform import DataTransform

# RoboMME environment runner.
# Set PYTHONPATH=<robomme_policy_learning>/examples/robomme:$PYTHONPATH before running.
from env_runner import EnvRunner
from utils import RolloutRecorder


class _RoboMMEFrameHistory:
    """Stores the last n_obs_steps observations."""

    def __init__(self, n_obs_steps: int):
        self.n = int(n_obs_steps)
        self.buf: deque = deque(maxlen=self.n)

    def push(self, obs: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        self.buf.append(obs)

    def warmfill(self, obs: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        self.buf.clear()
        for _ in range(self.n):
            self.buf.append(obs)

    def get_history(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        assert len(self.buf) == self.n
        return list(self.buf)


class Evaluator:
    """
    Direct (non-websocket) RoboMME evaluator for Diffusion Policy.
    Runs n_eval_episodes per task and reports success rate.
    """

    def __init__(
        self,
        output_dir: str,
        task_name: str,
        *,
        n_eval_episodes: Optional[int] = None,
        max_steps: int = 1300,
        n_obs_steps: int = 1,
        n_action_steps: int = 16,
        n_action_exec: int = 8,
        seed: int = 0,
        use_wrist: bool = True,
        save_video: bool = False,
        video_dir: Optional[str] = None,
    ):
        assert n_action_exec >= 1
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.task_name = task_name
        self.max_steps = int(max_steps)
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_plan = int(n_action_steps)
        self.n_action_exec = int(n_action_exec)
        self.seed = int(seed)
        self.use_wrist = bool(use_wrist)
        self.save_video = bool(save_video)
        self.video_dir = video_dir or os.path.join(self.output_dir, "videos")
        os.makedirs(self.video_dir, exist_ok=True)

        # EnvRunner does not accept a render flag
        self.env_runner = EnvRunner(self.task_name, self.video_dir, self.max_steps)

        if n_eval_episodes is None:
            self.n_eval_episodes = int(self.env_runner.num_episodes)
        else:
            self.n_eval_episodes = int(n_eval_episodes)

        self._img_hw = None

    def _img_to_chw_float01(self, img: np.ndarray, device: torch.device) -> torch.Tensor:
        t = torch.as_tensor(img, device=device)
        if t.ndim != 3:
            raise ValueError(f"Expected 3D image, got {tuple(t.shape)}")
        if t.shape[-1] == 3:
            t = t.permute(2, 0, 1)
        elif t.shape[0] == 3:
            pass
        else:
            raise ValueError(f"Expected RGB image with 3 channels, got {tuple(t.shape)}")
        t = t.to(torch.float32)
        if t.max() > 1.0:
            t = t / 255.0
        return t.contiguous()

    def _build_obs_from_history(
        self,
        history: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        data_transform: DataTransform,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        T = self.n_obs_steps
        assert len(history) == T

        imgs_t: List[torch.Tensor] = []
        states_t: List[np.ndarray] = []

        for (img, wrist_img, robot_state) in history:
            img_t = self._img_to_chw_float01(img, device=device)
            if self.use_wrist:
                w_t = self._img_to_chw_float01(wrist_img, device=device)
                img_t = torch.cat([img_t, w_t], dim=0)

            st = np.asarray(robot_state, dtype=np.float32).reshape(-1)
            if st.shape[0] != 8:
                raise ValueError(f"Expected robot_state dim=8, got {st.shape}.")

            imgs_t.append(img_t)
            states_t.append(st)

        imgs_t = torch.stack(imgs_t, dim=0)          # (T, C, H, W)
        states_np = np.stack(states_t, axis=0)        # (T, 8)

        if self._img_hw is None:
            self._img_hw = tuple(imgs_t.shape[-2:])

        batch_np = {
            "image": imgs_t.detach().cpu().numpy()[None, ...],  # (1, T, C, H, W)
            "state": states_np[None, ...],                      # (1, T, 8)
        }
        batch_np = data_transform.transform_in(batch_np)

        return {
            "image": torch.from_numpy(batch_np["image"]).float().to(device),
            "state": torch.from_numpy(batch_np["state"]).float().to(device),
        }

    @staticmethod
    def _to_numpy_tree(tree):
        def f(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return x
        if isinstance(tree, dict):
            return {k: Evaluator._to_numpy_tree(v) for k, v in tree.items()}
        if isinstance(tree, (list, tuple)):
            return type(tree)(Evaluator._to_numpy_tree(v) for v in tree)
        return f(tree)

    def run(self, policy: BaseModel, data_transform: DataTransform) -> Dict[str, Any]:
        """
        Runs n_eval_episodes on a single RoboMME task.
        Success is determined by step() returning outcome == "success".
        """
        policy.eval()
        device = next(policy.parameters()).device

        successes: List[float] = []

        for ep in range(self.n_eval_episodes):
            self.env_runner.make_env(ep)
            pre_traj = self.env_runner.get_init_obs()

            task_goal: str = pre_traj["task_goal"]
            images = pre_traj["images"]
            wrist_images = pre_traj["wrist_images"]
            states = pre_traj["states"]

            recorder = None
            if self.save_video:
                recorder = RolloutRecorder(self.video_dir, task_goal, fps=30)
                for i in range(len(images)):
                    recorder.record(
                        image=images[i].copy(),
                        wrist_image=wrist_images[i].copy(),
                        state=states[i].copy(),
                    )

            init_obs = (images[-1], wrist_images[-1], states[-1])
            hist = _RoboMMEFrameHistory(self.n_obs_steps)
            hist.warmfill(init_obs)

            steps = 0
            done = False
            outcome = "unknown"
            policy_info: Dict[str, Any] = {}

            with torch.no_grad():
                while steps < self.max_steps and not done:
                    obs_dict = self._build_obs_from_history(
                        hist.get_history(), data_transform, device
                    )
                    action_dict, policy_info = policy.predict_action(obs_dict, policy_info)
                    np_action_dict = self._to_numpy_tree(action_dict)
                    np_action_dict = data_transform.transform_out(np_action_dict)

                    act = np.asarray(np_action_dict["action"], dtype=np.float32)
                    if act.ndim == 3:
                        act = act[0]
                    elif act.ndim == 1:
                        act = act[None, :]

                    K = min(self.n_action_exec, act.shape[0])

                    for t in range(K):
                        if done or steps >= self.max_steps:
                            break

                        step_action = act[t].reshape(-1).astype(np.float32)
                        # RoboMME step returns ((img, wrist_img, state), stop_flag, outcome_str)
                        (img, wrist_img, state), stop_flag, outcome = self.env_runner.step(step_action)

                        rs = np.asarray(state, dtype=np.float32).reshape(-1)
                        hist.push((img, wrist_img, rs))

                        if recorder is not None:
                            recorder.record(
                                image=img.copy(),
                                wrist_image=wrist_img.copy(),
                                state=rs.copy(),
                                action=step_action.copy(),
                            )

                        steps += 1
                        done = bool(stop_flag)
                        if done:
                            break

            success_bool = (outcome == "success")
            successes.append(1.0 if success_bool else 0.0)

            if recorder is not None:
                vid_name = f"{self.task_name}_ep{ep}_{outcome}.mp4"
                recorder.save_video(vid_name)

            print(
                f"[RoboMME Eval] task={self.task_name} ep={ep+1}/{self.n_eval_episodes} "
                f"steps={steps} result={outcome}"
            )
            self.env_runner.close_env()

        mean_success = float(np.mean(successes)) if successes else 0.0
        std_success = float(np.std(successes)) if successes else 0.0
        results = {
            "success_rate": mean_success,
            "std_success_rate": std_success,
            "individual_successes": successes,
            "n_episodes": self.n_eval_episodes,
            "task_name": self.task_name,
        }

        out_path = os.path.join(self.output_dir, "robomme_eval_stats.pt")
        torch.save(results, out_path)
        print(
            f"[RoboMME Eval] Final: {mean_success:.3f} ± {std_success:.3f} "
            f"({int(sum(successes))}/{len(successes)})"
        )
        print(f"[RoboMME Eval] Saved: {out_path}")
        return results

    def close(self):
        if hasattr(self.env_runner, "close_env"):
            try:
                self.env_runner.close_env()
            except Exception:
                pass
