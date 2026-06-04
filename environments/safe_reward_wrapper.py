import gymnasium as gym
import numpy as np
from configs.config import SAFE_CONFIG

class SafeRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.cfg = SAFE_CONFIG
        
        # Metric Counters
        self.collision_count = 0
        self.tailgating_count = 0
        self.unsafe_lane_change_count = 0
        self.overspeed_count = 0
        self.total_steps = 0

    @staticmethod
    def _action_to_int(action):
        if isinstance(action, np.ndarray):
            return int(action.item())
        if hasattr(action, "item"):
            return int(action.item())
        return int(action)

    def reset(self, **kwargs):
        self.collision_count = 0
        self.tailgating_count = 0
        self.unsafe_lane_change_count = 0
        self.overspeed_count = 0
        self.total_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_steps += 1
        
        original_reward = float(reward)
        info["original_reward"] = original_reward
        ego_vehicle = self.env.unwrapped.vehicle
        lam = self.cfg["safety_lambda"]
        action_id = self._action_to_int(action)
        collision_violation = False
        tailgating_violation = False
        unsafe_lane_change_violation = False
        overspeed_violation = False
        reward_components = {
            "progress_reward": original_reward,
            "collision_penalty": 0.0,
            "tailgating_penalty": 0.0,
            "lane_change_penalty": 0.0,
            "speed_penalty": 0.0,
        }
        
        # In highway-env, lane_index is usually (from, to, index)
        # We need the index (the third element)
        ego_lane_id = ego_vehicle.lane_index[2]
        
        # 1. Collision Penalty
        if info.get("crashed", False):
            penalty = lam * self.cfg["collision_penalty"]
            reward -= penalty
            reward_components["collision_penalty"] = penalty
            self.collision_count += 1
            collision_violation = True
            
        # 2. Tailgating Penalty (Time Gap based)
        for vehicle in self.env.unwrapped.road.vehicles:
            if vehicle is ego_vehicle:
                continue
            
            # Check if vehicle is in front and in the same lane
            # vehicle.lane_index[2] is the lane id
            if (vehicle.lane_index[2] == ego_lane_id and 
                vehicle.position[0] > ego_vehicle.position[0]):
                
                distance = vehicle.position[0] - ego_vehicle.position[0]
                time_gap = distance / max(ego_vehicle.speed, 0.1)
                
                if time_gap < self.cfg["safe_time_gap"]:
                    penalty = lam * self.cfg["tailgating_penalty"]
                    reward -= penalty
                    reward_components["tailgating_penalty"] = penalty
                    self.tailgating_count += 1
                    tailgating_violation = True
                    break

        # 3. Unsafe Lane Change Penalty
        # Action 0: Lane Left, Action 2: Lane Right
        if action_id in [0, 2]:
            target_lane_id = ego_lane_id
            lanes_count = self.env.unwrapped.config.get("lanes_count", 4)
            
            if action_id == 0: target_lane_id = max(0, target_lane_id - 1)
            if action_id == 2: target_lane_id = min(lanes_count - 1, target_lane_id + 1)
            
            for vehicle in self.env.unwrapped.road.vehicles:
                if vehicle is ego_vehicle: continue
                
                # If vehicle is in target lane and within distance
                if vehicle.lane_index[2] == target_lane_id:
                    if abs(vehicle.position[0] - ego_vehicle.position[0]) < self.cfg["min_lane_change_distance"]:
                        penalty = lam * self.cfg["lane_change_penalty"]
                        reward -= penalty
                        reward_components["lane_change_penalty"] = penalty
                        self.unsafe_lane_change_count += 1
                        unsafe_lane_change_violation = True
                        break
            
        # 4. Overspeeding Penalty
        if ego_vehicle.speed > self.cfg["target_speed"]:
            penalty = lam * self.cfg["speed_penalty"]
            reward -= penalty
            reward_components["speed_penalty"] = penalty
            self.overspeed_count += 1
            overspeed_violation = True

        # Add metrics to info
        safety_violation_score = (
            int(collision_violation) * 10
            + int(tailgating_violation)
            + int(unsafe_lane_change_violation)
        )
        info.update({
            "collision": collision_violation,
            "tailgating": tailgating_violation,
            "unsafe_lane_change": unsafe_lane_change_violation,
            "overspeed": overspeed_violation,
            "collision_count": self.collision_count,
            "tailgating_count": self.tailgating_count,
            "unsafe_lane_change_count": self.unsafe_lane_change_count,
            "overspeed_count": self.overspeed_count,
            "total_steps": self.total_steps,
            "speed": float(ego_vehicle.speed),
            "safety_violation_score": safety_violation_score,
            "reward_progress": reward_components["progress_reward"],
            "collision_penalty_cost": reward_components["collision_penalty"],
            "tailgating_penalty_cost": reward_components["tailgating_penalty"],
            "lane_change_penalty_cost": reward_components["lane_change_penalty"],
            "speed_penalty_cost": reward_components["speed_penalty"],
            "reward_components": reward_components,
        })

        return obs, reward, terminated, truncated, info
