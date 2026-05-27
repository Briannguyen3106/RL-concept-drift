import numpy as np
from abc import abstractmethod
from typing import Optional
import pickle
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TD_GAMMA, TD_ALPHA, TD_EPSILON, TD_EPISODE_STARTS
from agents.base_agent import BaseAgent
from environment.drift_env import DriftStreamEnv, ACTIONS

#=============================TD agent=========================
class TDAgent(BaseAgent):
    def __init__(self, n_actions, name):
        super().__init__(n_actions=n_actions, name = name)
        self.gamma = TD_GAMMA
        self.alpha = TD_ALPHA
        self.epsilon = TD_EPSILON

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.get_q_values(state)))
    
    def update(self, state, action, reward, next_state, next_action = None):
        q_current = self.get_q_values(state)[action]
        target = self._compute_target(reward, next_state, next_action)
        self.Q[state][action] += self.alpha * (target - q_current)

    def train(self, env, n_episodes, verbose = True):
        print(f"\n{'='*60}")
        print(f"Training TD Agent: {self.name} for {n_episodes} episodes")
        print(f"Fixed Epsilon: {self.epsilon}")
        print(f"{'='*60}")

        local_history = {
            "episode_rewards": [],
            "episode_lengths": [],
            "action_counts": {k: 0 for k in ACTIONS.values()}
        }

        for ep in range(n_episodes):
            result = self.run_episode(env, ep)
            self.episode_rewards.append(result["total_reward"])
            self.episode_lengths.append(result["n_steps"])
            # 2. Cập nhật dữ liệu ngắn hạn (chỉ trả về cho đợt gọi hàm train này)
            local_history["episode_rewards"].append(result["total_reward"])
            local_history["episode_lengths"].append(result["n_steps"])

            for action, count in result["action_counts"].items():
                self.action_counts[action] += count
                local_history["action_counts"][action] += count

            if verbose:
                # Tính moving average 10 eps dựa trên thuộc tính nội bộ
                avg_reward = np.mean(self.episode_rewards[-10:])
                print(
                    f"Episode {ep+1:>4}/{n_episodes} | "
                    f"Reward: {result['total_reward']:>8.2f} | "
                    f"Avg(10): {avg_reward:>8.2f} | "
                    f"Steps: {result['n_steps']:>4}"
                )
        print(f"\n[{self.name}] Training complete!")
        return local_history

    @abstractmethod
    def _compute_target(self, reward, next_state, next_action):
        pass


    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump({
                'Q'              : self.Q,
                'epsilon'        : self.epsilon,
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths,
                'action_counts'  : self.action_counts,
            }, f)
        print(f'[{self.name}] Saved → {path}')

    @classmethod
    def _load_state(cls, path: str) -> dict:
        """Helper dùng chung cho tất cả agent con khi load."""
        with open(path, 'rb') as f:
            return pickle.load(f)
 
    def _restore(self, data: dict) -> None:
        """Restore state từ dict sau khi load."""
        self.Q               = data['Q']
        self.epsilon         = data['epsilon']
        self.episode_rewards = data['episode_rewards']
        self.episode_lengths = data['episode_lengths']
        self.action_counts   = data['action_counts']
        print(f'[{self.name}] Loaded | Episodes: {len(self.episode_rewards)} '
              f'| States: {len(self.Q)}')

        
#==========================Sarsa=========================================
class SARSAAgent(TDAgent):
    def __init__(self):
        super().__init__(n_actions=5, name = "SARSAAgent")

    def _compute_target(self, reward, next_state, next_action):
        if next_action is None:
            raise ValueError
        q_next = self.get_q_values(next_state)[next_action]
        return reward + self.gamma * q_next
    
    def run_episode(self, env: DriftStreamEnv, ep = 0):
        start_batch = TD_EPISODE_STARTS[ep % len(TD_EPISODE_STARTS)]
        state = env.reset(start_batch=start_batch, mode='train')

        action      = self.select_action(state)
        total_reward= 0.0
        action_counts = {k: 0 for k in ACTIONS.values()}
        info = {}

        while not env.is_done:
            next_state, reward, done, info = env.step(action,current_state=state, update_explorer=False)
            if done:
                break
            
            total_reward += reward
            action_counts[info["action_taken"]] += 1

            next_action = self.select_action(next_state)

            self.update(state, action, reward, next_state, next_action)

            state = next_state
            action = next_action

        return {
            "total_reward": total_reward,
            "n_steps": sum(action_counts.values()),
            "action_counts": action_counts,
            "final_info": info,
        }
    
    @classmethod
    def load(cls, path: str) -> 'SARSAAgent':
        agent = cls()
        agent._restore(cls._load_state(path))
        return agent
    
#==========================Expected Sarsa================================
class ExpectedSARSAAgent(TDAgent):
    def __init__(self):
        super().__init__(n_actions = 5, name = "ExpectedSARSAAgent")

    def _compute_target(self, reward, next_state, next_action = None):
        q_next = self.get_q_values(next_state)
        expected = (self.epsilon/self.n_actions) * np.sum(q_next) + (1-self.epsilon) * np.max(q_next)
        return reward + self.gamma * expected
    
    def run_episode(self, env: DriftStreamEnv, ep = 0):
        start_batch = TD_EPISODE_STARTS[ep % len(TD_EPISODE_STARTS)]
        state = env.reset(start_batch=start_batch, mode='train')

        action      = self.select_action(state)
        total_reward= 0.0
        action_counts = {k: 0 for k in ACTIONS.values()}
        info = {}

        while not env.is_done:
            next_state, reward, done, info = env.step(action, current_state=state, update_explorer=False)
            if done:
                break
            
            total_reward += reward
            action_counts[info["action_taken"]] += 1

            next_action = self.select_action(next_state)

            self.update(state, action, reward, next_state, next_action)

            state = next_state
            action = next_action

        return {
            "total_reward": total_reward,
            "n_steps": sum(action_counts.values()),
            "action_counts": action_counts,
            "final_info": info,
        }
    
    @classmethod
    def load(cls, path: str) -> 'ExpectedSARSAAgent':
        agent = cls()
        agent._restore(cls._load_state(path))
        return agent
    
#===============================Q_Learning==============================
class QLearningAgent(TDAgent):
    def __init__(self):
        super().__init__(n_actions = 5, name = "QLearningAgent")
    
    def _compute_target(self, reward, next_state, next_action=None):
        return reward + self.gamma * np.max(self.get_q_values(next_state))
    
    def run_episode(self, env: DriftStreamEnv, ep = 0):
        start_batch  = TD_EPISODE_STARTS[ep % len(TD_EPISODE_STARTS)]
        state        = env.reset(start_batch=start_batch, mode="train")
        action       = self.select_action(state)
        total_reward = 0.0
        action_counts = {k: 0 for k in ACTIONS.values()}
        info          = {}

        while not env.is_done:
            next_state, reward, done, info = env.step(action=action, current_state=state, update_explorer=False)
            if done:
                break

            total_reward += reward
            action_counts[info["action_taken"]] += 1

            self.update(state, action, reward, next_state)
            next_action = self.select_action(next_state)
            state = next_state
            action = next_action

        return {
            "total_reward" : total_reward,
            "n_steps"      : sum(action_counts.values()),
            "action_counts": action_counts,
            "final_info"   : info,
        }
    
    @classmethod
    def load(cls, path: str) -> 'QLearningAgent':
        agent = cls()
        agent._restore(cls._load_state(path))
        return agent
    
class DoubleQLearningAgent(TDAgent):
    def __init__(self):
        super().__init__(n_actions=5, name='DoubleQLearningAgent')
        self.Q2 = {}
    
    def get_q_values_2(self, state):
        if state not in self.Q2:
            self.Q2[state] = np.zeros(self.n_actions)
        return self.Q2[state]
    
    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        q_avg = self.get_q_values(state) + self.get_q_values_2(state)
        return int(np.argmax(q_avg))
    
    def _compute_target(self, reward, next_state, next_action):
        return super()._compute_target(reward, next_state, next_action)

    def update(self, state, action, reward, next_state, next_action=None):
        if np.random.random() <0.5:
            a_star = int(np.argmax(self.get_q_values(next_state)))
            target = reward + self.gamma * self.get_q_values_2(next_state)[a_star]
            q_current = self.get_q_values(state)[action]
            self.Q[state][action] += self.alpha*(target - q_current)
        else:
            a_star = int(np.argmax(self.get_q_values_2(next_state)))
            target = reward + self.gamma * self.get_q_values(next_state)[a_star]
            q_current = self.get_q_values_2(state)[action]
            self.Q2[state][action] += self.alpha * (target - q_current)

    def run_episode(self, env: DriftStreamEnv, ep=0):
        start_batch  = TD_EPISODE_STARTS[ep % len(TD_EPISODE_STARTS)]
        state        = env.reset(start_batch=start_batch, mode="train")
        action       = self.select_action(state)
        total_reward = 0.0
        action_counts = {k: 0 for k in ACTIONS.values()}
        info          = {}

        while not env.is_done:
            next_state, reward, done, info = env.step(action=action, current_state=state, update_explorer=False)
            if done:
                break

            total_reward += reward
            action_counts[info["action_taken"]] += 1

            self.update(state, action, reward, next_state)
            next_action = self.select_action(next_state)
            state = next_state
            action = next_action

        return {
            "total_reward" : total_reward,
            "n_steps"      : sum(action_counts.values()),
            "action_counts": action_counts,
            "final_info"   : info,
        }
    
    def reset(self) -> None:
        """Reset cả Q1 và Q2."""
        super().reset()
        self.Q2.clear()

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump({
                'Q'              : self.Q,
                'Q2'             : self.Q2,
                'epsilon'        : self.epsilon,
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths,
                'action_counts'  : self.action_counts,
            }, f)
        print(f'[{self.name}] Saved → {path}')
 
    @classmethod
    def load(cls, path: str) -> 'DoubleQLearningAgent':
        agent = cls()
        data  = cls._load_state(path)
        agent._restore(data)
        agent.Q2 = data['Q2']
        return agent
    
# ================================================================== #
#  QUICK TEST                                                          #
# ================================================================== #
 
if __name__ == "__main__":
    from utils.data_loader import AirlinesDataLoader
    from models.LightGBM import LightGBM
    from metrics.stream_metrics import StreamMetrics
    from environment.drift_env import DriftStreamEnv
 
    loader  = AirlinesDataLoader()
    clf     = LightGBM()
    metrics = StreamMetrics()
    env     = DriftStreamEnv(loader, clf, metrics)
 
    agents = [
        SARSAAgent(),
        ExpectedSARSAAgent(),
        QLearningAgent(),
        DoubleQLearningAgent(),
    ]
 
    for agent in agents:
        print(f"\n{'='*50}")
        print(f"Testing {agent.name}")
        history = agent.train(env, n_episodes=3)
        print(f"  States visited : {len(agent.Q)}")
        print(f"  Avg reward     : {np.mean(history['episode_rewards']):.4f}")
        print(f"  Action counts  : {history['action_counts']}")
        




        
    



    

