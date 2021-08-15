import math
import time

import numpy as np
import ray
import torch

import net
from utils import GameHistory
from mcts import MCTS


@ray.remote
class SelfPlay:
    """
    class which run in a dedicated thread to play games and save them to the replay-buffer.
    """
    def __init__(self, initial_checkpoint, Game,config, seed):
        self.config = config
        self.game = Game(seed)
    
        np.random.choice(seed)
        torch.manual_seed(seed)

        self.model = net.WormZeroNetwork(self.config)
        self.model.set_weights(initial_checkpoint["weights"])
        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.eval()

    def continuous_self_play(self, shared_storage, replay_buffer, test_mode=False):
        while ray.get(shared_storage.get_info.remote("training_step")) < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            self.model.set_weighs(ray.get(shared_storage.get_info.remote("weights")))

            if not test_mode:
                game_history = self.play_game(
                    self.config.visit_softmax_temperature_fn(
                        trained_steps=ray.get(shared_storage.get_info.remote("training_step"))
                    ),
                    self.config.temperature_threshold,
                    False,
                    "self",
                    0,
                )
                replay_buffer.save_game.remote(game_history, shared_storage)
            
            else:
                game_history = self.play_game(
                    0,
                    self.config.temperature_threshold,
                    False,
                    "self" if len(self.config.players) == 1 else self.config.opponent,
                    self.config.muzero_player,
                )

                shared_storage.set_info_remote(
                    {
                        "episode_length": len(game_history.action_history) - 1,
                        "total_reward": sum(game_history.reward_history),
                        "mean_value": np.mean(
                            [value for value in game_history.root_values if value]
                        ),
                    }
                )
                if  1 < len(self.config.players):
                    shared_storage.set_info.remote(
                        {
                            "wormzero_reward": sum(
                                reward
                                for i, reward in enumerate(game_history.reward_history)
                                if game_history.to_play_history[i - 1]
                                == self.config.wormzero_player
                            ),
                            "opponent_reward": sum(
                                reward 
                                for i, reward in enumerate(game_history.reward_history)
                                if game_history.to_play_history[i - 1]
                                != self.config.wormzero_player
                            ),
                        }
                    )
                
            # Managing the self-play / training ratio
            if not test_mode and self.config.self_play_delay:
                time.sleep(self.config.self_play_delay)
            if not test_mode and self.config.ratio:
                while(
                    ray.get(shared_storage.get_info.remote("tarining_step"))
                    / max(
                        1, ray.get(shared_storage.get_info.remote("num_played_steps"))
                    )
                    < self.config.ratio
                    and ray.get(shared_storage.get_info.remote("training_step"))
                    < self.config.training_steps
                    and not ray.get(shared_storage.get_info.remote("terminate"))
                ):
                    time.sleep(0.5)
        
        self.close_game()

    def play_game(self, temperature, temperature_threshold, render, opponent, wormzero_player):
        """
        play one game with actions based on the Monte Carlo tree search at each moves.
        """
        game_history = GameHistory()
        observation = self.game.reset()
        game_history.action_history.append(0)
        game_history.observation_history.append(observation)
        game_history.reward_history.append(0)
        game_history.to_play_history.append(self.game.to_play())

        done = False

        if render:
            self.game.render()
        
        with torch.no_grad():
            while(
                not done and len(game_history.action_history) < self.config.max_moves
            ):
                assert(
                    len(np.array(observation).shape) == 3
                ), f"Observation should be 3 dimensional instead of {len(np.array(observation).shape)} dimensional. Got observation of shape: {np.array(observation).shape}"
                #TODO
                stacked_observations = game_history.get_stacked_observations(
                    -1,
                    self.config.stacked_observations,
                )

                if opponent == "self" or wormzero_player == self.game.to_play():
                    root, mcts_info = MCTS(self.config).run(
                        self.model,
                        stacked_observations,
                        self.game.legal_actions(),
                        self.game.to_play(),
                        True,
                    )
                    action = self.select_action(
                        root,
                        temperature
                        if not temperature_threshold 
                        or len(game_history.action_history) < temperature_threshold
                        else 0,
                    )

                    if render:
                        print(f'Tree depth: {mcts_info["max_tree_depth"]}')
                        print(
                            f"Root value for player {self.game.to_play()}: {root.value():.2f}"
                        )
                
                else:
                    action, root = self.select_opponent_action(
                        opponent, stacked_observations
                    )
                
                observation, reward, done = self.game.step(action)

                if render:
                    print(f"Played action: {self.game.action_to_string(action)}")
                    self.game.render()
                
                game_history.store_search_statistics(root, self.config.action_space)

                game_history.action_history.append(action)
                game_history.observation_history.append(observation)
                game_history.reward_history.append(reward)
                game_history.to_play_history.append(self.game.to_play())

        return game_history
    
    def close_game(self):
        self.game.close()
    
    def select_opponent_action(self, opponent, stacked_observations):
        """
        select opponent action for evaluating WormZero level.
        """
        if opponent == "human":
            root, mcts_info = MCTS(self.config).run(
                self.model,
                stacked_observations,
                self.game.legal_actions(),
                self.game.to_play(),
                True
            )
            print(f'Tree depth: {mcts_info["max_tree_depth"]}')
            print(f"Root value for player {self.game.to_play()}: {root.value():.2f}")
            print(
                f"Player {self.game.to_play()} turn. WormZero suggests {self.game.action_to_string(self.select_action(root, 0))}"
            )
            return self.game.human_to_action(), root
        elif opponent == "expert":
            return self.game.expert_agent(), None
        elif opponent == "random":
            assert (
                self.game.legal_actions()
            ), f"Legal actions should not be an empty array. Got {self.game.legal_actions()}."
            assert set(self.game.legal_actions()).issubset(
                set(self.config.action_space)
            ), "Legal actions should be a subset of the action space."

            return np.random.choice(self.game.legal_actions()), None
        elif opponent == "negamax":
            return self.game.negamax_agent()
        else:
            raise NotImplementedError(
                'Wrong argument: "opponent" argument should be "self", "human", "expert" or "random"'
            )

    #TODO select action according to \bar{\pi}, use search Q values
    @staticmethod
    def select_action(node, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        The temperature is changed dynamically with the visit_softmax_temperature function
        in the config.
        """
        visit_counts = np.array(
            [child.visit_count for child in node.children.values()], dtype="int32"
        )
        actions = [action for action in node.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            visit_count_distribution = visit_counts**(1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            action = np.random.choice(actions, p = visit_count_distribution)
        return action
                    
