import copy
import time
import ray
import torch
import net
import numpy as np


@ray.remote
class   ReplayBuffer:
    """
    class which run in a dedicated thread to store played games and generate batch.
    """
    def __init__(self, initial_checkpoint, initial_buffer, config):
        self.config = config
        self.buffer = copy.deepcopy(initial_buffer)
        self.num_played_games = initial_checkpoint["num_played_games"]
        self.num_played_steps = initial_buffer["num_played_steps"]
        self.total_samples = sum(
            [len(game_history.root_values) for game_history in self.buffer.values()]
        )
        if self.total_samples != 0:
            print(
                f"Replay buffer initialized with {self.total_samples} samples ({self.num_played_games} games).\n"
            )
        np.random.seed(self.config.seed)
    
    def save_game(self, game_history, shared_storage=None):
        self.buffer[self.num_played_games] = game_history
        self.num_played_games += 1
        self.num_played_steps += len(game_history.root_values)
        self.total_samples += len(game_history.root_values)

        if self.config.replay_buffer_size < len(self.buffer):
            del_id = self.num_played_games - len(self.buffer)
            self.total_samples -= len(self.buffer[del_id].root_values)
            del self.buffer[del_id]
        
        if shared_storage:
            shared_storage.set_info.remote("num_played_games", self.num_played_games)
            shared_storage.set_info.remote("num_played_steps", self.num_played_steps)
        
    def get_buffer(self):
        return self.buffer
    
    def get_batch(self):
        (
            index_batch,
            observation_batch,
            action_batch,
            reward_batch,
            value_batch,
            policy_batch,
            gradient_scale_batch,   # what?
        ) = ([], [], [], [], [], [], [])

        for game_id, game_history, game_prob in self.sample_n_games(self.config.batch_size):
            game_pos, pos_prob = self.sample_position(game_history)

            values, rewards, polices, actions= self.make_target(
                game_history, game_pos
            )

            index_batch.append([game_id, game_pos])
            observation_batch.append(
                game_history.get_stacked_observations(
                    game_pos, self.config.stacked_observations
                )
            )
            action_batch.append(actions)
            value_batch.append(values)
            reward_batch.append(rewards)
            policy_batch.append(polices)
            gradient_scale_batch.append(
                [
                    min(
                        self.config.num_unroll_steps,
                        len(game_history.action_history) - game_pos
                    )
                ]
                * len(actions)
            )
        return (
            index_batch,
            (
                observation_batch,
                action_batch,
                value_batch,
                reward_batch,
                policy_batch,
                gradient_scale_batch,
            ),
        )

    def sample_game(self):
        game_index = np.random.choice(len(self.buffer))
        game_id = self.num_played_games - len(self.buffer) + game_index
        
        return game_id, self.buffer[game_id]
    
    def sample_n_games(self, n_games):
        selected_games = np.random.choice(list(self.buffer.keys()), n_games)
        ret = [(game_id, self.buffer.game_id)
                for game_id in selected_games]
        return ret

    def sample_position(self, game_history):
        
        position_index = np.random.choice(len(game_history.root_values))
        return position_index
    
    def update_game_history(self, game_id, game_history):
        
        if next(iter(self.buffer)) <= game_id:
            self.buffer[game_id] = game_history


@ray.remote
class Reanalyse:
    """
    class which run in a dedicated thread to update the replay buffer with fresh information in MuZero
    """
    def __init__(self, initial_checkpoint, config) -> None:
        self.config = config

        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        self.model = models.WormZeroNetwork(self.config)
        self.model.set_weights(initial_checkpoint["weights"])
        self.model.to(torch.device("cuda" if self.config.reanalyse_on_gpu else "cpu"))
        self.model.eval()

    def reanalyse(self, replay_buffer, shared_storage):
        while ray.get(shared_storage.get_info.remote("num_played_games")) < 1:
            time.sleep(0.1)
        
        while ray.get(shared_storage.get_info.remote("training_step")) < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            self.model.set_weighs(ray.get(shared_storage.get_info.remote("weights")))

            game_id, game_history, _ = ray.get(
                replay_buffer.sample_game.remote()
            )

            if self.config.use_last_model_value:
                observations = [
                    observations.get_stacked_observations(
                        i, self.config.stacked_observatins
                    )
                    for i in range(len(game_history.root_values))
                ]

                observations = (
                    torch.tensor(observations)
                    .float()
                    .to(next(self.model.parameters()).device)
                )
                #TODO, not sure
                # values = 
                # game_history.reanalysed_predicted_root_values = ()
            replay_buffer.update_game_history.remote(game_id, game_history)
            self.num_reanalysed_games += 1
            shared_storage.set_info.remote(
                "num_reanalysed_games", self.num_reanalysed_games
            )