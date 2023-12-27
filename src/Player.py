import numpy as np
import tensorflow as tf
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import trajectory

from src.models.BidActions import BidActions
from src.models.PlayActions import PlayActions
from src.models import SUITS_COLORS, CARDS


class Player:
    def __init__(self, player_id, bidding_model, play_model, bid_actions, play_actions):
        self.player_id = player_id
        self.bidding_model = bidding_model
        self.play_model = play_model
        self.bid_actions = bid_actions
        self.play_actions = play_actions

        # Create bid data spec
        bid_state_spec = {
            'hand': tf.TensorSpec(shape=(24,), dtype=tf.int32),
            'dealer_position': tf.TensorSpec(shape=1, dtype=tf.int32),
            'score': tf.TensorSpec(shape=(2,), dtype=tf.int32),
            'previous_bids': tf.TensorSpec(shape=(15,), dtype=tf.int32),
        }
        bid_action_spec = tensor_spec.BoundedTensorSpec(shape=(2,), dtype=tf.int32, minimum=0, maximum=4)
        bid_data_spec = trajectory.Trajectory(
            observation=bid_state_spec,
            action=bid_action_spec,
            policy_info=(),
            reward=tf.TensorSpec(shape=(), dtype=tf.float32),
            discount=tf.TensorSpec(shape=(), dtype=tf.float32),
            step_type=tf.TensorSpec(shape=(), dtype=tf.int32),
            next_step_type=tf.TensorSpec(shape=(), dtype=tf.int32),
        )

        # Create play data spec
        play_state_spec = {
            'player_id': tf.TensorSpec(shape=(4,), dtype=tf.int32),
            'hand': tf.TensorSpec(shape=(24,), dtype=tf.int32),
            'played_cards': tf.TensorSpec(shape=(6 * 4 * 24,), dtype=tf.int32),
            'bidding_order': tf.TensorSpec(shape=(4, 4), dtype=tf.int32),
            'all_bids': tf.TensorSpec(shape=(20,), dtype=tf.int32),
            'winning_bid': tf.TensorSpec(shape=(14,), dtype=tf.int32),
            'lead_players': tf.TensorSpec(shape=(30,), dtype=tf.int32),
            'trick_winners': tf.TensorSpec(shape=(30,), dtype=tf.int32),
            'current_trick': tf.TensorSpec(shape=(6,), dtype=tf.int32),
            'score': tf.TensorSpec(shape=(2,), dtype=tf.int32),
        }
        play_action_spec = tensor_spec.BoundedTensorSpec(shape=(1,), dtype=tf.int32, minimum=0, maximum=23)
        play_data_spec = trajectory.Trajectory(
            observation=play_state_spec,
            action=play_action_spec,
            policy_info=(),
            reward=tf.TensorSpec(shape=(), dtype=tf.float32),
            discount=tf.TensorSpec(shape=(), dtype=tf.float32),
            step_type=tf.TensorSpec(shape=(), dtype=tf.int32),
            next_step_type=tf.TensorSpec(shape=(), dtype=tf.int32),
        )

        # Initialize replay buffers
        max_length = 1_000
        self.bid_replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            bid_data_spec, batch_size=1, max_length=max_length
        )
        self.play_replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            play_data_spec, batch_size=1, max_length=max_length
        )

    def make_bid(self, bid_input):
        """
        Makes a bid and chooses a trump suit based on the current game state.

        Args:
            bid_input (BidInput): The game state as known by the current player.

        Returns:
            tuple: The chosen bid and trump suit.
        """

        # Get the encoded state as a dictionary
        encoded_state = bid_input.encode()

        # Get the Q-values from the bidding model
        q_values = self.bidding_model.predict(encoded_state)

        # Create a mask based on the highest previous bid
        highest_previous_bid = max(bid_input.previous_bids, default=0)
        mask = np.ones(11)  # 11 is the total number of actions (6 bids + 5 suits)
        mask[1:highest_previous_bid + 1] = 0  # Mask out illegal bids

        # Apply the mask to modify the Q-values
        masked_q_values = q_values * mask + np.invert(mask.astype(bool)) * -np.inf

        # The action (bid) with the highest Q-value is the best bid
        bid_value_index = np.argmax(masked_q_values[:6])
        bid_value = BidActions.get_bid_value_action(bid_value_index)

        # The action (suit) with the highest Q-value is the best suit to choose as trump
        bid_suit_index = np.argmax(masked_q_values[6:])
        bid_suit = BidActions.get_bid_suit_action(bid_suit_index)

        return bid_value, bid_suit

    def make_play(self, play_input):
        """
        Makes a play by choosing a card using the Q-network.

        Args:
            play_input (PlayInput): The game state as known by the current player.

        Returns:
            action (PlayInput): The chosen action (card to play).
        """
        # Get current state of the game and encode it
        state_vector = play_input.encode()

        # Get Q-values for the current state
        # q_values = self.play_model.predict(state_vector[None, :], verbose=0)[0]
        q_values = self.play_model.predict(x=state_vector, verbose=0)

        # Mask out illegal actions (cards not in hand)
        mask = [1 if card in play_input.hand else 0 for card in CARDS]

        # If it's not the first card in the trick, it must follow suit if possible
        if play_input.played_cards[-1]:
            lead_suit = play_input.played_cards[-1][0][1][-1]
            trump_suit = play_input.winning_bid[2]

            # Define the left bauer based on trump suit
            opposite_color_suits = {"black": ["S", "C"], "red": ["H", "D"]}
            left_bauer_suit = [s for s in opposite_color_suits[SUITS_COLORS[trump_suit]] if s != trump_suit][0]
            right_bauer = 'J' + trump_suit
            left_bauer = 'J' + left_bauer_suit

            # Modify mask for lead suit, right bauer, and left bauer
            mask_suit = []
            for card in CARDS:
                if card[-1] == lead_suit or card == right_bauer or card == left_bauer:
                    mask_suit.append(1 if card in play_input.hand else 0)
                else:
                    mask_suit.append(0)

            if any(val == 1 for val in mask_suit):  # If any card can be played, update the mask
                mask = mask_suit

        q_values = q_values * np.array(mask) + np.invert(mask) * -np.inf

        # Choose action with highest Q-value
        action_index = np.argmax(q_values)
        action = self.play_actions.get_action(action_index)

        return action

    def save_to_bid_buffer(self, bid_input, action_taken, reward_received):
        # Encode the current bidding state
        bid_input_encoded = bid_input.encode()

        bid_value = action_taken[0]
        bid_suit = action_taken[1]
        bid_value_encoded = BidActions.get_bid_value_index(bid_value)
        bid_suit_encoded = BidActions.get_bid_suit_index(bid_suit)
        action_taken_encoded = tf.constant([bid_value_encoded, bid_suit_encoded], dtype=tf.int32)

        # Create a trajectory with the experience
        experience = trajectory.Trajectory(
            step_type=tf.expand_dims(tf.constant(1, dtype=tf.int32), axis=0),
            observation={key: tf.expand_dims(tf.constant(value), axis=0) for key, value in bid_input_encoded.items()},
            action=tf.expand_dims(action_taken_encoded, axis=0),
            policy_info=(),
            next_step_type=tf.expand_dims(tf.constant(1, dtype=tf.int32), axis=0),
            reward=tf.constant([reward_received], dtype=tf.float32),
            discount=tf.constant([1.0], dtype=tf.float32),
        )

        # Add the experience to the bid replay buffer
        self.bid_replay_buffer.add_batch(experience)

    def save_to_play_buffer(self, play_input, action_taken, reward_received):
        # Encode the current playing state
        play_input_encoded = play_input.encode()
        action_taken_encoded = PlayActions.get_index(action_taken)

        step_type = [0, 1, 1, 1, 1, 2][play_input.current_trick]
        next_step_type = [1, 1, 1, 1, 2, 0][play_input.current_trick]

        # Create a trajectory with the experience
        experience = trajectory.Trajectory(
            step_type=tf.expand_dims(tf.constant(step_type, dtype=tf.int32), axis=0),
            observation={key: tf.expand_dims(tf.constant(value), axis=0) for key, value in play_input_encoded.items()},
            action=tf.expand_dims(tf.constant([action_taken_encoded], dtype=tf.int32), axis=0),
            policy_info=(),
            next_step_type=tf.expand_dims(tf.constant(next_step_type, dtype=tf.int32), axis=0),
            reward=tf.constant([reward_received], dtype=tf.float32),
            discount=tf.constant([1.0], dtype=tf.float32),
        )

        # Add the experience to the play replay buffer
        self.play_replay_buffer.add_batch(experience)
