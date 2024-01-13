import numpy as np
import numpy.ma as ma
import tensorflow as tf
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import trajectory

from src.models import SUITS_COLORS, CARDS, BID_VALUES
from src.models.BidActions import BidActions
from src.models.BidInput import BidInput
from src.models.PlayActions import PlayActions


class Player:
    def __init__(self, player_id, bid_model, play_model):
        self.player_id = player_id
        self.bid_model = bid_model
        self.play_model = play_model

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
            'bidding_order': tf.TensorSpec(shape=(16,), dtype=tf.int32),
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
        # Flatten and concatenate encoded states
        flattened_encoded_state = tf.concat([tf.reshape(tensor, [-1]) for tensor in encoded_state.values()], 0)
        # Reshape tensor to meet model's input requirement
        flattened_encoded_state = tf.reshape(flattened_encoded_state, (-1, 42))

        # Get the Q-values from the bidding model
        q_values = self.bid_model.predict(flattened_encoded_state)[0]

        # Create a mask based on the highest previous bid
        mask = np.ones(10)  # 10 is the total number of actions (5 bids + 5 suits)
        if all(elem == 0 for elem in bid_input.previous_bids):
            # Dealer is "hung", must bid something either than 0
            mask[0] = 0  # Mask out illegal bid
        else:
            # Must bid '0', or bid higher than the highest bid
            highest_previous_bid = max((bid for bid in bid_input.previous_bids if bid is not None),
                                       key=BID_VALUES.index, default=0)
            mask[1:BidActions.get_bid_value_index(highest_previous_bid) + 1] = 0  # Mask out illegal bids

        # The action (bid) with the highest Q-value is the best bid
        masked_q_values = ma.masked_array(q_values[:5], mask=np.logical_not(mask[:5]))
        bid_value_index = np.argmax(masked_q_values)
        bid_value = BidActions.get_bid_value_action(bid_value_index)

        # The action (suit) with the highest Q-value is the best suit to choose as trump
        masked_q_values = ma.masked_array(q_values[5:], mask=np.logical_not(mask[5:]))
        bid_suit_index = np.argmax(masked_q_values)
        bid_suit = BidActions.get_bid_suit_action(bid_suit_index)

        return bid_value, bid_suit

    def make_play(self, play_input):
        """
        Makes a play by choosing a card using the Q-network.
        Removes the card from the player's hand.

        Args:
            play_input (PlayInput): The game state as known by the current player.

        Returns:
            action (PlayInput): The chosen action (card to play).
        """
        # Get current state of the game and encode it
        encoded_state = play_input.encode()
        # Flatten and concatenate encoded states
        flattened_encoded_state = tf.concat([tf.reshape(tensor, [-1]) for tensor in encoded_state.values()], 0)
        # Reshape tensor to meet model's input requirement
        flattened_encoded_state = tf.reshape(flattened_encoded_state, (-1, 722))

        # Get Q-values for the current state
        q_values = self.play_model.predict(flattened_encoded_state)[0]

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
                if (card[-1] == lead_suit or
                        # card == right_bauer or
                        (card == left_bauer and lead_suit == trump_suit)):
                    mask_suit.append(1 if card in play_input.hand else 0)
                else:
                    mask_suit.append(0)

            if any(val == 1 for val in mask_suit):  # If any card can be played, update the mask
                mask = mask_suit

        masked_q_values = ma.masked_array(q_values, mask=np.logical_not(mask))
        play_value_index = np.argmax(masked_q_values)
        card_played = PlayActions.get_action(play_value_index)

        # Log play
        print(f"Player {self.player_id} played {card_played}")
        if card_played not in play_input.hand:
            print(f"Invalid play\n"
                  f"Played Cards: {play_input.played_cards}\n"
                  f"Hand: {play_input.hand}\n"
                  f"Mask: {mask}\n"
                  f"q_values: {q_values}\n"
                  f"masked_q_values: {masked_q_values}\n"
                  f"masked_q_values: {masked_q_values}\n"
                  f"play_value_index: {play_value_index}\n"
                  f"card_played: {card_played}")

        return card_played

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
