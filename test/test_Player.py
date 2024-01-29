from unittest import TestCase
from unittest.mock import Mock

import numpy as np
import tensorflow as tf

from src.models.BidActions import BidActions
from src.models.BidInput import BidInput
from src.models.PlayActions import PlayActions
from src.models.PlayInput import PlayInput
from src.Player import Player


class TestPlayer(TestCase):

    def test_make_bid(self):
        # Mocks and initial setup
        mock_bidding_model = Mock()

        # Initialize Player object
        player = Player(0, mock_bidding_model, Mock())

        # Prepare BidInput object
        hand = ['9S', 'TS', 'JS', 'QS', 'KS', 'AS']
        dealer_position = 3
        score = [0, 0]

        # First bid, may bid 0
        previous_bids = [None, None, None, None]
        bid_input = BidInput(hand, previous_bids, dealer_position, score)
        mock_bidding_model.predict.return_value = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]]
        bid_value, bid_suit = player.make_bid(bid_input)
        self.assertEqual(0, bid_value)

        # Last bid, may not bid '0'
        previous_bids = [0, 0, 0, None]
        bid_input = BidInput(hand, previous_bids, dealer_position, score)
        mock_bidding_model.predict.return_value = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]]
        bid_value, bid_suit = player.make_bid(bid_input)
        self.assertEquals(4, bid_value)

        # Overbids other players
        previous_bids = [4, None, None, None]
        bid_input = BidInput(hand, previous_bids, dealer_position, score)
        mock_bidding_model.predict.return_value = [[0, 1, 0.5, 0, 0, 0, 0, 0, 0, 0, ]]
        bid_value, bid_suit = player.make_bid(bid_input)
        self.assertEqual(5, bid_value)

        # May only bid '0' when someone has 'pfeffered'
        previous_bids = [0, 0, 'pfeffer', None]
        bid_input = BidInput(hand, previous_bids, dealer_position, score)
        mock_bidding_model.predict.return_value = [[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, ]]
        bid_value, bid_suit = player.make_bid(bid_input)
        self.assertEqual(0, bid_value)

    def test_make_play(self):
        # Mocks and initial setup
        mock_play_model = Mock()

        # Initialize Player object
        player = Player(0, Mock(), mock_play_model)

        # Prepare PlayInput object
        player_id = 0
        hand = ['TS', 'JS', 'QS', 'KS', 'AS']
        played_cards = [
            [(0, '9S'), (1, 'TC'), (2, '9D'), (3, 'TD')],
            [(2, 'JC'), (3, 'QC')],
            [],
            [],
            [],
            [],
        ]
        bidding_order = [0, 1, 2, 3]
        all_bids = [4, 5, 0, 0]
        winning_bid = (5, 1, 'D')
        lead_players = [0, 1, None, None, None, None]
        trick_winners = [0, 1, None, None, None, None]
        current_trick = 1
        score = [0, 0]
        play_input = PlayInput(player_id, hand, played_cards, bidding_order, all_bids, winning_bid, lead_players,
                               trick_winners, current_trick, score)

        mock_return_value = [np.zeros(24)]
        mock_play_model.predict.return_value = mock_return_value
        card_played = player.make_play(play_input)
        self.assertEqual('TS', card_played)

        play_input.winning_bid = (5, 1, 'NT')
        card_played = player.make_play(play_input)
        self.assertEqual('TS', card_played)

    def test_save_to_bid_buffer(self):
        # Mocks and initial setup
        mock_bidding_model = Mock()
        mock_play_model = Mock()

        # Initialize Player object
        player = Player(0, mock_bidding_model, mock_play_model)

        # Prepare BidInput object
        hand = ['9S', 'TS', 'JS', 'QS', 'KS', 'AS']
        previous_bids = [4, 5, 0, ]
        dealer_position = 3
        score = [0, 0]
        bid_input = BidInput(hand, previous_bids, dealer_position, score)

        # Prepare action taken and reward received
        action_taken = (5, 'S')  # Bid value and Suit
        reward_received = 10.0  # Example reward

        # Run the method to be tested
        player.save_to_bid_buffer(bid_input, action_taken, reward_received)

        # Verify that the bid_replay_buffer has one more item
        self.assertEqual(player.bid_replay_buffer.num_frames(), 1)

        # Retrieve the item from the replay buffer and convert to numpy arrays
        dataset = player.bid_replay_buffer.as_dataset(sample_batch_size=1, num_steps=1)
        iterator = iter(dataset)
        first_item = next(iterator)
        first_item_numpy = tf.nest.map_structure(lambda t: t.numpy(), first_item)
        # print("First item from replay buffer:", first_item_numpy)

        # Extract the trajectory and observation from the first item
        trajectory = first_item_numpy[0]
        cached_observation = trajectory.observation
        cached_action = trajectory.action[0][0]

        # Decode hand
        decoded_hand = BidInput.decode_hand(np.array(cached_observation['hand'][0][0]))

        # Decode previous bids
        decoded_previous_bids = []
        for i in range(0, len(cached_observation['previous_bids'][0][0]), 5):
            bid_encoded = cached_observation['previous_bids'][0][0][i:i + 5]
            bid = BidInput.decode_bid(np.array(bid_encoded))
            decoded_previous_bids.append(bid)

        # Decode dealer_position and score
        decoded_dealer_position = cached_observation['dealer_position'][0][0][0]
        decoded_score = cached_observation['score'][0][0]

        # Check if hand is saved correctly
        self.assertEqual(decoded_hand, hand)

        # Check if previous_bids are saved correctly
        self.assertEqual(decoded_previous_bids, previous_bids)

        # Check if dealer_position is saved correctly
        self.assertEqual(decoded_dealer_position, dealer_position)

        # Check if score is saved correctly
        self.assertEqual(decoded_score.tolist(), score)

        # Check if action is saved correctly
        bid_value_encoded = cached_action[0]
        bid_suit_encoded = cached_action[1]
        decoded_bid_value = BidActions.get_bid_value_action(bid_value_encoded)
        decoded_bid_suit = BidActions.get_bid_suit_action(bid_suit_encoded)
        decoded_action = (decoded_bid_value, decoded_bid_suit)
        self.assertEqual(decoded_action, action_taken)

        # Check if reward is saved correctly
        self.assertEqual(trajectory.reward[0][0], reward_received)

    def test_save_to_play_buffer(self):
        # Mocks and initial setup
        mock_bidding_model = Mock()
        mock_play_model = Mock()

        # Initialize Player object
        player = Player(0, mock_bidding_model, mock_play_model)

        # Initialize PlayInput object
        player_id = 0
        hand = ['9S', 'TS', 'JS', 'QS', 'KS', 'AS']
        played_cards = [
            [('9S', 0), ('TC', 1), ('9D', 2), ('TD', 3)],
            [('JC', 2), ('QC', 3)],
            [],
            [],
            [],
            [],
        ]
        bidding_order = [0, 1, 2, 3]
        all_bids = [4, 5, 0, 0]
        winning_bid = (5, 1, 'S')
        lead_players = [0, 1, None, None, None, None]
        trick_winners = [0, 1, None, None, None, None]
        current_trick = 1
        score = [0, 0]
        play_input = PlayInput(player_id, hand, played_cards, bidding_order, all_bids, winning_bid, lead_players,
                               trick_winners, current_trick, score)

        # Prepare action taken and reward received
        action_taken = 'QS'  # Example action taken (you'll need to specify this based on your game logic)
        reward_received = 10.0  # Example reward

        # Run the method to be tested
        player.save_to_play_buffer(play_input, action_taken, reward_received)

        # Verify that the play_replay_buffer has one more item
        self.assertEqual(player.play_replay_buffer.num_frames(), 1)

        # Retrieve the item from the replay buffer and convert to numpy arrays
        dataset = player.play_replay_buffer.as_dataset(sample_batch_size=1, num_steps=1)
        iterator = iter(dataset)
        first_item = next(iterator)
        first_item_numpy = tf.nest.map_structure(lambda t: t.numpy(), first_item)

        # Extract the trajectory and observation from the first item
        trajectory = first_item_numpy[0]
        cached_observation = trajectory.observation
        cached_action = trajectory.action[0][0][0]

        # Check if play state is saved correctly
        denested_observation = denest_arrays(cached_observation)
        decoded_play_input = PlayInput.decode(denested_observation)
        self.assertEqual(decoded_play_input, play_input)

        # Check if action is saved correctly
        decoded_action = PlayActions.get_action(cached_action)
        self.assertEqual(action_taken, decoded_action)

        # Check if reward is saved correctly
        self.assertEqual(reward_received, trajectory.reward[0][0])


def denest_arrays(dictionary):
    """
    This function takes in a dictionary with numpy arrays and outputs the dictionary with all numpy arrays denested.
    :param dictionary: Dict with numpy arrays
    :return: Dictionary with denest numpy arrays
    """
    for key in dictionary.keys():
        dictionary[key] = np.squeeze(dictionary[key])
    return dictionary
