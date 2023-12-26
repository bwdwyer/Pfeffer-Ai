from unittest import TestCase
from unittest.mock import Mock

from src.pfeffer_simulation import *


# noinspection DuplicatedCode
class TestGame(TestCase):

    def test_reset(self):
        # Initialize models for bidding and playing (replace with actual models if needed)
        bid_model = None
        play_model = None

        # Initialize a game
        game = Game(bid_model, play_model)

        # Modify the game state to simulate a game in progress
        game.game_state["score"] = [15, 20]
        game.game_state["played_cards"][0] = [('9S', 0), ('9H', 1), ('9D', 2), ('JC', 3)]
        game.game_state["all_bids"] = [4, 5, 0, 'pfeffer']
        game.game_state["winning_bid"] = ('pfeffer', 3, 'C')
        game.game_state["lead_players"][0] = 0
        game.game_state["trick_winners"][0] = 3

        # Call the reset method
        game.reset()

        # Verify that the game state has been reset
        assert [0, 0] == game.game_state["score"]
        assert all(len(trick) == 0 for trick in game.game_state["played_cards"])
        assert [] == game.game_state["all_bids"]
        assert (-1, -1, None) == game.game_state["winning_bid"]
        assert [None] * 6 == game.game_state["lead_players"], game.game_state["lead_players"]
        assert [None] * 6 == game.game_state["trick_winners"], game.game_state["trick_winners"]

        # Verify that the hands have been reset (if you want to test this, you may need additional logic)
        for player in game.players:
            assert 6 == len(player.hand)

        # Verify that the play_experiences_cache is empty
        assert all(item[1] is None for item in game.bid_experiences_cache)
        assert all(all(item[1] is None for item in row) for row in game.play_experiences_cache)

    def test_reset_round(self):
        # Initialize models for bidding and playing (replace with actual models if needed)
        bid_model = None
        play_model = None

        # Initialize a game
        game = Game(bid_model, play_model)

        # Modify the game state to simulate a game in progress
        game.game_state["score"] = [15, 20]
        game.game_state["played_cards"][0] = [('9S', 0), ('9H', 1), ('9D', 2), ('JC', 3)]
        game.game_state["all_bids"] = [4, 5, 0, 'pfeffer']
        game.game_state["winning_bid"] = ('pfeffer', 3, 'C')
        game.game_state["lead_players"][0] = 0
        game.game_state["trick_winners"][0] = 3
        game.game_state["dealer_position"] = 0

        # Store the original score to verify that it doesn't change
        original_score = game.game_state["score"].copy()

        # Call the reset_round method
        game.reset_round()

        # Verify that round-specific state has been reset
        assert all(len(trick) == 0 for trick in game.game_state["played_cards"])
        assert [] == game.game_state["all_bids"]
        assert (-1, -1, None) == game.game_state["winning_bid"]
        assert [None] * 6 == game.game_state["lead_players"], game.game_state["lead_players"]
        assert [None] * 6 == game.game_state["trick_winners"], game.game_state["trick_winners"]

        # Verify that the score has not been changed
        assert original_score == game.game_state["score"]

        # Verify that the hands have been reset (if you want to test this, you may need additional logic)
        for player in game.players:
            assert 6 == len(player.hand)

        # Dealer moves to next position
        self.assertEqual(1, game.game_state["dealer_position"])

        # Verify that the play_experiences_cache is empty
        assert all(item[1] is None for item in game.bid_experiences_cache)
        assert all(all(item[1] is None for item in row) for row in game.play_experiences_cache)

    def test_evaluate_round_4bid_4_2(self):
        game = Game(Mock(), Mock())

        game.game_state["winning_bid"] = (4, 0, 'C')
        game.game_state["trick_winners"] = [0, 1, 2, 3, 0, 2]

        score_team1, score_team2 = game.evaluate_round()

        self.assertEqual(4, score_team1)
        self.assertEqual(2, score_team2)

    def test_evaluate_round_4bid_6_0(self):
        game = Game(Mock(), Mock())

        game.game_state["winning_bid"] = (4, 0, 'C')
        game.game_state["trick_winners"] = [0, 2, 0, 2, 0, 2]

        score_team1, score_team2 = game.evaluate_round()

        self.assertEqual(6, score_team1)
        self.assertEqual(-5, score_team2)

    def test_evaluate_round_4bid_3_3(self):
        game = Game(Mock(), Mock())

        game.game_state["winning_bid"] = (4, 0, 'C')
        game.game_state["trick_winners"] = [0, 2, 0, 1, 1, 3]

        score_team1, score_team2 = game.evaluate_round()

        self.assertEqual(-5, score_team1)
        self.assertEqual(3, score_team2)

    def test_evaluate_round_pfeffer_bid_6_0(self):
        game = Game(Mock(), Mock())

        game.game_state["winning_bid"] = ('pfeffer', 0, 'C')
        game.game_state["trick_winners"] = [0, 0, 0, 0, 0, 0]

        score_team1, score_team2 = game.evaluate_round()

        self.assertEqual(12, score_team1)
        self.assertEqual(-5, score_team2)

    def test_evaluate_round_pfeffer_bid_5_1(self):
        game = Game(Mock(), Mock())

        game.game_state["winning_bid"] = ('pfeffer', 0, 'C')
        game.game_state["trick_winners"] = [0, 0, 0, 0, 0, 1]

        score_team1, score_team2 = game.evaluate_round()

        self.assertEqual(-12, score_team1)
        self.assertEqual(1, score_team2)

    def test_bid_and_play_round(self):
        bid_model = Mock()
        bid_model.predict.return_value = np.zeros(11)
        play_model = Mock()
        play_model.predict.return_value = np.zeros(24)

        game = Game(bid_model, play_model)
        game.reset()

        game.bid_round()
        game.play_round()

        # Each player should have:
        #   1 bid experience
        #   6 play experiences
        for i, player in enumerate(game.players):
            self.assertEqual(1, player.bid_replay_buffer.num_frames())
            self.assertEqual(6, player.play_replay_buffer.num_frames())

            trajectories, _ = player.play_replay_buffer.get_next()
            print(trajectories.observation)


class TestPlayer(TestCase):

    def test_save_to_bid_buffer(self):
        # Mocks and initial setup
        mock_bidding_model = Mock()
        mock_play_model = Mock()
        mock_bid_actions = Mock()
        mock_play_actions = Mock()

        # Initialize Player object
        player = Player(0, mock_bidding_model, mock_play_model, mock_bid_actions, mock_play_actions)

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
        mock_bid_actions = Mock()
        mock_play_actions = Mock()

        # Initialize Player object
        player = Player(0, mock_bidding_model, mock_play_model, mock_bid_actions, mock_play_actions)

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
        action_taken = '9S'  # Example action taken (you'll need to specify this based on your game logic)
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
        cached_action = trajectory.action[0][0]

        # Decode the saved state (you'll need to use decoding methods similar to those in BidInput)
        decoded_play_input = PlayInput.decode(cached_observation)  # Replace with the actual decoding logic

        # Check if play state is saved correctly
        self.assertEqual(decoded_play_input, play_input)

        # Check if action is saved correctly
        self.assertEqual(cached_action, action_taken)

        # Check if reward is saved correctly
        self.assertEqual(trajectory.reward[0][0], reward_received)


class TestBidInput(TestCase):

    def test_encode_decode(self):
        # Define a sample game state
        game_state = {
            'hands': {
                0: ['9S', 'TS', 'JS', 'QS', 'KS', 'AS'],
                # other player hands if needed
            },
            'all_bids': [4, 5, 'pfeffer', ],
            'dealer_position': 2,
            'score': [10, 12],
            # other game state attributes if needed
        }

        # Create a BidInput object using from_game_state
        player_id = 0
        bidding_input = BidInput.from_game_state(player_id, game_state)

        # Encode the state
        encoded_state = bidding_input.encode()

        # Decode the state
        decoded_bidding_input = BidInput.decode(encoded_state)

        # Validate the decoded state against the original
        assert bidding_input.__dict__ == decoded_bidding_input.__dict__, \
            f"Original and decoded objects do not match: \n{bidding_input.__dict__} \n{decoded_bidding_input.__dict__}"


class TestPlayInput(TestCase):

    def test_encode_decode_all_cards(self):
        possible_ranks = ['9', 'T', 'J', 'Q', 'K', 'A']
        possible_suits = ['S', 'H', 'D', 'C']
        all_cards = [rank + suit for rank in possible_ranks for suit in possible_suits]

        for card in all_cards:
            encoded_card = PlayInput.encode_card(card)
            decoded_card = PlayInput.decode_card(encoded_card)
            print(f"{card} {decoded_card} {encoded_card}")
            assert card == decoded_card, f"{card} != {decoded_card}"

    def test_encode_decode_played_cards(self):
        played_cards = [[('9S', 0), ('9H', 1), ('9D', 2), ('9C', 3)], [('JC', 3), ], [], [], [], []]
        # played_cards = [[('9S', 0), ('9H', 1), ('9D', 2), ('9C', 3)], [('JC', 3), ]]

        # Encode and then decode the played cards
        encoded_played_cards = PlayInput.encode_played_cards(played_cards)

        # Check if the length of the encoded array is correct
        expected_length = 4 * 24 * 6  # 4 players * 24 cards * 6 tricks
        assert len(encoded_played_cards) == expected_length, \
            f"Expected length: {expected_length}, got: {len(encoded_played_cards)}"

        # Decode the played cards
        decoded_played_cards = PlayInput.decode_played_cards(encoded_played_cards)

        # The decoded played cards should match the original played cards (without the player ids)
        assert played_cards == decoded_played_cards, \
            f"\n{played_cards}\n{decoded_played_cards}"

    def test_encode_and_decode_list_of_players(self):
        # The original list of player states.
        # Please replace with your actual test data
        lead_players = [0, 3, None, None, None, None]

        encoded_players = PlayInput.encode_list_of_players(lead_players)
        expected_encoded_players = [
            0, 1, 0, 0, 0,
            0, 0, 0, 0, 1,
            1, 0, 0, 0, 0,
            1, 0, 0, 0, 0,
            1, 0, 0, 0, 0,
            1, 0, 0, 0, 0,
        ]

        assert encoded_players == expected_encoded_players, \
            f"encoded_players {encoded_players}\nexpected_encoded_players {expected_encoded_players}"

        decoded_data = PlayInput.decode_list_of_players(encoded_players)

        assert lead_players == decoded_data,  \
            f"lead_players {lead_players}\ndecoded_data {decoded_data}"

    def test_encode_decode(self):
        # Example Game State (This would normally be constructed and updated during the game)
        game_state = {
            "hands": {
                0: ['9S', 'TS', 'JS', 'QS', 'KS', 'AS'],
                1: ['9H', 'TH', 'JH', 'QH', 'KH', 'AH'],
                2: ['9D', 'TD', 'JD', 'QD', 'KD', 'AD'],
                3: ['9C', 'TC', 'JC', 'QC', 'KC', 'AC']
            },
            "played_cards": [
                [('9S', 0), ('9H', 1), ('9D', 2), ('9C', 3)],  # First trick
                [('JC', 3), ],  # Second trick not yet played
                [],  # and so on...
                [],
                [],
                []
            ],
            "bidding_order": [0, 1, 2, 3],
            "all_bids": [4, 5, 0, 'pfeffer'],
            "winning_bid": ('pfeffer', 3, 'C'),
            "lead_players": [0, 3, None, None, None, None],
            "trick_winners": [3, None, None, None, None, None],
            "score": [12, -5]
        }

        # 1. Create a PlayInput object from the game_state
        play_input_original = PlayInput.from_game_state(0, game_state)

        # 2. Encode the PlayInput object
        encoded_state = play_input_original.encode()
        # print(f"encoded state: {encoded_state}")

        # 3. Decode the encoded state
        play_input_decoded = PlayInput.decode(encoded_state)
        print(f"original game state: {game_state}")
        print(f"original play input: {play_input_original.__dict__}")
        print(f"decoded play input:  {play_input_decoded.__dict__}")

        # 4. Compare the original and decoded PlayInput objects
        assert play_input_original.__dict__ == play_input_decoded.__dict__, \
            "The original and decoded PlayInput objects are not equal!"
