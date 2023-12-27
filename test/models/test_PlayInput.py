from unittest import TestCase

from src.models.PlayInput import PlayInput


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

        assert lead_players == decoded_data, \
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
