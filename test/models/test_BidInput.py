from unittest import TestCase

from src.models.BidInput import BidInput


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
