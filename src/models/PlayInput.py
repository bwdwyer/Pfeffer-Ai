import numpy as np
import tensorflow as tf

from src.models import find_index, BID_VALUES, BID_SUITS


class PlayInput:
    """
    Represents the input to the Q-network for the game Pfeffer.

    Attributes:
        player_id (int): The ID of the player (0-3).
        hand (list): The list of cards in the player's hand.
        played_cards (list): A 6x4 matrix representing played cards.
        bidding_order (list): The sequence of players who bid.
        all_bids (list): The list of bids made by all players.
        winning_bid (tuple): The bid and suit/no-trump that won the bidding phase.
        lead_players (list): The list of players who led each trick.
        current_trick (int): The current trick number.
        score (list): The current score for each team.
    """

    def __init__(self, player_id, hand, played_cards, bidding_order, all_bids, winning_bid, lead_players, trick_winners,
                 current_trick, score, ):
        self.player_id = player_id
        self.hand = hand
        self.played_cards = played_cards
        self.bidding_order = bidding_order
        self.all_bids = all_bids
        self.winning_bid = winning_bid
        self.lead_players = lead_players
        self.trick_winners = trick_winners
        self.current_trick = current_trick
        self.score = score

    def __eq__(self, other):
        if isinstance(other, PlayInput):
            return self.__dict__ == other.__dict__
        return False

    @staticmethod
    def encode_card(card):
        """
        Encodes a card into a one-hot vector.

        Args:
            card (str): The card to encode.

        Returns:
            list: A 24-element one-hot encoding of the card.
        """
        if card is None:
            return [0] * 24
        rank, suit = card[:-1], card[-1]
        rank_index = ['9', 'T', 'J', 'Q', 'K', 'A'].index(rank)
        suit_index = ['S', 'H', 'D', 'C'].index(suit)
        encoding = [0] * 24
        encoding[rank_index * 4 + suit_index] = 1
        return encoding

    @staticmethod
    def decode_card(encoded_card: tf.Tensor):
        """
        Decodes a one-hot encoded card back to its string representation.

        Args:
            encoded_card (tf.Tensor): A 24-element one-hot encoding of the card.

        Returns:
            str: The decoded card in string format or None if the encoding is all zeros.
        """
        # If the encoded card is a vector of zeros, return None
        if tf.reduce_sum(encoded_card) == 0:
            return None

        possible_ranks = ['9', 'T', 'J', 'Q', 'K', 'A']
        possible_suits = ['S', 'H', 'D', 'C']

        index = tf.argmax(encoded_card).numpy()
        rank = possible_ranks[index // 4]
        suit = possible_suits[index % 4]

        return rank + suit

    @staticmethod
    def encode_played_cards(played_cards):
        """
        Encodes a list of played cards.

        Args:
            played_cards (list): A list of played cards, organized by trick.

        Returns:
            np.array: A one-hot encoding of the played cards.
        """
        encoding = []
        for trick in played_cards:
            for player_id in range(4):
                card_encoding = [0] * 24  # Initialize encoding for each player
                for card, player in trick:
                    if player == player_id:  # If the player played a card in this trick
                        card_encoding = PlayInput.encode_card(card)  # Encode the card
                        break  # Only one card per player per trick, so we can stop here
                encoding += card_encoding  # Add encoding to the list
        return np.array(encoding)

    @staticmethod
    def decode_played_cards(encoded_cards):
        """
        Decodes a one-hot encoded list of played cards.

        Args:
            encoded_cards (tf.Tensor): A one-hot encoding of the played cards.

        Returns:
            list: The decoded played cards, organized by trick.
        """
        played_cards = []
        for i in range(6):  # for each trick
            trick = []
            for j in range(4):  # for each player
                card_encoded = encoded_cards[i * 4 * 24 + j * 24: i * 4 * 24 + (j + 1) * 24]
                card = PlayInput.decode_card(tf.cast(card_encoded, tf.int32))
                if card:
                    trick.append((card, j))
            played_cards.append(trick)
        return played_cards

    @staticmethod
    def decode_hand(encoded_hand):
        """
        Decodes a one-hot encoded hand back to its card representations.

        Args:
            encoded_hand (tf.Tensor): A 24-element one-hot encoding of the hand.

        Returns:
            list: The decoded cards in the hand.
        """
        hand = []
        for i in range(24):  # Iterate over each possible card
            if encoded_hand[i] == 1:
                card_encoding = tf.one_hot(i, 24)  # Create one-hot encoding for the card
                card = PlayInput.decode_card(card_encoding)
                hand.append(card)
        return hand

    @staticmethod
    def encode_bid(bid):
        """
        Encodes a bid into a one-hot vector.

        Args:
            bid (int/str): The bid to encode.

        Returns:
            list: A 5-element one-hot encoding of the bid.
        """
        possible_bids = [0, 4, 5, 6, 'pfeffer']
        encoding = [0] * len(possible_bids)

        index = possible_bids.index(bid)
        encoding[index] = 1

        return encoding

    @staticmethod
    def encode_winning_bid(winning_bid):
        """
        Encodes the winning bid into a one-hot format.

        Args:
            winning_bid (tuple): The winning bid in the form (quantity, player, suit).

        Returns:
            np.array: The one-hot encoding of the bid.
        """
        bid_quantity, player, bid_suit = winning_bid
        # possible_bids = [0, 4, 5, 6, 'pfeffer']
        # possible_suits = ['S', 'H', 'D', 'C', 'NT']
        possible_players = [0, 1, 2, 3]

        bid_quantity_encoding = [0] * len(BID_VALUES)
        bid_quantity_encoding[BID_VALUES.index(bid_quantity)] = 1

        player_encoding = [0] * len(possible_players)
        player_encoding[possible_players.index(player)] = 1

        bid_suit_encoding = [0] * len(BID_SUITS)
        bid_suit_encoding[BID_SUITS.index(bid_suit)] = 1

        return np.concatenate([bid_quantity_encoding, player_encoding, bid_suit_encoding])

    @staticmethod
    def decode_winning_bid(encoded_bid):
        """
        Decodes a one-hot encoded bid back to its original form.

        Args:
            encoded_bid (tf.Tensor): A one-hot encoding of the bid.

        Returns:
            tuple: The decoded bid in the form (quantity, player, suit) or None if the encoding is all zeros.
        """
        # If the encoded bid is a vector of zeros, return None
        if tf.reduce_sum(encoded_bid) == 0:
            return None

        possible_bids = [0, 4, 5, 6, 'pfeffer']
        possible_players = [0, 1, 2, 3]
        possible_suits = ['S', 'H', 'D', 'C', 'no-trump']

        bid_encoded = encoded_bid[:5]
        player_encoded = encoded_bid[5:9]
        suit_encoded = encoded_bid[9:]

        bid_quantity = possible_bids[tf.argmax(bid_encoded).numpy()]
        player = possible_players[tf.argmax(player_encoded).numpy()]
        bid_suit = possible_suits[tf.argmax(suit_encoded).numpy()]

        return bid_quantity, player, bid_suit

    @staticmethod
    def encode_list_of_players(players):
        """
        Encodes a list of players into one-hot vectors.

        Args:
            players (list): The players to be encoded.

        Returns:
            list: A 30-element one-hot encoding of a list of players.
        """
        possible_players = [-1, 0, 1, 2, 3]  # Include -1 as a possible player for tricks that haven't happened yet
        lead_players_encoding = []

        for player in players:
            player = player if player is not None else -1  # Set player to -1 if it's None
            player_encoding = [0] * len(possible_players)
            player_encoding[possible_players.index(player)] = 1
            lead_players_encoding.append(player_encoding)

        return [item for sublist in lead_players_encoding for item in sublist]

    @staticmethod
    def decode_list_of_players(encoded_players, include_none=True):
        """
        Decodes an encoded tensor of players to a list of players.

        Args:
            :param encoded_players: (tf.Tensor): An encoded tensor of players.
            :param include_none: Whether the encoded_players include values for no specified player.

        Returns:
            list: The decoded list of players.
        """
        lead_players = []
        possible_values = [-1, 0, 1, 2, 3] if include_none else [0, 1, 2, 3]
        vector_length = len(possible_values)
        num_players = int(len(encoded_players) / vector_length)
        for i in range(num_players):
            player_encoded = encoded_players[i * vector_length:i * vector_length + vector_length]
            player = PlayInput.decode_one_hot(player_encoded, possible_values)
            if player == -1:
                player = None
            lead_players.append(player)

        return lead_players

    @staticmethod
    def encode_current_trick(current_trick):
        """
        Encodes the current trick number into a one-hot vector.

        Args:
            current_trick (int): The current trick number.

        Returns:
            list: A 6-element one-hot encoding of the current trick.
        """
        possible_tricks = [0, 1, 2, 3, 4, 5]
        current_trick_encoding = [0] * len(possible_tricks)
        current_trick_encoding[possible_tricks.index(current_trick)] = 1

        return current_trick_encoding

    @staticmethod
    def encode_bidding_order(bidding_order):
        """
        Encodes the bidding order into one-hot vectors.

        Args:
            bidding_order (list): The bidding order.

        Returns:
            list: A flattened list of the one-hot encodings of the bidding order.
        """
        possible_players = [0, 1, 2, 3]
        bidding_order_encoding = []

        for player in bidding_order:
            player_encoding = [0] * len(possible_players)
            player_encoding[possible_players.index(player)] = 1
            bidding_order_encoding.extend(player_encoding)

        return bidding_order_encoding

    @staticmethod
    def decode_bidding_order(bidding_order_encoding):
        """
        Decode the bidding order from one-hot vectors.

        Args:
            bidding_order_encoding (list): A list of one-hot encodings of the bidding order.

        Returns:
            list: The original bidding order.
        """
        # We know that each player was represented by 4 bits in the encoding
        players_encoding_length = 4
        bidding_order = []

        # We'll iterate our encoded list 4 elements at a time
        for i in range(0, len(bidding_order_encoding), players_encoding_length):
            # We'll slice our list to get the current player's encoding
            player_encoding = bidding_order_encoding[i:i + players_encoding_length]

            # The index of '1' in the player_encoding list is the player number
            player = find_index(player_encoding, 1)

            # We add this player to our bidding_order
            bidding_order.append(player)

        return bidding_order

    def encode(self):
        """
        Encodes the state into a dictionary of lists.

        Returns:
            dict: A dictionary representing the encoded state.
        """
        player_id_encoding = [0] * 4
        player_id_encoding[self.player_id] = 1

        hand_encoding = [0] * 24
        for card in self.hand:
            card_one_hot = self.encode_card(card)
            hand_encoding = [x or y for x, y in zip(hand_encoding, card_one_hot)]

        played_cards_encoding = self.encode_played_cards(self.played_cards)
        bidding_order_encoding = self.encode_bidding_order(self.bidding_order)

        all_bids_encoding = [self.encode_bid(bid) for bid in self.all_bids]
        all_bids_encoding = [item for sublist in all_bids_encoding for item in sublist]

        winning_bid_encoding = self.encode_winning_bid(self.winning_bid)
        lead_players_encoding = self.encode_list_of_players(self.lead_players)
        trick_winners_encoding = self.encode_list_of_players(self.trick_winners)
        current_trick_encoding = self.encode_current_trick(self.current_trick)

        score_encoding = self.score

        return {
            'player_id': tf.constant(player_id_encoding, dtype=tf.int32),
            'hand': tf.constant(hand_encoding, dtype=tf.int32),
            'played_cards': tf.constant(played_cards_encoding, dtype=tf.int32),
            'bidding_order': tf.constant(bidding_order_encoding, dtype=tf.int32),
            'all_bids': tf.constant(all_bids_encoding, dtype=tf.int32),
            'winning_bid': tf.constant(winning_bid_encoding, dtype=tf.int32),
            'lead_players': tf.constant(lead_players_encoding, dtype=tf.int32),
            'trick_winners': tf.constant(trick_winners_encoding, dtype=tf.int32),
            'current_trick': tf.constant(current_trick_encoding, dtype=tf.int32),
            'score': tf.constant(score_encoding, dtype=tf.int32),
        }

    @staticmethod
    def decode_one_hot(one_hot_tensor, possible_values):
        """
        Decodes a one-hot encoded tensor.

        Args:
            one_hot_tensor (tf.Tensor): One-hot encoded tensor of shape [n], where n is the number of possible values.
            possible_values (List[Any]): Possible values that can be represented by the one-hot encoding.

        Returns:
            Any: The value represented by the one-hot encoding, or None if the one-hot encoding is all zeros.
        """

        # Check if the tensor is all zeros
        if tf.reduce_sum(one_hot_tensor) == 0:
            return None

        # Find the index of the '1' in the one-hot tensor
        index = tf.argmax(one_hot_tensor, axis=0).numpy()

        return possible_values[index]

    @classmethod
    def decode(cls, encoded_state):
        """
        Decodes an encoded state back to a PlayInput object.

        Args:
            encoded_state (dict): The encoded state.

        Returns:
            PlayInput: The decoded PlayInput object.
        """
        # Decode player_id
        player_id = PlayInput.decode_one_hot(encoded_state['player_id'], [0, 1, 2, 3])

        # Decode hand
        hand = PlayInput.decode_hand(encoded_state['hand'])

        # Decode played_cards
        played_cards = PlayInput.decode_played_cards(encoded_state['played_cards'])

        # Decode bidding_order
        bidding_order = PlayInput.decode_bidding_order(encoded_state['bidding_order'])

        # Decode all_bids
        all_bids = [
            PlayInput.decode_one_hot(
                encoded_state['all_bids'][i:i + 5],
                [0, 4, 5, 6, 'pfeffer']
            ) for i in range(0, len(encoded_state['all_bids']), 5)
        ]

        # Decode winning_bid
        winning_bid = PlayInput.decode_winning_bid(encoded_state['winning_bid'])

        # Decode lead_players
        lead_players = PlayInput.decode_list_of_players(encoded_state['lead_players'])

        # Decode trick_winners
        trick_winners = PlayInput.decode_list_of_players(encoded_state['trick_winners'])

        # Decode current_trick
        current_trick = PlayInput.decode_one_hot(encoded_state['current_trick'], [0, 1, 2, 3, 4, 5])

        # Extract score
        score = list(encoded_state['score'])

        return cls(player_id, hand, played_cards, bidding_order, all_bids, winning_bid, lead_players, trick_winners,
                   current_trick, score)

    @classmethod
    def from_game_state(cls, player_id, game_state):
        """Creates a PlayInput object from a game state."""
        # Extract relevant information from the game state
        hand = game_state["hands"][player_id]
        played_cards = game_state["played_cards"]
        bidding_order = game_state["bidding_order"]
        all_bids = game_state["all_bids"]
        winning_bid = game_state["winning_bid"]
        lead_players = game_state["lead_players"]
        trick_winners = game_state["trick_winners"]
        current_trick = max((index for index, trick in enumerate(played_cards) if trick), default=0) \
            if played_cards else 0
        score = game_state["score"]

        # Create a PlayInput object
        play_input = cls(player_id, hand, played_cards, bidding_order, all_bids, winning_bid, lead_players,
                         trick_winners, current_trick, score)

        return play_input
