import numpy as np
import tensorflow as tf


class BidInput:
    """
    A class to represent the state of the game during the bidding phase, from the perspective of a single player.

    ...

    Attributes
    ----------
    hand : list
        A list of cards in the player's hand, e.g., ['9S', 'TS', 'JD', 'AS', 'KH', ...]
    previous_bids : list
        A list of previous bids for all players.
    dealer_position : int
        The position of the dealer.
    score : list
        The current score for all players or teams.

    Methods
    -------
    encode() -> np.ndarray
        Encodes the bidding input into a single numpy array. The encoded array includes one-hot encodings for the cards
        in the player's hand and previous bids, as well as the dealer position and score.

    from_game_state(player_id: int, game_state: dict) -> BidInput:
        Creates a BidInput object from a game state. The game state is a dictionary that includes information about
        the current state of the game, such as the hands of all players, all bids made so far, the position of the
        dealer, and the score.

    encode_hand(hand: list) -> np.ndarray:
        Encodes a hand of cards into a one-hot vector. The vector has 24 elements, one for each possible card (6 ranks
        and 4 suits).

    encode_bid(bid: int or str) -> np.ndarray:
        Encodes a bid into a one-hot vector. The vector has elements corresponding to possible bids including 'pfeffer'.
        The bid is assigned a value of 1 in the vector, with all other elements set to 0.
    """

    def __init__(self, hand, previous_bids, dealer_position, score):
        self.hand = hand  # List of cards in hand, e.g., ['9S', 'TS', 'JD', 'AS', 'KH', ...]
        self.previous_bids = previous_bids  # List of previous bids for all players
        self.dealer_position = dealer_position  # Position of the dealer
        self.score = score  # Current score for all players or teams

    @staticmethod
    def encode_hand(hand):
        """
        Encodes a hand of cards into a binary representation.

        This method assumes that the hand is a list of strings, each string
        representing a card. The cards are encoded in the order of ranks and suits
        defined by the list ['9', 'T', 'J', 'Q', 'K', 'A'] and ['S', 'H', 'D', 'C']
        respectively.

        Each card in the hand is represented as a one-hot encoded vector of length 24
        (6 ranks x 4 suits). The vector is then flattened into a 1D array.

        Args:
            hand (list): A list of strings, each string representing a card in the hand.

        Returns:
            numpy.ndarray: A 24-element binary array representing the hand.
        """
        hand_encoding = np.zeros((24,))  # 6 ranks x 4 suits
        for card in hand:
            rank, suit = card[:-1], card[-1]
            rank_index = ['9', 'T', 'J', 'Q', 'K', 'A'].index(rank)
            suit_index = ['S', 'H', 'D', 'C'].index(suit)
            hand_encoding[rank_index * 4 + suit_index] = 1

        return hand_encoding

    @staticmethod
    def decode_hand(encoded_hand):
        """
        Decodes a one-hot encoded hand back to its original form.

        Args:
            encoded_hand (ndarray): A 24-element one-hot encoding of the hand.

        Returns:
            list: The decoded hand as a list of strings representing cards.
        """
        possible_ranks = ['9', 'T', 'J', 'Q', 'K', 'A']
        possible_suits = ['S', 'H', 'D', 'C']

        hand = []
        for i in range(0, len(encoded_hand), 4):
            if sum(encoded_hand[i:i + 4]) != 0:  # If the card exists in the hand
                index = np.argmax(encoded_hand[i:i + 4])
                rank = possible_ranks[i // 4]
                suit = possible_suits[index]
                hand.append(rank + suit)
        return hand

    @staticmethod
    def encode_bid(bid):
        """
        Encodes a bid into a binary representation.

        This method assumes that the bid is either a string or an integer, and
        that the possible bids are [0, 4, 5, 6, 'pfeffer'].

        Each bid is represented as a one-hot encoded vector of length equal to
        the number of possible bids. If the bid is None, all elements of the
        vector are zero.

        Args:
            bid (int or str): The bid to encode.

        Returns:
            list: A one-hot encoded list representing the bid.
        """
        possible_bids = [0, 4, 5, 6, 'pfeffer']
        encoding = [0] * len(possible_bids)

        index = possible_bids.index(bid) if bid is not None else -1
        if index != -1:
            encoding[index] = 1

        return encoding

    @staticmethod
    def decode_bid(encoded_bid):
        """
        Decodes a one-hot encoded bid back to its original form.

        Args:
            encoded_bid (ndarray): A 6-element one-hot encoding of the bid.

        Returns:
            str or int: The decoded bid.
        """
        possible_bids = [0, 4, 5, 6, 'pfeffer']
        index = np.argmax(encoded_bid)
        return possible_bids[index]

    def encode(self):
        """
        Encodes the bidding state into a dictionary of TensorFlow tensors.

        Returns:
            dict: A dictionary representing the encoded bidding state.
        """
        hand_encoding = BidInput.encode_hand(self.hand)

        # Flatten the previous bids encoding for only the first three bids
        previous_bids_encoding = [BidInput.encode_bid(bid) for bid in self.previous_bids[:3]]
        previous_bids_encoding = [item for sublist in previous_bids_encoding for item in sublist]

        dealer_position_encoding = [self.dealer_position]
        score_encoding = self.score

        return {
            'hand': tf.constant(hand_encoding, dtype=tf.int32),
            'previous_bids': tf.constant(previous_bids_encoding, dtype=tf.int32),
            'dealer_position': tf.constant(dealer_position_encoding, dtype=tf.int32),
            'score': tf.constant(score_encoding, dtype=tf.int32),
        }

    @classmethod
    def decode(cls, encoded_state):
        """
        Decodes the encoded state into a BidInput object.

        Args:
            encoded_state (dict): Encoded state as a dictionary of TensorFlow tensors.

        Returns:
            BidInput: Decoded BidInput object.
        """
        # Convert TensorFlow tensors to numpy arrays
        hand_encoded = encoded_state['hand'].numpy()
        previous_bids_encoded = encoded_state['previous_bids'].numpy()
        dealer_position_encoded = encoded_state['dealer_position'].numpy()
        score_encoded = encoded_state['score'].numpy()

        # Decode hand
        hand = cls.decode_hand(hand_encoded)

        # Decode previous bids
        previous_bids = []
        for i in range(0, len(previous_bids_encoded), 5):
            bid_encoded = previous_bids_encoded[i:i + 5]
            bid = cls.decode_bid(bid_encoded)
            previous_bids.append(bid)

        # Decode dealer position
        dealer_position = int(dealer_position_encoded[0])

        # Decode score
        score = score_encoded.tolist()

        return cls(hand, previous_bids, dealer_position, score)

    @classmethod
    def from_game_state(cls, player_id, game_state):
        """Creates a BidInput object from a game state."""
        # Extract relevant information from the game state
        hand = game_state["hands"][player_id]
        previous_bids = game_state["all_bids"]
        dealer_position = game_state["dealer_position"]
        score = game_state["score"]

        # Create a BidInput object
        bid_input = cls(hand, previous_bids, dealer_position, score)

        return bid_input
