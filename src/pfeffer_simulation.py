import random

import numpy as np
import tensorflow as tf

SUITS_COLORS = {"S": "black", "C": "black", "H": "red", "D": "red"}


class Game:

    def __init__(self, bid_model, play_model):
        self.bid_actions = BiddingActions()
        self.play_actions = PlayActions()

        self.players = [Player(i, bid_model, play_model, self.bid_actions, self.play_actions) for i in range(4)]

        self.game_state = {
            "hands": {player.player_id: [] for player in self.players},  # Empty hands at the start
            "score": [0, 0],  # Score starts at 0 for each team
            "current_trick": [],  # No tricks have been played
            "played_cards": [[] for _ in range(6)],  # List of played cards for each trick
            "bidding_order": [0, 1, 2, 3],  # Bidding order might be fixed, or determined by some rule
            "all_bids": [],  # No bids have been made
            "winning_bid": (-1, -1, None),  # No winning bid yet, (bid, player_id, trump_suit)
            "lead_players": [None] * 6,  # No lead players yet
        }

    def reset(self):
        """Resets the game to the initial state."""
        # Shuffle and deal new hands to each player
        deck = ['9S', 'TS', 'JS', 'QS', 'KS', 'AS',
                '9H', 'TH', 'JH', 'QH', 'KH', 'AH',
                '9D', 'TD', 'JD', 'QD', 'KD', 'AD',
                '9C', 'TC', 'JC', 'QC', 'KC', 'AC']
        random.shuffle(deck)
        for i, player in enumerate(self.players):
            self.game_state["hands"][player.player_id] = deck[i * 6:(i + 1) * 6]  # Deal 6 cards to each player

        # Reset score
        self.game_state["score"] = [0, 0]

        # Reset current trick, played cards, all bids, winning bid, and lead players
        self.game_state["current_trick"] = []
        self.game_state["played_cards"] = [[] for _ in range(6)]
        self.game_state["all_bids"] = []
        self.game_state["winning_bid"] = (-1, -1, None)
        self.game_state["lead_players"] = [None] * 6

    def step(self, player, action):
        """Executes a player's action and updates the game state."""
        # Update game state based on action
        pass

    def bid_round(self):
        """Conducts the bidding round."""
        for player in self.players:
            # Get bid from player
            bid, trump_suit = player.make_bid(self.game_state)

            # Update game state with bid
            self.game_state["all_bids"].append(bid)

            # Check if this bid is higher than the current winning bid
            if bid > self.game_state["winning_bid"][0]:
                self.game_state["winning_bid"] = (bid, player.player_id, trump_suit)

    @staticmethod
    def generate_player_order(start_player):
        """Generates the order of players for a trick, starting with the specified player."""
        return [start_player, (start_player + 1) % 4, (start_player + 2) % 4, (start_player + 3) % 4]

    def play_round(self):
        """Plays a round by choosing an action using the Q-network.

        Returns:
            action (str): The chosen action (card to play).
        """
        # Get the player who won the bid
        bid_winner = self.game_state["winning_bid"][1]
        bid_value = self.game_state["winning_bid"][0]

        # Determine if the bid winner is playing alone (i.e., bid 'pfeffer')
        playing_alone = bid_value == 'pfeffer'

        # Get the partner of the bid winner (the player across)
        partner_id = (bid_winner + 2) % 4

        # Get a copy of the bidding order and rotate it so the bid winner goes first
        play_order = self.game_state["bidding_order"][:]
        while play_order[0] != bid_winner:
            play_order.append(play_order.pop(0))

        for trick in range(6):  # 6 tricks in a round
            for player_id in play_order:
                # Skip the partner's turn if the bid winner is playing alone
                if playing_alone and player_id == partner_id:
                    self.game_state["played_cards"][trick].append(
                        (partner_id, None))  # Record that the partner did not play a card
                    continue

                # Ask the player to make a play
                play = self.players[player_id].make_play(self.game_state)

                # Update the game state with the play
                self.game_state["played_cards"][trick].append((player_id, play))

            # Update the play order so the player who won the trick leads the next one
            trick_winner = self.evaluate_trick()
            # noinspection PyTypeChecker
            self.game_state["lead_players"][trick] = trick_winner
            while play_order[0] != trick_winner:
                play_order.append(play_order.pop(0))

    def evaluate_trick(self):
        """
        Evaluates the winner of a trick.

        Returns:
            int: player_id of the winner.
        """
        trick = self.game_state["played_cards"][-1]  # Last trick
        lead_suit = trick[0][1][-1]  # Suit of the first card played in the trick
        trump_suit = self.game_state["winning_bid"][2]

        # Define the left bauer based on trump suit
        opposite_color_suits = {"black": ["S", "C"], "red": ["H", "D"]}
        left_bauer_suit = [s for s in opposite_color_suits[SUITS_COLORS[trump_suit]] if s != trump_suit][0]

        # Rank the cards based on their value
        def card_value(card):
            rank, suit = card[:-1], card[-1]
            rank_values = {'9': 0, 'T': 1, 'J': 2, 'Q': 3, 'K': 4, 'A': 5}  # Default values

            # Check if the card is the right bauer, left bauer, another trump, or if it's following the lead suit
            if suit == trump_suit:
                if rank == 'J':  # Right bauer
                    return 20
                return rank_values[rank] + 10  # Trump cards are ranked higher
            elif suit == left_bauer_suit and rank == 'J':  # Left bauer
                return 19
            elif suit == lead_suit:
                return rank_values[rank]
            else:
                return -1  # Cards not following the lead and not trump are ranked lowest

        # Find the highest card played in the trick
        winning_card = max(trick, key=lambda x: card_value(x[1]))

        # Return the player_id of the winner
        return winning_card[0]

    def play_game(self):
        """Plays a full game."""
        self.reset()
        while not self.game_over():
            self.bid_round()
            self.play_round()

    def game_over(self):
        """Checks if the game is over."""
        # The game is over if either team's score is 32 or more
        return any(score >= 32 for score in self.game_state["score"])


class Player:
    def __init__(self, player_id, bidding_model, play_model, bidding_actions, play_actions):
        self.player_id = player_id
        self.bidding_model = bidding_model
        self.play_model = play_model
        self.bidding_actions = bidding_actions
        self.play_actions = play_actions

    def make_bid(self, game_state):
        """Makes a bid and chooses a trump suit based on the current game state."""
        # Create a BiddingInput object from the game state
        bidding_input = BiddingInput.from_game_state(self.player_id, game_state)

        # Get the Q-values from the bidding model
        q_values = self.bidding_model.predict(bidding_input.encode()[tf.newaxis, :])

        # Get the highest previous bid
        highest_previous_bid = max(game_state["all_bids"])

        # Modify Q-values for bids that are not greater than the highest previous bid and are not 0
        q_values[0][1:highest_previous_bid + 1] = -np.inf  # Start from index 1 to exclude 0

        # The action (bid) with the highest Q-value is the best bid
        bid_index = np.argmax(q_values[0][:6])  # Only consider the first 6 elements (bids)
        bid = self.bidding_actions.get_action(bid_index)

        # The action (suit) with the highest Q-value is the best suit to choose as trump
        trump_suit_index = np.argmax(q_values[0][6:]) + 6  # Only consider the last 5 elements (suits)
        trump_suit = self.bidding_actions.get_action(trump_suit_index)

        return bid, trump_suit

    def make_play(self, game_state):
        """
        Makes a play by choosing a card using the Q-network.

        Args:
            game_state (dict): The current state of the game.

        Returns:
            action (str): The chosen action (card to play).
        """
        # Get current state of the game and encode it
        play_input = PlayInput.from_game_state(self.player_id, game_state)
        state_vector = play_input.encode()

        # Get Q-values for the current state
        q_values = self.play_model.predict(state_vector[None, :], verbose=0)[0]

        # Mask out illegal actions (cards not in hand)
        mask = [1 if card in play_input.hand else 0 for card in self.play_actions.CARDS]

        # If it's not the first card in the trick, it must follow suit if possible
        if game_state["played_cards"][-1]:
            lead_suit = game_state["played_cards"][-1][0][1][-1]
            trump_suit = game_state["winning_bid"][2]

            # Define the left bauer based on trump suit
            opposite_color_suits = {"black": ["S", "C"], "red": ["H", "D"]}
            left_bauer_suit = [s for s in opposite_color_suits[SUITS_COLORS[trump_suit]] if s != trump_suit][0]
            right_bauer = 'J' + trump_suit
            left_bauer = 'J' + left_bauer_suit

            # Modify mask for lead suit, right bauer, and left bauer
            mask_suit = []
            for card in self.play_actions.CARDS:
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


class BiddingInput:
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

    from_game_state(player_id: int, game_state: dict) -> BiddingInput:
        Creates a BiddingInput object from a game state. The game state is a dictionary that includes information about
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
        # Encoding cards (9 through Ace for 4 suits)
        hand_encoding = np.zeros((24,))  # 6 ranks x 4 suits
        for card in hand:
            rank, suit = card[:-1], card[-1]
            rank_index = ['9', 'T', 'J', 'Q', 'K', 'A'].index(rank)
            suit_index = ['S', 'H', 'D', 'C'].index(suit)
            hand_encoding[rank_index * 4 + suit_index] = 1

        return hand_encoding

    @staticmethod
    def encode_bid(bid):
        # Define possible bids including 'pfeffer'
        possible_bids = [0, 4, 5, 6, 'pfeffer']
        encoding = [0] * len(possible_bids)

        index = possible_bids.index(bid) if bid is not None else -1
        if index != -1:
            encoding[index] = 1

        return encoding

    def encode(self):
        hand_encoding = BiddingInput.encode_hand(self.hand)

        previous_bids_encoding = [BiddingInput.encode_bid(bid) for bid in self.previous_bids]
        previous_bids_encoding = [item for sublist in previous_bids_encoding for item in sublist]

        # Combine all features
        input_vector = np.concatenate([
            hand_encoding,
            np.array([self.dealer_position]),
            np.array(self.score),
            np.array(previous_bids_encoding),
        ])

        return input_vector

    @classmethod
    def from_game_state(cls, player_id, game_state):
        """Creates a BiddingInput object from a game state."""
        # Extract relevant information from the game state
        hand = game_state["hands"][player_id]
        previous_bids = game_state["all_bids"]
        dealer_position = game_state["dealer_position"]
        score = game_state["score"]

        # Create a BiddingInput object
        bidding_input = cls(hand, previous_bids, dealer_position, score)

        return bidding_input


class BiddingActions:
    """Class to handle the action space for the bidding phase."""

    BIDS = [0, 4, 5, 6, 'pfeffer']
    SUITS = ['S', 'H', 'D', 'C', 'NT']  # NT represents No-Trump

    def __init__(self):
        """Initializes the action space for the bidding phase."""
        self.actions = self.BIDS + self.SUITS

    def get_action(self, index):
        """Returns the action corresponding to a given index.

        Args:
            index (int): The index of the action.

        Returns:
            int or str: The action corresponding to the index.
        """
        return self.actions[index]

    def get_index(self, action):
        """Returns the index corresponding to a given action.

        Args:
            action (int or str): The action.

        Returns:
            int: The index corresponding to the action.
        """
        return self.actions.index(action)

    def get_number_of_actions(self):
        """Returns the total number of actions in the action space.

        Returns:
            int: The total number of actions.
        """
        return len(self.actions)

    def is_bid(self, action):
        """Checks if an action is a bid.

        Args:
            action (int or str): The action.

        Returns:
            bool: True if the action is a bid, False otherwise.
        """
        return action in self.BIDS

    def is_suit_choice(self, action):
        """Checks if an action is a suit choice.

        Args:
            action (str): The action.

        Returns:
            bool: True if the action is a suit choice, False otherwise.
        """
        return action in self.SUITS


class PlayInput:
    """Represents the input to the Q-network for the game Pfeffer.

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

    def __init__(self, player_id, hand, played_cards, bidding_order, all_bids, winning_bid, lead_players, current_trick,
                 score, ):
        self.player_id = player_id
        self.hand = hand
        self.played_cards = played_cards
        self.bidding_order = bidding_order
        self.all_bids = all_bids
        self.winning_bid = winning_bid
        self.lead_players = lead_players
        self.current_trick = current_trick
        self.score = score

    @staticmethod
    def encode_card(card):
        """Encodes a card into a one-hot vector.

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
    def encode_bid(bid):
        """Encodes a bid into a one-hot vector.

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
        """Encodes a winning bid into a one-hot vector.

        Args:
            winning_bid (tuple): The winning bid to encode.

        Returns:
            list: A 10-element one-hot encoding of the winning bid.
        """
        possible_bids = [0, 4, 5, 6, 'pfeffer']
        possible_suits = ['S', 'H', 'D', 'C', 'no-trump']

        bid_quantity, bid_suit = winning_bid
        bid_quantity_encoding = [0] * len(possible_bids)
        bid_suit_encoding = [0] * len(possible_suits)

        bid_quantity_encoding[possible_bids.index(bid_quantity)] = 1
        bid_suit_encoding[possible_suits.index(bid_suit)] = 1

        return bid_quantity_encoding + bid_suit_encoding

    @staticmethod
    def encode_lead_players(lead_players):
        """Encodes the players who led each trick into one-hot vectors.

        Args:
            lead_players (list): The players who led each trick.

        Returns:
            list: A 24-element one-hot encoding of the lead players.
        """
        possible_players = [-1, 0, 1, 2, 3]  # Include -1 as a possible player for tricks that haven't happened yet
        lead_players_encoding = []

        for player in lead_players:
            player = player if player is not None else -1  # Set player to -1 if it's None
            player_encoding = [0] * len(possible_players)
            player_encoding[possible_players.index(player)] = 1
            lead_players_encoding.append(player_encoding)

        return [item for sublist in lead_players_encoding for item in sublist]

    @staticmethod
    def encode_current_trick(current_trick):
        """Encodes the current trick number into a one-hot vector.

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
        """Encodes the bidding order into one-hot vectors.

        Args:
            bidding_order (list): The bidding order.

        Returns:
            list: A 16-element one-hot encoding of the bidding order.
        """
        possible_players = [0, 1, 2, 3]
        bidding_order_encoding = []

        for player in bidding_order:
            player_encoding = [0] * len(possible_players)
            player_encoding[possible_players.index(player)] = 1
            bidding_order_encoding.append(player_encoding)

        return [item for sublist in bidding_order_encoding for item in sublist]

    def encode(self):
        """Encodes the state into a single vector.

        Returns:
            np.array: A single vector representing the encoded state.
        """
        player_id_encoding = [0] * 4
        player_id_encoding[self.player_id] = 1

        card_encoding = [0] * 24
        for card in self.hand:
            card_one_hot = self.encode_card(card)
            card_encoding = [x or y for x, y in zip(card_encoding, card_one_hot)]

        played_cards_encoding = [self.encode_card(card) for trick in self.played_cards for card in trick]
        played_cards_encoding = [item for sublist in played_cards_encoding for item in sublist]

        bidding_order_encoding = self.encode_bidding_order(self.bidding_order)

        all_bids_encoding = [self.encode_bid(bid) for bid in self.all_bids]
        all_bids_encoding = [item for sublist in all_bids_encoding for item in sublist]

        winning_bid_encoding = self.encode_winning_bid(self.winning_bid)

        lead_players_encoding = self.encode_lead_players(self.lead_players)

        current_trick_encoding = self.encode_current_trick(self.current_trick)

        score_encoding = self.score

        input_vector = np.concatenate([
            player_id_encoding,
            card_encoding,
            played_cards_encoding,
            bidding_order_encoding,
            all_bids_encoding,
            winning_bid_encoding,
            lead_players_encoding,
            current_trick_encoding,
            score_encoding,
        ])
        return input_vector

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
        current_trick = len(played_cards) - 1 if played_cards else 0
        score = game_state["score"]

        # Create a PlayInput object
        play_input = cls(player_id, hand, played_cards, bidding_order, all_bids, winning_bid, lead_players,
                         current_trick, score)

        return play_input


class PlayActions:
    """
    A class to manage and encode actions related to playing a card.

    Attributes:
        CARDS (list): List of possible cards a player can have.
    """

    CARDS = [
        '9S', 'TS', 'JS', 'QS', 'KS', 'AS',
        '9H', 'TH', 'JH', 'QH', 'KH', 'AH',
        '9D', 'TD', 'JD', 'QD', 'KD', 'AD',
        '9C', 'TC', 'JC', 'QC', 'KC', 'AC'
    ]

    def get_action(self, index):
        """
        Get the card action based on the given index.

        Args:
            index (int): The index of the action in the list of cards.

        Returns:
            str: The card represented by the given index.
        """
        return self.CARDS[index]

    def get_index(self, action):
        """
        Get the index of a given card action.

        Args:
            action (str): The card action.

        Returns:
            int: The index of the card in the list of cards.
        """
        return self.CARDS.index(action)

    def get_number_of_actions(self):
        """
        Get the total number of possible card actions.

        Returns:
            int: The number of possible card actions.
        """
        return len(self.CARDS)
