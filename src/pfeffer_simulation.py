import random
from typing import List

import numpy as np
import tensorflow as tf
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import trajectory

SUITS_COLORS = {"S": "black", "C": "black", "H": "red", "D": "red"}

CARDS = [
    '9S', 'TS', 'JS', 'QS', 'KS', 'AS',
    '9H', 'TH', 'JH', 'QH', 'KH', 'AH',
    '9D', 'TD', 'JD', 'QD', 'KD', 'AD',
    '9C', 'TC', 'JC', 'QC', 'KC', 'AC'
]

BID_VALUES = [0, 4, 5, 6, 'pfeffer']
BID_SUITS = ['S', 'H', 'D', 'C', 'NT']  # NT represents No-Trump


class Game:

    def __init__(self, bid_model, play_model):
        self.bid_actions = BidActions()
        self.play_actions = PlayActions()

        self.players = [Player(i, bid_model, play_model, self.bid_actions, self.play_actions) for i in range(4)]

        self.game_state = {
            "hands": {player.player_id: [] for player in self.players},  # Empty hands at the start
            "score": [0, 0],  # Score starts at 0 for each team
            "current_trick": 0,  # No tricks have been played
            "played_cards": [[] for _ in range(6)],  # List of played cards for each trick
            "bidding_order": [0, 1, 2, 3],  # Bidding order might be fixed, or determined by some rule
            "all_bids": [],  # No bids have been made
            "winning_bid": (-1, -1, None),  # No winning bid yet, (bid, player_id, trump_suit)
            "lead_players": [None] * 6,  # No lead players yet
            "trick_winners": [None] * 6,
            "dealer_position": 0,
        }

        self.bid_experiences_cache = [[BidInput, None] for _ in range(4)]
        # Creates a 4x6x2 matrix to store the input and action for each experience
        self.play_experiences_cache = [[[PlayInput, None] for _ in range(6)] for _ in range(4)]

    def reset(self):
        """Resets the game to the initial state."""

        self.reset_round()

        # Reset score
        self.game_state["score"] = [0, 0]

    def reset_round(self):
        """Resets all round-specific variables and states."""

        # Reset cards in hands
        deck = ['9S', 'TS', 'JS', 'QS', 'KS', 'AS',
                '9H', 'TH', 'JH', 'QH', 'KH', 'AH',
                '9D', 'TD', 'JD', 'QD', 'KD', 'AD',
                '9C', 'TC', 'JC', 'QC', 'KC', 'AC']
        random.shuffle(deck)
        for i, player in enumerate(self.players):
            player.hand = deck[i * 6: (i + 1) * 6]

        # Reset current trick
        self.game_state["current_trick"] = 0

        # Reset played cards
        self.game_state["played_cards"] = [[] for _ in range(6)]

        # Reset bids
        self.game_state["all_bids"] = []
        self.game_state["winning_bid"] = (-1, -1, None)

        # Reset lead players for tricks
        self.game_state["lead_players"] = [None] * 6

        # Reset trick winners
        self.game_state["trick_winners"] = [None] * 6

        dp = self.game_state["dealer_position"]
        self.game_state["dealer_position"] = 0 if dp >= 3 else dp + 1

        # Reset experience caches
        self.bid_experiences_cache = [[BidInput, None] for _ in range(4)]
        self.play_experiences_cache = [[[PlayInput, None] for _ in range(6)] for _ in range(4)]

    def bid_round(self):
        """Conducts the bidding round of the game."""
        for player_id, player in enumerate(self.players):
            bid_input = BidInput.from_game_state(player_id, self.game_state)

            # Get bid from player
            bid, trump_suit = player.make_bid(bid_input)

            # Update game state with bid
            self.game_state["all_bids"].append(bid)

            # Check if this bid is higher than the current winning bid
            if bid > self.game_state["winning_bid"][0]:
                self.game_state["winning_bid"] = (bid, player_id, trump_suit)

            # Save experience to cache
            self.bid_experiences_cache[player_id][0] = bid_input
            self.bid_experiences_cache[player_id][1] = (bid, trump_suit)

    @staticmethod
    def generate_player_order(start_player):
        """Generates the order of players for a trick, starting with the specified player."""
        return [start_player, (start_player + 1) % 4, (start_player + 2) % 4, (start_player + 3) % 4]

    def play_round(self):
        """
        Plays a round by choosing an action using the Q-network.

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
            self.game_state["current_trick"] = trick
            for player_id in play_order:
                # Skip the partner's turn if the bid winner is playing alone
                if playing_alone and player_id == partner_id:
                    self.game_state["played_cards"][trick].append(
                        (partner_id, None))  # Record that the partner did not play a card
                    continue

                play_input = PlayInput.from_game_state(player_id, self.game_state)

                # Ask the player to make a play
                play = self.players[player_id].make_play(play_input)

                self.play_experiences_cache[player_id][trick][0] = play_input
                self.play_experiences_cache[player_id][trick][1] = play

                # Update the game state with the play
                self.game_state["played_cards"][trick].append((player_id, play))

            # Update the play order so the player who won the trick leads the next one
            trick_winner = self.evaluate_trick()

            if 0 < trick < 5:
                # noinspection PyTypeChecker
                self.game_state["lead_players"][trick + 1] = trick_winner
            # noinspection PyTypeChecker
            self.game_state["trick_winners"][trick] = trick_winner
            while play_order[0] != trick_winner:
                play_order.append(play_order.pop(0))

        # Update the score after the round
        (score_team1, score_team2) = self.evaluate_round()
        self.game_state["score"][0] += score_team1
        self.game_state["score"][1] += score_team2

        # Save results to bid_buffer
        for player_id in play_order:
            reward = score_team1 - score_team2 if player_id in [0, 2] else -(score_team1 - score_team2)
            self.players[player_id].save_to_bid_buffer(
                bid_input=self.bid_experiences_cache[player_id][0],
                action_taken=self.bid_experiences_cache[player_id][1],
                reward_received=reward
            )

        # Save results to play_buffer
        for trick in range(6):
            for player_id in play_order:
                reward = score_team1 - score_team2 if player_id in [0, 2] else -(score_team1 - score_team2)
                self.players[player_id].save_to_play_buffer(
                    play_input=self.play_experiences_cache[player_id][trick][0],
                    action_taken=self.play_experiences_cache[player_id][trick][1],
                    reward_received=reward
                )

    def evaluate_trick(self):
        """
        Evaluates the winner of a trick.

        Returns:
            int: player_id of the winner.
        """
        current_trick = self.game_state["current_trick"]
        cards_played_in_last_trick = self.game_state["played_cards"][current_trick]
        lead_suit = cards_played_in_last_trick[0][1][-1]  # Suit of the first card played in the trick

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
        winning_card = max(cards_played_in_last_trick, key=lambda x: card_value(x[1]))

        # Return the player_id of the winner
        return winning_card[0]

    def evaluate_round(self):
        # Extract the winning bid information
        bid_winner = self.game_state["winning_bid"][1]
        bid_value = self.game_state["winning_bid"][0]

        # Determine the teams (e.g., players 0 and 2 are on one team, 1 and 3 are on the other)
        team1 = [0, 2]
        team2 = [1, 3]

        # Track the number of tricks won by each team
        tricks_won_team1 = 0
        tricks_won_team2 = 0

        # Calculate the number of tricks won by each team
        for trick_winner in self.game_state["trick_winners"]:
            if trick_winner in team1:
                tricks_won_team1 += 1
            else:
                tricks_won_team2 += 1

        # Determine the winning team and calculate the new score
        bidding_team_tricks_required = 6 if bid_value == 'pfeffer' else bid_value
        if bid_winner in team1:
            if tricks_won_team1 >= bidding_team_tricks_required:
                score_team1 = 12 if bid_value == 'pfeffer' else tricks_won_team1
            else:
                score_team1 = -12 if bid_value == 'pfeffer' else -5
            if tricks_won_team2 > 0:
                score_team2 = tricks_won_team2
            else:
                score_team2 = -5
        else:
            if tricks_won_team1 > 0:
                score_team1 = tricks_won_team2
            else:
                score_team1 = -5
            if tricks_won_team2 >= bidding_team_tricks_required:
                score_team2 = 12 if bid_value == 'pfeffer' else tricks_won_team1
            else:
                score_team2 = -12 if bid_value == 'pfeffer' else -5

        return score_team1, score_team2

    def play_game(self):
        """Plays a full game."""
        self.reset()
        while not self.game_over():
            self.reset_round()
            self.bid_round()
            self.play_round()

    def game_over(self):
        """Checks if the game is over."""
        # The game is over if either team's score is 32 or more
        return any(score >= 32 for score in self.game_state["score"])


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
            'winning_bid_encoding': tf.TensorSpec(shape=(14,), dtype=tf.int32),
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


class BidActions:
    """Represents the possible bid actions in Pfeffer."""

    @staticmethod
    def get_bid_value_action(index):
        """
        Returns the bid value action corresponding to a given index.

        Args:
            index (int): The index of the action.

        Returns:
            int or str: The bid value action corresponding to the index.
        """
        return BID_VALUES[index]

    @staticmethod
    def get_bid_suit_action(index):
        """
        Returns the bid value action corresponding to a given index.

        Args:
            index (int): The index of the action.

        Returns:
            str: The bid suit action corresponding to the index.
        """
        return BID_SUITS[index]

    @staticmethod
    def get_bid_value_index(action):
        """
        Returns the index corresponding to a given bid value action.

        Args:
            action int or str: The bid value action.

        Returns:
            int: The index corresponding to the bid value action.
        """
        return BID_VALUES.index(action)

    @staticmethod
    def get_bid_suit_index(action):
        """
        Returns the index corresponding to a given bid suit action.

        Args:
            action str: The bid suit action.

        Returns:
            int: The index corresponding to the bid suit action.
        """
        return BID_SUITS.index(action)

    @staticmethod
    def get_number_of_bid_value_actions():
        """
        Returns the total number of bid value actions in the bid value action space.

        Returns:
            int: The total number of bid value actions.
        """
        return len(BID_VALUES)

    @staticmethod
    def get_number_of_bid_suit_actions():
        """
        Returns the total number of bid suit actions in the bid suit action space.

        Returns:
            int: The total number of bid suit actions.
        """
        return len(BID_SUITS)

    @staticmethod
    def is_bid(action):
        """
        Checks if an action is a bid.

        Args:
            action (int or str): The action.

        Returns:
            bool: True if the action is a bid, False otherwise.
        """
        return action in BID_VALUES

    @staticmethod
    def is_suit_choice(action):
        """
        Checks if an action is a suit choice.

        Args:
            action (str): The action.

        Returns:
            bool: True if the action is a suit choice, False otherwise.
        """
        return action in BID_SUITS


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
        possible_bids = [0, 4, 5, 6, 'pfeffer']
        possible_suits = ['S', 'H', 'D', 'C', 'no-trump']
        possible_players = [0, 1, 2, 3]

        bid_quantity_encoding = [0] * len(possible_bids)
        bid_quantity_encoding[possible_bids.index(bid_quantity)] = 1

        player_encoding = [0] * len(possible_players)
        player_encoding[possible_players.index(player)] = 1

        bid_suit_encoding = [0] * len(possible_suits)
        bid_suit_encoding[possible_suits.index(bid_suit)] = 1

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
    def decode_list_of_players(encoded_players):
        """
        Decodes an encoded tensor of players to a list of players.

        Args:
            encoded_players (tf.Tensor): An encoded tensor of players.

        Returns:
            list: The decoded list of players.
        """
        lead_players = []
        for i in range(6):
            player_encoded = encoded_players[i * 5:i * 5 + 5]
            player = PlayInput.decode_one_hot(player_encoded, [-1, 0, 1, 2, 3])
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
            list: A list of 4-element one-hot encodings of the bidding order.
        """
        possible_players = [0, 1, 2, 3]
        bidding_order_encoding = []

        for player in bidding_order:
            player_encoding = [0] * len(possible_players)
            player_encoding[possible_players.index(player)] = 1
            bidding_order_encoding.append(player_encoding)

        return bidding_order_encoding  # Returning a list of lists instead of a flattened list

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
            'winning_bid_encoding': tf.constant(winning_bid_encoding, dtype=tf.int32),
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
        bidding_order = [PlayInput.decode_one_hot(player_encoded, [0, 1, 2, 3]) for player_encoded in
                         encoded_state['bidding_order']]

        # Decode all_bids
        all_bids = [
            PlayInput.decode_one_hot(
                encoded_state['all_bids'][i:i + 5],
                [0, 4, 5, 6, 'pfeffer']
            ) for i in range(0, len(encoded_state['all_bids']), 5)
        ]

        # Decode winning_bid
        winning_bid = PlayInput.decode_winning_bid(encoded_state['winning_bid_encoding'])

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


class PlayActions:
    """
    Represents the possible play actions in Pfeffer.
    """

    @staticmethod
    def get_action(index):
        """
        Get the card action based on the given index.

        Args:
            index (int): The index of the action in the list of cards.

        Returns:
            str: The card represented by the given index.
        """
        return CARDS[index]

    @staticmethod
    def get_index(action):
        """
        Get the index of a given card action.

        Args:
            action (str): The card action.

        Returns:
            int: The index of the card in the list of cards.
        """
        return CARDS.index(action)

    @staticmethod
    def get_number_of_actions():
        """
        Get the total number of possible card actions.

        Returns:
            int: The number of possible card actions.
        """
        return len(CARDS)
