import random

from src.Player import Player
from src.models import SUITS_COLORS
from src.models.BidActions import BidActions
from src.models.BidInput import BidInput
from src.models.PlayInput import PlayInput


class Game:

    def __init__(self, bid_model, play_model):
        self.players = [Player(i, bid_model, play_model) for i in range(4)]

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
        self.game_state["hands"].clear()
        for i, player in enumerate(self.players):
            hand = deck[i * 6: (i + 1) * 6]
            self.game_state["hands"][player.player_id] = hand

        # Reset current trick
        self.game_state["current_trick"] = 0

        # Reset played cards
        self.game_state["played_cards"] = [[] for _ in range(6)]

        # Reset bids
        self.game_state["all_bids"] = [None for _ in range(4)]
        self.game_state["winning_bid"] = (0, -1, None)

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
            self.game_state["all_bids"][player_id] = bid

            # Check if this bid is higher than the current winning bid
            if BidActions.get_bid_value_index(bid) > BidActions.get_bid_value_index(self.game_state["winning_bid"][0]):
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
        bid_suit = self.game_state["winning_bid"][2]

        # Determine if the bid winner is playing alone (i.e., bid 'pfeffer')
        playing_alone = bid_value == 'pfeffer'

        # Get the partner of the bid winner (the player across)
        partner_id = (bid_winner + 2) % 4

        # Get a copy of the bidding order and rotate it so the bid winner goes first
        play_order = self.game_state["bidding_order"][:]
        while play_order[0] != bid_winner:
            play_order.append(play_order.pop(0))

        print(f"Staring round. \nBid winner: {bid_winner}.\nBid value: {bid_value}\nBid suit: {bid_suit}")
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
                card_played = self.players[player_id].make_play(play_input)

                # Save experience
                # print(f"Saving play experience for {player_id}")
                self.play_experiences_cache[player_id][trick][0] = play_input
                self.play_experiences_cache[player_id][trick][1] = card_played

                # Update the game state with the play
                self.game_state["played_cards"][trick].append((player_id, card_played))
                # print(f"Removing {card_played} from {player_id}, hand: {self.game_state['hands'][player_id]}")
                self.game_state["hands"][player_id].remove(card_played)

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
        left_bauer_suit = None if trump_suit == 'NT' else \
            [s for s in opposite_color_suits[SUITS_COLORS[trump_suit]] if s != trump_suit][0]

        # Rank the cards based on their value
        def card_value(card):
            if card is None:
                return -1

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
        round_iteration = 0
        while not self.game_over() and round_iteration < 64:
            round_iteration += 1
            self.reset_round()
            self.bid_round()
            self.play_round()
            print(f"Finished round {round_iteration} {self.game_state}")

    def game_over(self):
        """Checks if the game is over."""
        # The game is over if either team's score is 32 or more
        return any(score >= 32 for score in self.game_state["score"])
