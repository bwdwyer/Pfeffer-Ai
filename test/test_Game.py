import unittest
from unittest import TestCase
from unittest.mock import Mock

import numpy as np
import tensorflow as tf

from src.Game import *


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
        self.assertEqual([None, None, None, None], game.game_state["all_bids"])
        assert (0, -1, None) == game.game_state["winning_bid"]
        assert [None] * 6 == game.game_state["lead_players"], game.game_state["lead_players"]
        assert [None] * 6 == game.game_state["trick_winners"], game.game_state["trick_winners"]

        # Verify that the hands have been reset (if you want to test this, you may need additional logic)
        for player in game.players:
            self.assertEqual(6, len(game.game_state["hands"][player.player_id]))

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
        self.assertEqual([None, None, None, None], game.game_state["all_bids"])
        assert (0, -1, None) == game.game_state["winning_bid"]
        assert [None] * 6 == game.game_state["lead_players"], game.game_state["lead_players"]
        assert [None] * 6 == game.game_state["trick_winners"], game.game_state["trick_winners"]

        # Verify that the score has not been changed
        assert original_score == game.game_state["score"]

        # Verify that the hands have been reset (if you want to test this, you may need additional logic)
        for player in game.players:
            self.assertEqual(6, len(game.game_state["hands"][player.player_id]))

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
        bid_model.predict.return_value = [np.zeros(10)]
        play_model = Mock()
        play_model.predict.return_value = [np.zeros(24)]

        game = Game(bid_model, play_model)
        game.reset()

        # Bid Round
        game.bid_round()
        self.assertEqual(3, game.game_state["winning_bid"][1])  # Bid winner
        self.assertEqual(4, game.game_state["winning_bid"][0])  # Winning bid value
        self.assertEqual('S', game.game_state["winning_bid"][2])  # Winning bid suit

        # Play Round
        game.play_round()

        # Each player should have:
        #   1 bid experience
        #   6 play experiences
        for i, player in enumerate(game.players):
            self.assertEqual(1, player.bid_replay_buffer.num_frames())
            self.assertEqual(6, player.play_replay_buffer.num_frames())

            trajectories, _ = player.play_replay_buffer.get_next()
            print(trajectories.observation)

    def test_bid_and_play_round_with_sequentials(self):
        bid_model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(24 + 1 + 2 + 15,), name="input"),  # 42
            tf.keras.layers.Dense(64, activation='relu', name="hidden1"),
            tf.keras.layers.Dense(64, activation='relu', name="hidden2"),
            tf.keras.layers.Dense(10, name="output"),
        ], name="bid_model")
        bid_model.compile(optimizer='adam', loss='mse')
        play_model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(4 + 24 + (6 * 4 * 24) + 16 + 20 + 14 + 30 + 30 + 6 + 2,), name="input"),  # 722
            tf.keras.layers.Dense(64, activation='relu', name="hidden1"),
            tf.keras.layers.Dense(64, activation='relu', name="hidden2"),
            tf.keras.layers.Dense(24, name="output"),
        ], name="play_model")
        play_model.compile(optimizer='adam', loss='mse')

        game = Game(bid_model, play_model)
        game.reset()
        game.reset_round()

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

    def test_play_game(self):
        bid_model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(24 + 1 + 2 + 15,), name="input"),
            tf.keras.layers.Dense(64, activation='relu', name="hidden1"),
            tf.keras.layers.Dense(64, activation='relu', name="hidden2"),
            tf.keras.layers.Dense(10, name="output"),
        ], name="bid_model")
        bid_model.compile(optimizer='adam', loss='mse')
        play_model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(4 + 24 + (6 * 4 * 24) + 16 + 20 + 14 + 30 + 30 + 6 + 2,), name="input"),  # 722
            tf.keras.layers.Dense(64, activation='relu', name="hidden1"),
            tf.keras.layers.Dense(64, activation='relu', name="hidden2"),
            tf.keras.layers.Dense(24, name="output"),
        ], name="play_model")
        play_model.compile(optimizer='adam', loss='mse')

        game = Game(bid_model, play_model)
        game.play_game()
