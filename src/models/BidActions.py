from src.models import BID_VALUES, BID_SUITS


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
