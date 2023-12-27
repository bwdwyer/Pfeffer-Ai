from src.models import CARDS


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
