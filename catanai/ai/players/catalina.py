from catanatron import Player
from catanatron.cli import register_cli_player

class Catalina(Player):
    def decide(self, game, playable_actions):
        """Should return one of the playable_actions.

        Args:
            game (Game): complete game state. read-only.
            playable_actions (Iterable[Action]): options to choose from
        Return:
            action (Action): Chosen element of playable_actions
        """
        # ===== YOUR CODE HERE =====
        return playable_actions[0]  # type: ignore
        # ===== END YOUR CODE =====

register_cli_player("Catalina", Catalina)
