import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()

from catanatron import Player
from catanatron.cli import register_cli_player
from pyDecision.algorithm import topsis_method
import numpy as np
import io
import contextlib

# Feature helpers
from catanatron.features import number_probability, get_player_expandable_nodes
from catanatron.models.actions import ActionType

class Catalina(Player):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.criteria = [
            'vp_yield',
            'pip_probability',
            'diversity',
            'accessibility'
        ]

    def evaluate_build(self, game, action):
        """
        Builds a feature vector for settlement/city placement:
        [vp_yield, pip_probability, diversity, accessibility]
        """
        board = game.state.board
        vp_yield = 0.0
        pip_prob = 0.0
        diversity = 0.0
        accessibility = 0.0

        if action.action_type in (ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY):
            node_id = action.value
            # 1. Immediate VP: settlement=1, city=2
            vp_yield = 1.0 if action.action_type == ActionType.BUILD_SETTLEMENT else 2.0

            # 2. Pip probability & diversity
            resources = []
            pip_sum = 0.0
            for tile in board.map.tiles.values():
                # Skip non-resource or port tiles
                if not hasattr(tile, 'number') or not hasattr(tile, 'resource'):
                    continue
                if node_id in tile.nodes.values() and tile.resource is not None:
                    resources.append(tile.resource)
                    pip_sum += number_probability(tile.number)
            pip_prob = pip_sum
            diversity = len(set(resources)) / len(resources) if resources else 0.0

            # 3. Accessibility: reachable node
            access_nodes = get_player_expandable_nodes(game, self.color)
            accessibility = 1.0 if node_id in access_nodes else 0.0

        return [vp_yield, pip_prob, diversity, accessibility]

    def decide(self, game, playable_actions):
        # Filter only settlement/city builds
        build_actions = [a for a in playable_actions if a.action_type in (ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY)]
        if build_actions:
            matrix = np.array([self.evaluate_build(game, a) for a in build_actions], dtype=float)
            weights = np.array([2.0, 1.5, 1.0, 0.5])
            impacts = ['+' for _ in self.criteria]

            valid = matrix.sum(axis=0) != 0
            if not np.any(valid):
                idx = int(np.nanargmax(matrix.sum(axis=1)))
                return build_actions[idx]
            mat = matrix[:, valid]
            w = weights[valid]
            imps = [impacts[i] for i, ok in enumerate(valid) if ok]

            with contextlib.redirect_stdout(io.StringIO()):
                res = topsis_method(mat, w, imps)
            scores = res[0] if isinstance(res, (tuple, list)) else res

            if np.all(np.isnan(scores)):
                idx = int(np.nanargmax(matrix.sum(axis=1)))
            else:
                idx = int(np.nanargmax(scores))
            return build_actions[idx]

        # Fallback for other actions
        return playable_actions[0]

register_cli_player("Catalina", Catalina)
