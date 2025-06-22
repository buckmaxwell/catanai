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
from random import choice

# Feature helpers
from catanatron.features import number_probability, get_player_expandable_nodes
from catanatron.models.actions import ActionType

class Catalina(Player):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        # Define criteria including road potential
        self.criteria = [
            'vp_yield',
            'pip_probability',
            'diversity',
            'accessibility',
            'road_potential'
        ]

    def evaluate_build(self, game, action):
        """
        Builds a feature vector for settlement/city or road actions:
        [vp_yield, pip_probability, diversity, accessibility, road_potential]
        """
        board = game.state.board
        # Initialize features
        vp_yield = 0.0
        pip_prob = 0.0
        diversity = 0.0
        accessibility = 0.0
        road_potential = 0.0

        # Settlement or city placement
        if action.action_type in (ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY):
            node_id = action.value
            # 1) Immediate VP: settlement=1, city=2
            vp_yield = 1.0 if action.action_type == ActionType.BUILD_SETTLEMENT else 2.0
            # 2) Pip probability & diversity
            resources = []
            pip_sum = 0.0
            for tile in board.map.tiles.values():
                if not hasattr(tile, 'number') or not hasattr(tile, 'resource'):
                    continue
                if node_id in tile.nodes.values() and tile.resource is not None:
                    resources.append(tile.resource)
                    pip_sum += number_probability(tile.number)
            pip_prob = pip_sum
            diversity = len(set(resources)) / len(resources) if resources else 0.0
            # 3) Accessibility: reachable node
            access_nodes = get_player_expandable_nodes(game, self.color)
            accessibility = 1.0 if node_id in access_nodes else 0.0

        # Road building potential
        elif action.action_type == ActionType.BUILD_ROAD:
            edge = action.value  # tuple of two node_ids
            # Road potential: small base plus fraction of endpoints in reach
            access_nodes = get_player_expandable_nodes(game, self.color)
            reachable_ends = sum(1 for nd in edge if nd in access_nodes)
            road_potential = 0.2 + (reachable_ends / 2.0)

        # Return feature vector
        return [vp_yield, pip_prob, diversity, accessibility, road_potential]

    def decide(self, game, playable_actions):
        # Include roads, settlements, and cities in MCDA
        build_actions = [
            a for a in playable_actions
            if a.action_type in (
                ActionType.BUILD_SETTLEMENT,
                ActionType.BUILD_CITY,
                ActionType.BUILD_ROAD
            )
        ]
        # DEBUG: inspect features for each potential build
        for a in build_actions:
            feats = self.evaluate_build(game, a)
            print(f"DEBUG: action={a.action_type.name}, feats={feats}")

        if build_actions:
            # Build decision matrix
            matrix = np.array([
                self.evaluate_build(game, a) for a in build_actions
            ], dtype=float)
                        # Weights: [vp_yield, pip_probability, diversity, accessibility, road_potential]
            weights = np.array([2.0, 1.5, 1.0, 0.5, 1.0])
            impacts = ['+' for _ in self.criteria]

            # Only include criteria with non-zero weight
            valid = weights != 0
            mat = matrix[:, valid]
            # Filter weights and impacts accordingly
            w = weights[valid]
            imps = [impacts[i] for i, ok in enumerate(valid) if ok]

            # Run TOPSIS silently
            with contextlib.redirect_stdout(io.StringIO()):
                res = topsis_method(mat, w, imps)
            scores = res[0] if isinstance(res, (tuple, list)) else res

            # Choose best or fallback to sum
            if np.all(np.isnan(scores)):
                idx = int(np.nanargmax(matrix.sum(axis=1)))
            else:
                idx = int(np.nanargmax(scores))
            return build_actions[idx]

        # Fallback: first available action
        #return playable_actions[0]
        # return a RANDOM action if no build actions are available
        return choice(playable_actions)

# Register Catalina
register_cli_player("Catalina", Catalina)
