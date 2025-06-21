import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()

from catanatron import Player
from catanatron.cli import register_cli_player
from pyDecision.algorithm import topsis_method
import numpy as np
from catanatron.models.actions import ActionType
import io
import contextlib

class Catalina(Player):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        # Define criteria for MCDA
        self.criteria = ['vp_yield', 'resource_gain', 'expansion', 'leverage']

    def evaluate_build(self, game, action):
        """
        Extracts a feature vector for build actions:
        [vp_yield, resource_gain, expansion, leverage]
        """
        # Determine node or edge target
        node_id = action.value if action.action_type in (ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY) else None

        # 1. VP yield: 1 for settlement, 2 for city, else 0
        if action.action_type == ActionType.BUILD_SETTLEMENT:
            vp_yield = 1
        elif action.action_type == ActionType.BUILD_CITY:
            vp_yield = 2
        else:
            vp_yield = 0

        # 2. Resource gain: unique adjacent resources at the node
        resource_gain = 0.0
        if node_id is not None:
            resources = []
            for tile in game.state.board.map.tiles.values():
                if node_id in tile.nodes.values():
                    res = getattr(tile, 'resource', None)
                    if res is not None:
                        resources.append(res)
            resource_gain = len(set(resources)) / len(resources) if resources else 0.0

        # 3. Expansion: fraction of buildable edges from this node (max 3)
        expansion = 0.0
        if node_id is not None:
            buildable = game.state.board.buildable_edges(self.color)
            edges_here = [edge for edge in buildable if node_id in edge]
            expansion = len(edges_here) / 3.0

        # 4. Leverage: placeholder for future heuristic
        leverage = 0.0

        return [vp_yield, resource_gain, expansion, leverage]

    def decide(self, game, playable_actions):
        # Filter build actions (road, settlement, city)
        build_actions = [a for a in playable_actions if a.action_type.name.startswith('BUILD_')]
        if build_actions:
            # Build decision matrix (alternatives x criteria)
            matrix = np.array([self.evaluate_build(game, a) for a in build_actions], dtype=float)
            # Initial weights and impacts
            weights = np.ones(len(self.criteria))
            impacts = ['+' for _ in self.criteria]

            # Drop criteria with zero total (to avoid division by zero in TOPSIS)
            col_sums = matrix.sum(axis=0)
            valid = col_sums != 0
            if not np.any(valid):
                # Fallback: choose alternative with highest sum of features
                totals = matrix.sum(axis=1)
                idx = int(np.nanargmax(totals))
                return build_actions[idx]

            # Filter out zero-variance columns
            mat = matrix[:, valid]
            w = weights[valid]
            imps = [impacts[i] for i, ok in enumerate(valid) if ok]

            # Run TOPSIS from pyDecision, suppressing its internal printouts
            with contextlib.redirect_stdout(io.StringIO()):
                result = topsis_method(mat, w, imps)
            # Extract closeness coefficients
            if isinstance(result, (tuple, list)):
                scores = result[0]
            else:
                scores = result

                        # Closeness coefficient is the score value itself for ranking
            # Handle all-NaN scores by falling back to highest sum of features
            if np.all(np.isnan(scores)):
                totals = matrix.sum(axis=1)
                idx = int(np.nanargmax(totals))
            else:
                idx = int(np.nanargmax(scores))
            return build_actions[idx]

        # Fallback: first available action
        return playable_actions[0]

# Register Catalina as a CLI player
register_cli_player("Catalina", Catalina)
