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
        # Define MCDA criteria including dev card potential
        self.criteria = [
            'vp_yield',
            'pip_probability',
            'diversity',
            'accessibility',
            'road_potential',
            'dev_potential'
        ]

    def evaluate_build(self, game, action):
        """
        Computes feature vector for an action: 
        [vp_yield, pip_probability, diversity, accessibility, road_potential, dev_potential]
        """
        board = game.state.board
        # Base feature values
        vp_yield = pip_prob = diversity = accessibility = road_potential = dev_potential = 0.0

        # Settlement or city placement
        if action.action_type in (ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY):
            node_id = action.value
            # Immediate VPs
            vp_yield = 1.0 if action.action_type == ActionType.BUILD_SETTLEMENT else 2.0
            # Pip probability & diversity
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
            # Accessibility
            access_nodes = get_player_expandable_nodes(game, self.color)
            accessibility = 1.0 if node_id in access_nodes else 0.0

        # Road building
        elif action.action_type == ActionType.BUILD_ROAD:
            edge = action.value
            access_nodes = get_player_expandable_nodes(game, self.color)
            reachable_ends = sum(1 for nd in edge if nd in access_nodes)
            road_potential = 0.2 + (reachable_ends / 2.0)

        # Development card purchase
        elif action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
            # Approximate expected dev-card utility
            dev_potential = 0.3

        return [vp_yield, pip_prob, diversity, accessibility, road_potential, dev_potential]

    def decide(self, game, playable_actions):
        # Consider build and dev-card actions
        options = [a for a in playable_actions if a.action_type in (
            ActionType.BUILD_SETTLEMENT,
            ActionType.BUILD_CITY,
            ActionType.BUILD_ROAD,
            ActionType.BUY_DEVELOPMENT_CARD
        )]
        # If none, pick random
        if not options:
            return choice(playable_actions)

        # Build decision matrix
        matrix = np.array([self.evaluate_build(game, a) for a in options], dtype=float)
        # Weights for each criterion
        weights = np.array([2.0, 1.5, 1.0, 0.5, 1.0, 0.5])
        impacts = ['+' for _ in self.criteria]

        # Mask zero-weight criteria
        valid = weights != 0
        mat = matrix[:, valid]
        w = weights[valid]
        imps = [impacts[i] for i, ok in enumerate(valid) if ok]

        # Compute TOPSIS scores
        with contextlib.redirect_stdout(io.StringIO()):
            res = topsis_method(mat, w, imps)
        scores = res[0] if isinstance(res, (tuple, list)) else res

        # Choose best alternative
        if np.all(np.isnan(scores)):
            idx = int(np.nanargmax(matrix.sum(axis=1)))
        else:
            idx = int(np.nanargmax(scores))
        return options[idx]

# Register Catalina
register_cli_player("Catalina", Catalina)
