#!/usr/bin/env python3
import random
import math

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR

import time


class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        """
        Function that generates the loop of the game. In each iteration
        the human plays through the keyboard and send
        this to the game through the sender. Then it receives an
        update of the game through receiver, with this it computes the
        next movement.
        :return:
        """

        while True:
            # send message to game that you are ready
            msg = self.receiver()
            if msg["game_over"]:
                return


class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        :return:
        """

        # Generate game tree object
        first_msg = self.receiver()
        # Initialize your minimax model
        model = self.initialize_model(initial_data=first_msg)

        while True:
            msg = self.receiver()

            # Create the root node of the game tree
            node = Node(message=msg, player=0)

            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move = self.search_best_next_move(
                model=model, initial_tree_node=node)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})

    def initialize_model(self, initial_data):
        """
        Initialize your minimax model
        :param initial_data: Game data for initializing minimax model
        :type initial_data: dict
        :return: Minimax model
        :rtype: object

        Sample initial data:
        { 'fish0': {'score': 11, 'type': 3},
          'fish1': {'score': 2, 'type': 1},
          ...
          'fish5': {'score': -10, 'type': 4},
          'game_over': False }

        Please note that the number of fishes and their types is not fixed between test cases.
        """
        # EDIT THIS METHOD TO RETURN A MINIMAX MODEL ###
        return None

    def calculate_heuristics(self, node):
        state = node.state
        r = state.player_scores[0]
        op_score = state.player_scores[1]
        su = 0
        for i in state.fish_positions:
            distance = self.calculate_distance(state.fish_positions[i], state.hook_positions[0])
            if distance == 0 and len(state.fish_positions)==1:
                return float('inf')
            su = max(su, state.fish_scores[i]*math.exp(-distance))
        return su + (r - op_score)*2

    def calculate_distance(self, fish_positions, hook_positions):
        y_distance = abs(fish_positions[1] - hook_positions[1])
        x_distance = min(abs(fish_positions[0] - hook_positions[0]), 20 - abs(fish_positions[0] - hook_positions[0]))
        distance = x_distance + y_distance
        return distance

    def alpha_beta_prunning(self, node, state, depth, alpha, beta, player, start,visited_nodes):
        if time.time() - start > 0.055:
            raise TimeoutError
        else:
            k = self.hash_key(state)
            if k in visited_nodes and visited_nodes[k][0] >= depth:
                return visited_nodes[k][1]
            children = node.compute_and_get_children()
            children.sort(key=self.calculate_heuristics, reverse = True)
            if depth == 0 or len(children) == 0:
                v = self.calculate_heuristics(node)
            elif player == 0:
                v = float('-inf')
                for child in children:
                    v = max(v, self.alpha_beta_prunning(child, child.state, depth - 1, alpha, beta, 1, start,visited_nodes))
                    alpha = max(alpha, v)
                    if alpha >= beta:
                        break
            else:
                v = float('inf')
                for child in children:
                    v = min(v, self.alpha_beta_prunning(child, child.state, depth - 1, alpha, beta, 0, start,visited_nodes))
                    beta = min(beta, v)
                    if beta <= alpha:
                        break

            key = self.hash_key(state)
            visited_nodes.update({key:[depth,v]})
        return v

    def hash_key(self,state):
        posdic = self.encode_fish(state)
        return str(state.get_hook_positions())+str(posdic)

    def encode_fish(self,state):
        pos_dic = dict()
        for pos,score in zip(state.get_fish_positions().items(),state.get_fish_scores().items()):
            score = score[1]
            pos = pos[1]
            x = pos[0]
            y = pos[1]
            k = str(x) + str(y)
            pos_dic.update({k:score})
        return pos_dic


    def concrete_depth_search(self, node, depth, start,visited_nodes):
        alpha = float('-inf')
        beta = float('inf')
        children = node.compute_and_get_children()
        #random.shuffle(children)
        heuristics = []
        for child in children:
            v = self.alpha_beta_prunning(child, child.state, depth, alpha, beta, 1, start,visited_nodes)
            heuristics.append(v)
        index = heuristics.index(max(heuristics))
        return children[index].move

    def search_best_next_move(self, model, initial_tree_node):
        """
        Use your minimax model to find best possible next move for player 0 (green boat)
        :param model: Minimax model
        :type model: object
        :param initial_tree_node: Initial game tree node
        :type initial_tree_node: game_tree.Node
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """

        # EDIT THIS METHOD TO RETURN BEST NEXT POSSIBLE MODE FROM MINIMAX MODEL ###

        # NOTE: Don't forget to initialize the children of the current node
        #       with its compute_and_get_children() method!

        start = time.time()
        depth = 0
        flag = True
        visited_nodes = dict()
        while flag == True:
            try:
                move = self.concrete_depth_search(initial_tree_node, depth, start,visited_nodes)
                depth = depth + 1
                best_move = move
            except:
                flag = False
        return ACTION_TO_STR[best_move]