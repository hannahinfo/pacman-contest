# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from captureAgents import CaptureAgent
from game import Directions
from util import nearestPoint
from util import manhattanDistance


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
    
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        # Initialize instance variables to keep track of food collected and saved
        self.collected_food = 0
        self.saved_food = 0
        self.is_scared = False

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        self.update_food_counts(game_state)

        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action
        
        if random.random() < 0.2:  # 20% chance of exploration
            best_action = random.choice(actions)

        return random.choice(best_actions)

    def get_features(self, game_state, action):
        """ 
        Returns a dictionary of features for the state 
        """

        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(self.get_food(successor).as_list())

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        # Additional features for offensive behavior
        my_state = successor.get_agent_state(self.index)
        #features['is_pacman'] = int(my_state.is_pacman)
        features['ghost_distance'] = self.get_distance_to_nearest_ghost(successor, action)
        features['distance_to_start'] = self.get_distance_to_start(successor)

        if action == Directions.STOP: features['stop'] = 1

        return features
    
    def update_food_counts(self, game_state):
        """
        Update the counts of collected and saved food based on the current game state.
        """
        my_state = game_state.get_agent_state(self.index)

        if not my_state.is_pacman:
            # Calculate saved food when the agent is a ghost
            saved_food_now = 20 - len(self.get_food(game_state).as_list())
            self.saved_food = saved_food_now
            self.collected_food = 0
        else:
            # Calculate collected food when the agent is a Pacman
            collected_food_now = 20 - self.saved_food - len(self.get_food(game_state).as_list())
            self.collected_food = collected_food_now


    def get_weights(self, game_state, action):
        """ 
        Returns a dictionary of weights for features 
        """
        features = self.get_features(game_state, action)
        
        if features['ghost_distance'] <= 5 or self.collected_food >= 5:
            return {
            'distance_to_food': -1,
            'successor_score': 100,
            'ghost_distance': 2,
            'distance_to_start': -2,
            'stop': -10
            }
        else: return {
            'distance_to_food': -1,
            'successor_score': 100,
            'ghost_distance': 1,
            'distance_to_start': -0.5,
            'stop': -10
            }
    

    def get_distance_to_nearest_ghost(self, game_state, action):
        """
        Calculate the distance to the nearest ghost.
        """
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()

        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if not(a.is_pacman) and a.get_position() is not None]
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
        else:
            dists = [10000]

        return min(dists)

    def get_distance_to_start(self, game_state):
        """ Calculate the distance to the starting point. """
        
        return self.get_maze_distance(game_state.get_agent_position(self.index), self.start)




class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """
    def choose_action(self, game_state):
        self.update_is_scared(game_state)

        actions = game_state.get_legal_actions(self.index)

        # Evaluate each action based on features and weights
        action_scores = [(action, self.evaluate(game_state, action)) for action in actions]

        # Choose the action with the highest score
        best_action = max(action_scores, key=lambda x: x[1])[0]
        if random.random() < 0.1:  # 10% chance of exploration
            best_action = random.choice(actions)

        return best_action
    
    def update_is_scared(self, game_state):
        my_state = game_state.get_agent_state(self.index)
        self.is_scared = my_state.scared_timer > 0

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        features['is_scared'] = my_state.scared_timer > 0

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        if self.is_scared:
            return {'num_invaders': -1000, 'on_defense': 0, 'invader_distance': -10, 'stop': -100, 'reverse': -2}
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}

