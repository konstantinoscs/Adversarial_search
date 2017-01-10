# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        capsules = successorGameState.getCapsules()
        score = successorGameState.getScore()

        if newPos in currentGameState.getGhostPositions():
            return -50

        if newPos in currentGameState.getFood().asList():
            score += 5

        score += manhattanDistance(newPos, min(successorGameState.getGhostPositions()))
        score += sum(newScaredTimes)
        if newFood:
            score -= manhattanDistance(newPos, max(newFood))
        if capsules:
            score -= manhattanDistance(newPos, max(capsules))

        return score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        actions = gameState.getLegalActions(0)
        first_state = gameState.generateSuccessor(0, actions[0])
        cost = self.minimax(first_state, self.depth, 1)
        move = actions[0]

        for i in range(1, len(actions)):
            state = gameState.generateSuccessor(0, actions[i])
            newcost = self.minimax(state, self.depth, 1)
            if newcost > cost:
                cost = newcost
                move = actions[i]
        return move

    def minimax(self, gameState, depth, agentIndex):

        if not depth:
            return self.evaluationFunction(gameState)
        else:
            possible_actions = gameState.getLegalActions(agentIndex)
            states = [gameState.generateSuccessor(agentIndex, action) for action in possible_actions]
            if not possible_actions:
                return self.evaluationFunction(gameState)
            if not agentIndex:
                return max([self.minimax(state, depth, agentIndex+1) for state in states])
            elif agentIndex + 1 == gameState.getNumAgents():
                return min([self.minimax(state, depth - 1, 0) for state in states])
            elif agentIndex + 1 < gameState.getNumAgents():
                return min([self.minimax(state, depth, agentIndex+1) for state in states])



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        actions = gameState.getLegalActions(0)
        move = None
        v = float("-inf")
        a = float("-inf")
        b = float("inf")
        for i in range(len(actions)):
            state = gameState.generateSuccessor(0, actions[i])
            cost = self.alpha_beta_pr(state, a, b, self.depth,  1)
            if cost > v:
                move = actions[i]
                v = cost
            if v > b:
                break
            a = max(a, v)
        return move

    def alpha_beta_pr(self, gameState, a, b, depth, agentIndex):

        if not depth:
            return self.evaluationFunction(gameState)
        else:
            if not agentIndex:
                return self.maxval(gameState, a, b, depth, agentIndex)
            else:
                return self.minval(gameState, a, b, depth, agentIndex)

    def maxval(self, gameState, a, b, depth, agentIndex):
        possible_actions = gameState.getLegalActions(agentIndex)
        if not possible_actions:
            return self.evaluationFunction(gameState)
        v = float("-inf")
        for i in range(len(possible_actions)):
            state = gameState.generateSuccessor(agentIndex, possible_actions[i])
            v = max(v, self.alpha_beta_pr(state, a, b, depth, agentIndex+1))
            if v > b:
                return v
            a = max(a, v)
        return v

    def minval(self, gameState, a, b, depth, agentIndex):
        possible_actions = gameState.getLegalActions(agentIndex)
        if not possible_actions:
            return self.evaluationFunction(gameState)
        v = float("inf")
        for i in range(len(possible_actions)):
            state = gameState.generateSuccessor(agentIndex, possible_actions[i])
            if agentIndex + 1 == gameState.getNumAgents():
                v = min(v, self.alpha_beta_pr(state, a, b, depth - 1, 0))
            elif agentIndex + 1 < gameState.getNumAgents():
                v = min(v, self.alpha_beta_pr(state, a, b, depth, agentIndex+1))
            if v < a:
                return v
            b = min(b, v)
        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        # exact same code as minimax (generic programming ftw)
        actions = gameState.getLegalActions(0)
        first_state = gameState.generateSuccessor(0, actions[0])
        cost = self.Expectimax(first_state, self.depth, 1)
        move = actions[0]

        for i in range(1, len(actions)):
            state = gameState.generateSuccessor(0, actions[i])
            newcost = self.Expectimax(state, self.depth, 1)
            if newcost > cost:
                cost = newcost
                move = actions[i]
        return move

    def Expectimax(self, gameState, depth, agentIndex):
        if not depth:
            return self.evaluationFunction(gameState)
        possible_actions = gameState.getLegalActions(agentIndex)
        if not possible_actions:
            return self.evaluationFunction(gameState)
        if not agentIndex:
            states = [gameState.generateSuccessor(0, action) for action in possible_actions]
            return max([self.Expectimax(state, depth, agentIndex+1) for state in states])
        elif agentIndex + 1 == gameState.getNumAgents():
            states = [gameState.generateSuccessor(agentIndex, action) for action in possible_actions]
            return sum([self.Expectimax(state, depth-1, 0)/len(states) for state in states])
        else:
            states = [gameState.generateSuccessor(agentIndex, action) for action in possible_actions]
            return sum([self.Expectimax(state, depth, agentIndex+1)/len(states) for state in states])

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION:
      we evaluate:
      the score of the state
      how far is pacman from the closest ghost
      how close is pacman to the furthest food
      how close is pacman to the furthest capsule
    """

    position = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    capsules = currentGameState.getCapsules()
    score = currentGameState.getScore()

    score += min(manhattanDistance(position, ghost) for ghost in currentGameState.getGhostPositions())
    score += sum(newScaredTimes)
    if newFood:
        score -= max(manhattanDistance(position, Food) for Food in newFood)
    if capsules:
        score -= max(manhattanDistance(position, capsule) for capsule in capsules)

    return score

# Abbreviation
better = betterEvaluationFunction
