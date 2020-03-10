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

MAX_VALUE = float('inf')
MIN_VALUE = float('-inf')
FOOD_BONUS = 100
DIST_PENALTY = 5
GHOST_DIST_BONUS = 3
SAFE_DIST = 2

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
        if successorGameState.isWin():
            return MAX_VALUE
        if successorGameState.isLose():
            return MIN_VALUE
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        for scaredTime,ghostState in zip(newScaredTimes,newGhostStates):
            if(scaredTime>SAFE_DIST):
              continue
            if util.manhattanDistance(ghostState.getPosition(), newPos) < SAFE_DIST:
                return MIN_VALUE
        minDist = MAX_VALUE
        for food in newFood.asList():
            minDist  = min(minDist, util.manhattanDistance(food, newPos))
        foodDiff = currentGameState.getNumFood()-successorGameState.getNumFood()
        foodPoints = FOOD_BONUS * foodDiff
        totPenalty = minDist * DIST_PENALTY
        return successorGameState.getScore() - totPenalty + foodPoints

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
        "*** YOUR CODE HERE ***"
        maxValue = MIN_VALUE
        maxAction = None
        currentDepth = 0
        for action in gameState.getLegalActions(0) :
            nextState = gameState.generateSuccessor(0, action)
            value = self.value(nextState, currentDepth, 1)
            if value > maxValue:
                maxAction = action
                maxValue = value
        return maxAction

    def value(self, gameState, currentDepth, agentIndex):
        #base case -> terminal state
        if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        # agent is MAX
        if agentIndex == 0:
            return self.maxValue(gameState,currentDepth, 0)
        # agent is MIN
        return self.minValue(gameState,currentDepth,agentIndex) 

    def maxValue(self, gameState, currentDepth, agentIndex):
        maxValue = MIN_VALUE
        for action in gameState.getLegalActions(agentIndex):
            nextState = gameState.generateSuccessor(agentIndex, action)
            maxValue = max(maxValue, self.value(nextState, currentDepth,1))
        return maxValue

    def minValue(self, gameState, currentDepth, agentIndex):
        minValue = MAX_VALUE
        ghostCount = gameState.getNumAgents() -1 ;
        for action in gameState.getLegalActions(agentIndex):
            nextState = gameState.generateSuccessor(agentIndex, action)
            nextDepth = currentDepth+1 if (agentIndex%ghostCount)==0 else currentDepth 
            nextAgent = 0 if (agentIndex%ghostCount)==0 else agentIndex+1
            minValue = min(minValue,self.value(nextState,nextDepth,nextAgent))
        return minValue




class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        maxValue = MIN_VALUE
        alpha = MIN_VALUE
        beta = MAX_VALUE
        maxAction = None
        currentDepth = 0
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            value = self.value(nextState, currentDepth, 1, alpha, beta)
            if value > maxValue:
                maxValue = value
                maxAction = action
                alpha = max(alpha, maxValue)
        return maxAction

    def value(self, gameState, currentDepth, agentIndex, alpha, beta):
        if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maxValue(gameState,currentDepth,0,alpha,beta)
        return self.minValue(gameState,currentDepth,agentIndex,alpha,beta)

    def maxValue(self, gameState, currentDepth, agentIndex, alpha, beta):
        maxValue = MIN_VALUE
        for action in gameState.getLegalActions(agentIndex):
            nextState = gameState.generateSuccessor(agentIndex, action)
            maxValue = max(maxValue, self.value(nextState , currentDepth, 1, alpha, beta))
            if maxValue > beta:
                break
            alpha = max(alpha, maxValue)
        return maxValue

    def minValue(self, gameState, currentDepth, agentIndex, alpha, beta):
        minValue = MAX_VALUE
        ghostCount = gameState.getNumAgents()-1
        for action in gameState.getLegalActions(agentIndex):
            nextState = gameState.generateSuccessor(agentIndex, action)
            nextDepth = currentDepth+1 if agentIndex%ghostCount==0 else currentDepth
            nextAgent = 0 if agentIndex%ghostCount==0 else agentIndex+1
            minValue = min(minValue,self.value(nextState,nextDepth,nextAgent,alpha,beta))
            if minValue < alpha:
                break
            beta = min(beta, minValue)
        return minValue

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
        "*** YOUR CODE HERE ***"
        maxValue = MIN_VALUE
        maxAction = Directions.STOP
        currentDepth = 0
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            value = self.value(nextState, currentDepth, 1)
            if value >  maxValue:
                maxValue = value
                maxAction = action
        return maxAction

    def value(self, gameState, currentDepth, agentIndex):
        if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maxValue(gameState,currentDepth,agentIndex)
        return self.expValue(gameState,currentDepth,agentIndex)

    def maxValue(self, gameState, currentDepth,agentIndex):
        maxValue = MIN_VALUE
        for action in gameState.getLegalActions(agentIndex):
            nextState = gameState.generateSuccessor(agentIndex, action)
            maxValue = max(maxValue,self.value(nextState, currentDepth, 1))
        return maxValue

    def expValue(self, gameState, currentDepth, agentIndex):
        weight = 0
        ghostCount = gameState.getNumAgents()-1
        for action in gameState.getLegalActions(agentIndex):
            nextState = gameState.generateSuccessor(agentIndex, action)
            nextDepth = currentDepth+1 if agentIndex%ghostCount==0 else currentDepth
            nextAgent = 0 if agentIndex%ghostCount==0 else agentIndex+1
            weight += self.value(nextState, nextDepth, nextAgent)
        return weight


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
        return MAX_VALUE
    if currentGameState.isLose():
        return MIN_VALUE
    newPos = currentGameState.getPacmanPosition()
    closeGhost = MAX_VALUE
    farGhost = 0
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    for scaredTime,ghostState in zip(newScaredTimes,newGhostStates):
        # if(scaredTime>2*SAFE_DIST):
        #       continue
        dist = util.manhattanDistance(ghostState.getPosition(), newPos)
        closeGhost = min(closeGhost,dist)
        farGhost = max(farGhost,dist)
        ''' Ghost is close and is not scared '''
        if(closeGhost<SAFE_DIST and scaredTime<(2*SAFE_DIST)):
          return MIN_VALUE

    ghostDistBonus = (closeGhost + farGhost) * GHOST_DIST_BONUS
    newFood = currentGameState.getFood()
    closeFood = MAX_VALUE
    farFood = MIN_VALUE

    for food in list(newFood.asList()):
        foodDist = util.manhattanDistance(food, newPos)
        closeFood = min(closeFood,foodDist)
        farFood = max(farFood,foodDist)
    foodDistPenalty = (closeFood+farFood)*DIST_PENALTY
    score = scoreEvaluationFunction(currentGameState)
    return score - foodDistPenalty - (FOOD_BONUS*currentGameState.getNumFood()) + ghostDistBonus



# Abbreviation
better = betterEvaluationFunction

