import random
import sys
sys.path.append("..")  #so other modules can be found in parent dir
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import *
from AIPlayerUtils import *

# Additinoal library
import math


##
#AIPlayer
#Description: The responsbility of this class is to interact with the game by
#deciding a valid move based on a given game state. This class has methods that
#will be implemented by students in Dr. Nuxoll's AI course.
#
#Variables:
#   playerId - The id of the player.
##
class AIPlayer(Player):
    # [Edit] Add this variable
    playerId = None
    #__init__
    #Description: Creates a new Player
    #
    #Parameters:
    #   inputPlayerId - The id to give the new player (int)
    #   cpy           - whether the player is a copy (when playing itself)
    ##
    def __init__(self, inputPlayerId):
        super(AIPlayer,self).__init__(inputPlayerId, "HW3_shinj28_oliveros27")
    
    ##
    #getPlacement
    #
    #Description: called during setup phase for each Construction that
    #   must be placed by the player.  These items are: 1 Anthill on
    #   the player's side; 1 tunnel on player's side; 9 grass on the
    #   player's side; and 2 food on the enemy's side.
    #
    #Parameters:
    #   construction - the Construction to be placed.
    #   currentState - the state of the game at this point in time.
    #
    #Return: The coordinates of where the construction is to be placed
    ##
    def getPlacement(self, currentState):
        self.playerId = currentState.whoseTurn

        numToPlace = 0
        #implemented by students to return their next move
        if currentState.phase == SETUP_PHASE_1:    #stuff on my side
            numToPlace = 11
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on your side of the board
                    y = random.randint(0, 3)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        elif currentState.phase == SETUP_PHASE_2:   #stuff on foe's side
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on enemy side of the board
                    y = random.randint(6, 9)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        else:
            return [(0, 0)]
    
    ##
    #getMove
    #Description: Gets the next move from the Player.
    #
    #Parameters:
    #   currentState - The state of the current game waiting for the player's move (GameState)
    #
    #Return: The Move to be made
    def getMove(self, currentState):
        # 1. Create the root node
        root_node = self.createNode(move=None, parentNode=None, currentState=currentState)
        
        # 2. Mini-max search
        mini_max_val, best_node = self.mini_max_search(curr_node=root_node, max_depth=3, remain_ratio=0.25, alpha=-float('inf'), beta=float('inf'))

        # 3. Find the best movement from best node
        best_move = best_node['move']

        return best_move

    ##
    #getAttack
    #Description: Gets the attack to be made from the Player
    #
    #Parameters:
    #   currentState - A clone of the current state (GameState)
    #   attackingAnt - The ant currently making the attack (Ant)
    #   enemyLocation - The Locations of the Enemies that can be attacked (Location[])
    ##
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        #Attack a random enemy.
        return enemyLocations[random.randint(0, len(enemyLocations) - 1)]

    ##
    #registerWin
    #
    # This agent doens't learn
    #
    def registerWin(self, hasWon):
        #method templaste, not implemented
        pass

    # Helper Functions ----------------------------------------------------------------------------------------------------
    def mini_max_search(self, curr_node, max_depth, remain_ratio, alpha, beta):
        # 1. Basic information
        currentState = curr_node['currentState']

        # 2. Check the termination condition:
        if curr_node['depth'] == max_depth:
            utility_val = curr_node['utility']
            # Leaf node does not have the child node!
            return utility_val, None
        
        # 3. Expand the node!	
        expanded_nodes = self.expandNode(curr_node)
        # Sort the expanded node with ascending order with utility value
        expanded_nodes.sort(key=lambda node: node['utility'])
        remain_nodes_num = max(10, math.ceil(len(expanded_nodes) * remain_ratio))

        # 4. Find the best child with node!
        if currentState.whoseTurn == self.playerId: # My turn
            best_val = -float('inf')
            expanded_nodes = expanded_nodes[-remain_nodes_num:]
        else: # Enemy's turn
            best_val = float('inf')
            expanded_nodes = expanded_nodes[:remain_nodes_num]
        best_child_node = None	

        for child_node in expanded_nodes:
            child_utility, _ = self.mini_max_search(child_node, max_depth, remain_ratio, alpha, beta)

            # Maximization (My Turn)
            if currentState.whoseTurn == self.playerId:
                if child_utility > best_val:
                    best_val = child_utility
                    best_child_node = child_node
                    alpha = max(alpha, best_val)

            # Minimization (Enemy's Turn)
            else:
                if child_utility < best_val:
                    best_val = child_utility
                    best_child_node = child_node
                    beta = min(beta, best_val)

            if beta <= alpha:
                break
        
        return best_val, best_child_node

            

    def createNode(self, move, parentNode, currentState):
        # 1. Update the depth
        if parentNode == None:
            depth = 0
        else:
            depth = parentNode['depth'] + 1

        # 2. Calculate the f-value for A* search
        if parentNode != None:
            parentState = parentNode['currentState']
            # f = g + h -> depth + h_value
            utility = self.evaluate_curr_state(parentState, currentState)
        else:
            utility = 0
                
        node = {'move': move,
                'utility': utility,
                'currentState': currentState,
                'parentNode': parentNode,
                'depth': depth
                }
        
        return node

    def expandNode(self, currentNode):
        # 1. Generate a list of all valid moves from the GameState in the given node
        currentState = currentNode['currentState']
        moves = listAllLegalMoves(currentState)

        # 2. Create properly initialized node for each valid move
        expanded_nodes = []
        for move in moves:
            # Use the 'getNextStateAdversarial()'
            next_node = self.createNode(move, currentNode, getNextStateAdversarial(currentState, move))
            expanded_nodes.append(next_node)

        # 3. Retrun a list of all the new nodes
        return expanded_nodes

    # Utility functions ----------------------------------------------------------------------------------------------------
   # In HW3, "maximizing utility" == "minimizing h_value"
    def evaluate_curr_state(self, parentState, currentState):
        MAX_EVAL_VAL = 100000
        h_value = self.h_func(parentState, currentState)
        return MAX_EVAL_VAL - h_value

    def h_func(self, parentState, currentState):
        # Goal 1. Get 11 foods
        h_food = self.get_h_food(parentState, currentState)
        
        # Goal 2. Kill the enemy queen
        h_attack = self.get_h_attack(parentState, currentState)

        # Goal 3. Move the queen -> do not block the worker
        h_queen = self.get_h_queen(parentState, currentState)

        # 4. Accumulate
        h_final = h_food + h_attack + h_queen

        return h_final
    
    def dist_to_moves (self, dist, ant_type):
        return math.ceil (dist / UNIT_STATS[ant_type][MOVEMENT])

    def get_h_food(self, parentState, currentState):
        # 1. Get basic information
        myId = self.playerId
        myInv = currentState.inventories[myId]

        workerList = getAntList(currentState, myId, (WORKER,))
        myTunnel = getConstrList(currentState, myId, (TUNNEL,))[0]

        # 1.1. If already satisfies FOOD_GOAL -> no more movements
        Food_needed = FOOD_GOAL - myInv.foodCount
        if Food_needed == 0:
            return 0
        
        # 1.2. myFood: closest from my tunnel
        Foods = getConstrList(currentState, None, (FOOD,))
        myFood_1, myFood_2 = None, None
        for food in Foods:
            # Out of 4 foods, the condition for my food is (y-axis < =3)
            if food.coords[1] <= 3:
                if myFood_1 == None:
                    myFood_1 = food
                else:
                    myFood_2 = food

        # 2. Calculate the total movements
        total_moves = 1000

        # 2.2. If there is worker
        if len(workerList) == 0 or len(workerList) > 2:
            return 100000
        
        if workerList != []:
            total_moves -= 1000
            # 2. How many moves do I need to gather with current worker?
            for worker in workerList:
                if (worker.carrying):
                    dist = approxDist(worker.coords, myTunnel.coords)
                    total_moves += self.dist_to_moves(dist, WORKER)
                    Food_needed -= 1 # each worker will deliver the food

                else:
                    # 2.2. (Dist from current worker to myFood)
                    dist_to_food_1 = approxDist(worker.coords, myFood_1.coords)
                    dist_to_food_2 = approxDist(worker.coords, myFood_2.coords)

                    if dist_to_food_1 < dist_to_food_2:
                        dist = dist_to_food_1
                        dist += approxDist(myFood_1.coords, myTunnel.coords)
                    else:
                        dist = dist_to_food_2
                        dist += approxDist(myFood_2.coords, myTunnel.coords)
                    total_moves += self.dist_to_moves(dist, WORKER)
                    Food_needed -= 1 # each worker will deliver the food
        
        # 3. Calculate the additional movements for more foods
        dist_between_tunnel_food = min (approxDist(myFood_1.coords, myTunnel.coords), approxDist(myFood_2.coords, myTunnel.coords))
        total_moves += 2 * Food_needed * self.dist_to_moves(dist_between_tunnel_food, WORKER)
            
        return  total_moves

    def get_h_attack(self, parentState, currentState):
        # 1. Basic information
        myId = self.playerId
        myInv = currentState.inventories[myId]

        enemyId = 1 - myId
        enemyInv = currentState.inventories[enemyId]

        myAntList = getAntList(currentState, myId)
        enemyTunnel = getConstrList(currentState, enemyId, (TUNNEL,))[0]
        enemyAnthill = getConstrList(currentState, enemyId, (ANTHILL,))[0]

        # 2. If enemy queen is killed
        if getAntList(currentState, enemyId, (QUEEN,)) == []:
            return 0
        
        Total_moves = 1200
        # 3. Check whether is there any attacker in my inventories
        num_r_soldier = 0
        num_soldier = 0
        num_drone = 0

        for ant in myAntList:
            if ant.type == R_SOLDIER:
                num_r_soldier += 1
            if ant.type == SOLDIER:
                num_soldier += 1
            if ant.type == DRONE:
                num_drone += 1

        if num_drone >= 1:
            return 100000
        
        if num_soldier > 1:
            return 100000
        
        if num_r_soldier > 1:
            return 100000

        if num_r_soldier != 0:
            Total_moves -= 600

        if num_soldier != 0:
            Total_moves -= 500

        # 4. If queen is still alive -> Find the location of queen
        enemyQueen = getAntList(currentState, enemyId, (QUEEN,))[0]
        enemyWorkers = getAntList(currentState, enemyId, (WORKER,))
        if enemyWorkers == []:
            Total_moves -= 100

        for ant in myAntList:
            if ant.type == SOLDIER:
                dist_to_target = approxDist(ant.coords, enemyAnthill.coords)
                moves_to_target = self.dist_to_moves (dist_to_target, SOLDIER)
                Total_moves += moves_to_target
            if ant.type == R_SOLDIER:
                if enemyWorkers == []:
                    dist_to_target = approxDist(ant.coords, enemyAnthill.coords)
                else:
                    dist_to_target = approxDist(ant.coords, enemyTunnel.coords)
                moves_to_target = self.dist_to_moves (dist_to_target, R_SOLDIER)
                Total_moves += moves_to_target

        return Total_moves
    
    # Check whethere the Queen is blocking the anthill
    def get_h_queen(self, parentState, currentState):
        # 1. Basic information
        myId = self.playerId
        myAnthill = getConstrList(currentState, myId, (ANTHILL,))[0]
        myTunnel = getConstrList(currentState, myId, (TUNNEL,))[0]

        if getAntList(currentState, myId, (QUEEN,)) == []:
            return 10000

        myQueen = getAntList(currentState, myId, (QUEEN,))[0]

        Foods = getConstrList(currentState, None, (FOOD,))
        myFood_1, myFood_2 = None, None
        for food in Foods:
            # Out of 4 foods, the condition for my food is (y-axis < =3)
            if food.coords[1] <= 3:
                if myFood_1 == None:
                    myFood_1 = food
                else:
                    myFood_2 = food

        # 2. Check the current condition of Queen
        Total_moves = 0
        # 2.1. Queen must not be on the anthill / tunnel
        if  (myQueen.coords == myAnthill.coords) and (myQueen.coords == myTunnel.coords):
            Total_moves = 10000
        # 2.2. Queen must not be on the foods
        elif (myQueen.coords == myFood_1.coords) and (myQueen.coords == myFood_2.coords):
            Total_moves = 10000

        dist_to_destination = approxDist(myQueen.coords, (1,2))
        Total_moves += self.dist_to_moves (dist_to_destination, QUEEN)
        
        return Total_moves
