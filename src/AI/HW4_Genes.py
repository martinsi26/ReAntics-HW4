# win %'s
# 69.2%, 71.5%

import random
import sys
import os
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
    
    filepath = "./oliveros27_population.txt"
    population:list[list[float]] = []
    next_to_evaluate:int = 0
    fitness:list[int] = []
    
    POPULATION_SIZE = 50
    mutation_rate = 0.01
    N_fittest = 10
    games_played = 0
    GAMES_TO_PLAY = 5
    NUM_ALLELES = 15
    
    #__init__
    #Description: Creates a new Player
    #
    #Parameters:
    #   inputPlayerId - The id to give the new player (int)
    #   cpy           - whether the player is a copy (when playing itself)
    ##
    def __init__(self, inputPlayerId):
        super(AIPlayer,self).__init__(inputPlayerId, "HW4_genetic_algo")
    
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
        # set my player ID
        self.playerId = currentState.whoseTurn
        # initialize the generation of genes
        if not self.population:
            self.initializePopulation(self.POPULATION_SIZE)
            
        # see if there are any legal moves
        moves = listAllLegalMoves(currentState)
        if not moves:
            return Move(END, None, None)    
        
        # epsilon-greedy exploration (5%)
        # 5% chance to choose a random move that isn't an END turn move
        if random.random() < 0.05:
            non_end_moves = [m for m in moves if m.moveType != END]
            if non_end_moves:
                return random.choice(non_end_moves)
        
        # initialize the root of the tree
        root = self.createNode(move=None, parentNode=None, currentState=currentState)
        # expand the root
        expanded_nodes = self.expandNode(root)
        if not expanded_nodes:
            return Move(END, None, None)
        
        scored = [self.scoreMove(currentState, move) for move in moves]
        best_score = max(scored, key=lambda d: d[0])[0]
        best_moves = [move for score, move in scored if score == best_score]
        while not self.regulateAnts(currentState, best_moves[0]):
            best_moves.pop(0)
            if not best_moves:
                break
        best_move = best_moves[0] if best_moves else None
        # print("best move: ", best_move)
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
        # if the gene has won +1 fitness and if it lost then -1 fitness
        if hasWon:
            self.fitness[self.next_to_evaluate] += 1
        else:
            self.fitness[self.next_to_evaluate] -= 1
        
        # win or lose we've played +1 games
        self.games_played += 1
        
        # check if the current gene has played N games
        if self.games_played >= self.GAMES_TO_PLAY:
            self.next_to_evaluate += 1
            self.games_played = 0
        
        # if we have evaluated all the genes from a population reset the next_to_evaluate back to 0
        # and create a new population using the fittest genes
        if self.next_to_evaluate == self.POPULATION_SIZE:
            self.generateNextGeneration(self.N_fittest)
        
    # returns true if best_move will NOT lead to the following:
    # 1. The agent has more than 2 workers
    # 2. The agent has more than 1 soldier
    # 3. The agent has more than 1 ranged soldier
    def regulateAnts(self, currentState, best_move):
        next_state = getNextStateAdversarial(currentState, best_move)
        myInv = currentState.inventories[self.playerId]
        my_workers = getAntList(next_state, self.playerId, [WORKER,])
        my_soldiers = getAntList(next_state, self.playerId, [SOLDIER,])
        my_ranged_soldiers = getAntList(next_state, self.playerId, [R_SOLDIER,])
        return len(my_workers) < 5 and len(my_soldiers) < 3 and len(my_ranged_soldiers) < 3

    
    def scoreMove(self, currentState, move):
        curr_state = currentState
        next_state = getNextStateAdversarial(currentState, move)
        
        curr_utility = self.utility(curr_state)
        next_utility = self.utility(next_state)
        
        delta = next_utility - curr_utility
        if move.moveType == END:
            delta -= 0.05       # nudge ties away from END
        delta += random.random() * 1e-6 # tie breaker
        
        return delta, move
    
    def initializePopulation(self, population_size):
        self.population = []
        self.fitness = [0 for _ in range(self.POPULATION_SIZE)]
        self.next_to_evaluate = 0
        self.games_played = 0
        if not os.path.exists(self.filepath):
            for gene in range(population_size):
                # initialize random weights for each gene
                self.population.append([random.uniform(-10, 10) for _ in range(self.NUM_ALLELES)])
            with open(self.filepath, "w") as f:
                # write the population to the file (one line per gene)
                for gene in self.population:
                    f.write(str(gene) + "\n")
        else:
            # open the population file and initialize the population with the values in the file
            with open(self.filepath, "r") as f:
                for gene in f:
                    alleles = gene.strip("[]\n").split(",")
                    if alleles:
                        self.population.append([float(x) for x in alleles])
                
    def crossoverMutate(self, parent1, parent2, mutation_rate):
        # split the parents at a random point
        split_point = random.randint(1, 10) # make the split crossover at least 1 allele from each parent
        child1 = parent1[:split_point] + parent2[split_point:]
        child2 = parent2[:split_point] + parent1[split_point:]
        # each child has a chance for mutation
        if random.random() < mutation_rate:
            child1[random.randint(0, 11)] = random.uniform(-10, 10)
        if random.random() < mutation_rate:
            child2[random.randint(0, 11)] = random.uniform(-10, 10)
        
        return child1, child2
    
    def generateNextGeneration(self, N_fittest: int):
        next_generation = []
        #self.population.sort(key=lambda x: self.fitness[self.population.index(x)], reverse=True)
        genes_zipped_sorted = sorted(zip(self.fitness, self.population), key=lambda x: x[0], reverse=True)
        
        fittest_genes = [g for _, g in genes_zipped_sorted[:N_fittest]]
        while len(next_generation) < self.POPULATION_SIZE:
            p1, p2 = random.sample(fittest_genes, 2)
            c1, c2 =  self.crossoverMutate(p1, p2, self.mutation_rate)
            next_generation.append(c1)
            if len(next_generation) < self.POPULATION_SIZE:
                next_generation.append(c2)
                
        # replace population
        self.population = next_generation
        
        # reset fitness, games_played, and next_to_evaluate
        self.fitness = [0 for _ in range(self.POPULATION_SIZE)]
        self.games_played = 0
        self.next_to_evaluate = 0
        
        # decay mutation rate with a floor
        self.mutation_rate = max(0.05, self.mutation_rate * 0.99)
        
        # write the new generation to the file
        with open(self.filepath, "w") as f:
            for gene in next_generation:
                f.write(str(gene) + "\n")
                # print("wrote gene: ", gene)
        return self.population
    
    def utility(self, currentState):
        ### get feature values ###
        feature_vector = []
        ## alleles 0-4 ##
        food_diff = self.get_food_val(currentState)
        queen_HP_diff = self.get_queen_HP_val(currentState)
        soldier_count_diff = self.get_ant_diff(currentState, SOLDIER)
        r_soldier_count_diff = self.get_ant_diff(currentState, R_SOLDIER)
        offensive_capability = self.better_offense(currentState)
        feature_vector.append(food_diff)
        feature_vector.append(queen_HP_diff)
        feature_vector.append(soldier_count_diff)
        feature_vector.append(r_soldier_count_diff)
        feature_vector.append(offensive_capability)
        ## alleles 5-9 ##
        # mqueen = my queen : oqueen = opp queen : etc.
        oqueen_mants_avg = self.get_avg_dist(currentState, 5)
        oants_mqueen_avg = self.get_avg_dist(currentState, 6)
        oanthill_mants_avg = self.get_avg_dist(currentState, 7)
        oants_manthill_avg = self.get_avg_dist(currentState, 8)
        mants_closest_oant = self.get_avg_dist(currentState, 9)
        feature_vector.append(oqueen_mants_avg)
        feature_vector.append(oants_mqueen_avg)
        feature_vector.append(oanthill_mants_avg)
        feature_vector.append(oants_manthill_avg)
        feature_vector.append(mants_closest_oant)
        ## alleles 10-11 ##
        mworkers_mqueen_avg = self.get_avg_dist(currentState, 10)
        mqueen_oqueen_dist = self.get_avg_dist(currentState, 11)
        feature_vector.append(mworkers_mqueen_avg)
        feature_vector.append(mqueen_oqueen_dist)
        ## alleles 12-14 ##
        g1, g2, g3 = self.worker_progress(currentState)
        feature_vector.append(g1)
        feature_vector.append(g2)
        feature_vector.append(g3)
        
        ### calculate utility ###
        # calculate the value of each feature's val multiplied by its weight from the gene
        gene_weights = self.population[self.next_to_evaluate]
        for i in range(len(feature_vector)):
            feature_vector[i] = feature_vector[i] * gene_weights[i]
        return sum(feature_vector)

    # max_dist on a 10x10 board is 18 (corner to corner)
    def normalize_dist(self, dist, max_dist=18):
        return min(1.0, dist / max_dist)

    def get_food_val(self, currentState):
        me = self.playerId
        opp = 1 - currentState.whoseTurn
        my_food_count = currentState.inventories[me].foodCount
        opp_food_count = currentState.inventories[opp].foodCount
        if opp_food_count >= 11:
            return 0
        return 1.0 if my_food_count >= opp_food_count else 0.0
        
    def get_queen_HP_val(self, currentState):
        me = self.playerId
        opp = 1 - currentState.whoseTurn
        my_queen = currentState.inventories[me].getQueen()
        opp_queen = currentState.inventories[opp].getQueen()
        # check if I or the opponent has a queen
        if my_queen is None:
            return 0.0
        if opp_queen is None:
            return 1.0
        my_queen_HP = currentState.inventories[me].getQueen().health
        opp_queen_HP = currentState.inventories[opp].getQueen().health
        return 1.0 if my_queen_HP >= opp_queen_HP else 0.0
    def get_ant_diff(self, currentState, ant_type):
        me = self.playerId
        opp = 1 - currentState.whoseTurn
        my_soldier_count = len(getAntList(currentState, me, [ant_type,]))
        opp_soldier_count = len(getAntList(currentState, opp, [ant_type,]))
        # if I have no soldiers or ranged soldiers that's bad :(
        if my_soldier_count == 0 or my_soldier_count > 1:
            return 0.0
        return 1.0 if my_soldier_count > opp_soldier_count else 0.0
    def better_offense(self, currentState):
        me = self.playerId
        opp = 1 - currentState.whoseTurn
        my_off_ant_count = len(getAntList(currentState, me, [DRONE, SOLDIER, R_SOLDIER]))
        opp_off_ant_count = len(getAntList(currentState, opp, [DRONE, SOLDIER, R_SOLDIER]))
        return 1.0 if my_off_ant_count > opp_off_ant_count else 0.0
    # allele_index tells the method which allele to calculate a value for
    def get_avg_dist(self, currentState, allele_index):
        me = self.playerId
        opp = 1 - currentState.whoseTurn
        # check if the opponent's queen is alive
        if currentState.inventories[opp].getQueen() is None:
                    return 1.0
        match allele_index:
            case 5:  # Avg distance between the opponent queen and my offensive ants (smaller distance == better)
                opp_queen = getAntList(currentState, opp, [QUEEN,])[0]
                my_ants = getAntList(currentState, me, [DRONE, SOLDIER, R_SOLDIER])
                if len(my_ants) == 0:
                    return 0.0
                avg = sum([approxDist(opp_queen.coords, ant.coords) for ant in my_ants]) / len(my_ants)
                return 1.0 - self.normalize_dist(avg)
            case 6:  # Avg distance between the opponent's offensive ants and my queen (bigger distance == better)
                my_queen = getAntList(currentState, me, [QUEEN,])[0]
                opp_ants = getAntList(currentState, opp, [DRONE, SOLDIER, R_SOLDIER])
                if len(opp_ants) == 0:
                    return 1.0
                avg = sum([approxDist(my_queen.coords, ant.coords) for ant in opp_ants]) / len(opp_ants)
                return 1.0 - self.normalize_dist(avg)
            case 7:  # Avg distance between the opponent's anthill and my offensive ants (smaller distance == better)
                my_ants = getAntList(currentState, me, [DRONE, SOLDIER, R_SOLDIER])
                if getConstrList(currentState, opp, [ANTHILL,]) is None:
                    return 1.0
                opp_anthill = getConstrList(currentState, opp, [ANTHILL,])[0]
                if len(my_ants) == 0:
                    return 0.0
                avg = sum([approxDist(opp_anthill.coords, ant.coords) for ant in my_ants]) / len(my_ants)
                return 1.0 - self.normalize_dist(avg)
            case 8:  # Avg distance between the opponent's offensive ants and my anthill (bigger distance == better)
                if getConstrList(currentState, me, [ANTHILL,]) is None:
                    return 0.0
                my_anthill = getConstrList(currentState, me, [ANTHILL,])[0]
                opp_ants = getAntList(currentState, opp, [DRONE, SOLDIER, R_SOLDIER])
                if len(opp_ants) == 0:
                    return 1.0
                avg = sum([approxDist(my_anthill.coords, ant.coords) for ant in opp_ants]) / len(opp_ants)
                return 1.0 - self.normalize_dist(avg)
            case 9:  # Avg distance between my offensive ants and the opponent's offensive ant that is closest to my queen
                opp_ants = getAntList(currentState, opp, [DRONE, SOLDIER, R_SOLDIER])
                if getAntList(currentState, me, [QUEEN,]) is None:
                    return 0.0
                my_queen = getAntList(currentState, me, [QUEEN,])[0]
                if len(opp_ants) == 0:
                    return 1.0 # very large number since there are no offensive opponent ants
                closest_threat_coords = min([ant.coords for ant in opp_ants], key=lambda coords: approxDist(coords, my_queen.coords))
                my_ants = getAntList(currentState, me, [DRONE, SOLDIER, R_SOLDIER])
                if len(my_ants) == 0:
                    return 0.0 # very small number since the agent has no offensive ants to defend
                avg = sum([approxDist(closest_threat_coords, ant.coords) for ant in my_ants]) / len(my_ants)
                return 1.0 - self.normalize_dist(avg)    
            case 10: # Avg distance between my workers and my queen
                # check if my queen is alive (edge case)
                if getAntList(currentState, me, [QUEEN,]) is None:
                    return 0
                my_queen = getAntList(currentState, me, [QUEEN,])[0]
                my_workers = getAntList(currentState, me, [WORKER,])
                if len(my_workers) == 0 or len(my_workers) > 2:
                    return 0
                avg = sum([approxDist(my_queen.coords, worker.coords) for worker in my_workers]) / len(my_workers)
                return self.normalize_dist(avg)
            case 11: # Distance between my queen and opponent's queen
                my_queen = getAntList(currentState, me, [QUEEN,])[0]
                opp_queen = getAntList(currentState, opp, [QUEEN,])[0]
                return self.normalize_dist(approxDist(my_queen.coords, opp_queen.coords))

    # helper methods for getting features for alleles 12-14
    def my_dropoffs(self, currentState):
        me = self.playerId
        tunnel = getConstrList(currentState, me, (TUNNEL,))[0]
        hill = getConstrList(currentState, me, (ANTHILL,))[0]
        dropoffs = [tunnel, hill]
        return dropoffs

    def closest_target(self, curr_coords, targets):
        return min((approxDist(curr_coords, target.coords) for target in targets), default=None)
    
    def worker_progress(self, currentState):
        me = self.playerId
        
        my_workers = getAntList(currentState, me, [WORKER,])
        if not my_workers:
            return [0.0, 0.0, 0.0]
        foods = getConstrList(currentState, None, [FOOD,])
        # this should never happen
        if not foods:
            return [0.0, 0.0, 0.0]
        dropoffs = self.my_dropoffs(currentState)
        # check if my dropoffs still exist
        if not dropoffs:
            return [0.0, 0.0, 0.0]
        g1 = g2 = g3_sum = 0.0
        
        for worker in my_workers:
            # boolean to track if any workers 
            on_food = any(worker.coords == food.coords for food in foods)
            on_dropoff = any(worker.coords == dropoff.coords for dropoff in dropoffs)
            # if a worker is carrying food prioritize getting it to a dropoff
            if worker.carrying:
                # may need to be inversed (if a worker is carrying food we don't want them occupying food tile)
                g1 += 0.0 if on_food else 1.0
                dist = self.closest_target(worker.coords, foods)
                # closer is better -> invert & normalize
                p = 1.0 - self.normalize_dist(dist or 0.0)
            else: # if a worker is not carrying food prioritize getting it to a food
                # may need to be inversed (if a worker is not carrying food we don't want them occupying the anthill or tunnel tiles)
                g2 += 0.0 if on_dropoff else 1.0
                dist = self.closest_target(worker.coords, dropoffs)
                p = 1.0 - self.normalize_dist(dist or 0.0)
                
            g3_sum += p
        num_workers = float(len(my_workers))
        return [g1/num_workers, g2/num_workers, g3_sum/num_workers]
                
        
        

    def createNode(self, move, parentNode, currentState):
        # 1. Update the depth
        if parentNode == None:
            depth = 0
        else:
            depth = parentNode['depth'] + 1

        if parentNode != None:
            parentState = parentNode['currentState']
            utility = self.utility(currentState)
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

        # 3. Return a list of all the new nodes
        return expanded_nodes
