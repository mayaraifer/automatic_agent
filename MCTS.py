from __future__ import division

import time
import math
import random
from dm_models import crf_eval as evalu
from update_and_push import push_combi_to_interaction
from new_emb_try import create_embadding_for_qdn
import operator
import copy
#from utils import f4
import numpy as np
from scipy.special import softmax
random.seed(130)



def knowlege_based_policy(state,bert_model):
    first=True
    action_return = None
    #8.5+ - only reviews with higher score
    #8.5-7.5 - also smaller but not more 1 diff lower
    #lower then 7.5 - all
    hotels_dic = {9.66: [4, 5, 6],#
                  9.71: [4, 5, 6],#
                  9.04: [5, 3, 6, 4],#
                  7.74: [4, 6, 5,3,2],
                  4.56: [2, 5, 3, 4, 6,1,0],#
                  8.33: [3, 2, 5, 4, 6],#
                  6.69: [2, 5, 3, 4, 6,1,0],
                  7.91: [4, 3, 5, 6,2],
                  8.04: [4, 5, 3, 6,2,1],#
                  8.89: [ 4, 6, 5]}#
    current_round = state.currentRound
    while not state.isTerminal():
        if random.uniform(0,1)>=0.1:
            action = random.choice(hotels_dic[state.board.at[state.currentRound - 1, 'group_average_score']])
            state = state.takeAction(action,bert_model)
        else:
            action = random.choice([0,1,2,3,4,5,6])
            state = state.takeAction(action,bert_model)
        if first == True:
            action_return = action
            first = False
    return state.getReward()/(10.0), action_return


def random_based_policy(state,bert_model):
    first = True
    action_return = None
    while not state.isTerminal():
        action = random.choice([0,1,2,3,4,5,6])
        state = state.takeAction(action,bert_model)
        if first == True:
            action_return = action
            first = False
    return state.getReward()/(10.0), action_return


def greedy_poicy(state):
    first=True
    action_return = None
    hotels_dic = {9.66: [2, 4, 5],
                  9.71: [4, 2, 3, 5, 6],
                  9.04: [5, 3, 1, 6, 4],
                  7.74: [4, 6, 5],
                  4.56: [2, 5, 3, 4, 6],
                  8.33: [3, 2, 5, 4, 6],
                  6.69: [4, 5, 6],
                  7.91: [4, 3, 5, 6],
                  8.04: [4, 5, 3, 6],
                  8.89: [2, 3, 4, 6, 5]}
    while not state.isTerminal():
        if random.uniform(0,1)>=0.1:
            action = random.choice(hotels_dic[state.board.at[state.currentRound - 1, 'group_average_score']])
            state = state.takeAction(action)
        else:
            action = random.choice([0,1,2,3,4,5,6])
            state = state.takeAction(action)
        if first == True:
            action_return = action
            first = False
    return state.getReward(), action_return


def softmaxpolicy(state,value_model,manual_features_file,bert_features_file):
    Qsa = {}
    first = True
    action_return = None
    start_time = time.time()
    while not state.isTerminal():
        # hotels_dic = {9.66 :[2, 4, 5],
        #                     9.71: [4, 2, 3, 5, 6],
        #                     9.04 :[5, 3, 1, 6, 4],
        #                     7.74 :[4, 6, 5],
        #                     4.56 :[2, 5, 3, 4, 6],
        #                     8.33 :[3, 2, 5, 4, 6],
        #                     6.69 :[4, 5, 6],
        #                     7.91 :[4, 3, 5, 6],
        #                     8.04 :[4, 5, 3, 6],
        #                     8.89 :[2, 3, 4, 6, 5]}
        hotels_dic = {9.66: [4, 5, 6],  #
                      9.71: [4, 5, 6],  #
                      9.04: [5, 3, 6, 4],  #
                      7.74: [4, 6, 5, 3, 2],
                      4.56: [2, 5, 3, 4, 6, 1],  #
                      8.33: [3, 2, 5, 4, 6],  #
                      6.69: [4, 5, 6, 1, 2, 3],
                      7.91: [4, 3, 5, 6, 2],
                      8.04: [4, 5, 3, 6, 2, 1],  #
                      8.89: [2, 3, 4, 6, 5]}  #
        for possible_action in hotels_dic[state.board.at[state.currentRound-1,'group_average_score']]:  # [0,1,2..6]
            demo_state = push_combi_to_interaction(interaction=state.board, rounds=state.currentRound,
                                                                action=possible_action, interaction_id=state.interaction_id,
                                                                this_round_payoff=0)
            demo_state_embadded = create_embadding_for_qdn(
                         demo_state[demo_state.index <= state.currentRound - 1], manual_features_file,
                          False)
            Qsa[possible_action] = evalu(
                demo_state_embadded[[f'features_round_{i}' for i in range(1, state.currentRound + 1)]], value_model,
                'Q')[0][-1].item()
            action = [k for k, v in Qsa.items() if v == max(Qsa.values())][0]#list(np.random.multinomial(1, softmax(np.array(list(Qsa.values()))), size=1)[0]).index(1)
    # #        print(f"--- %s rollout Q seconds ---" % float(time.time() - start_time))
    #     hotels_dic = {9.66: [2, 4, 5],
    #                                        9.71: [4, 2, 3, 5, 6],
    #                                        9.04 :[5, 3, 1, 6, 4],
    #                                        7.74 :[4, 6, 5],
    #                                        4.56 :[2, 5, 3, 4, 6],
    #                                        8.33 :[3, 2, 5, 4, 6],
    #                                        6.69 :[4, 5, 6],
    #                                        7.91 :[4, 3, 5, 6],
    #                                        8.04 :[4, 5, 3, 6],
    #                                        8.89 :[2, 3, 4, 6, 5]}
    #     action = random.choice(hotels_dic[state.board.at[state.currentRound-1,'group_average_score']])
        if first == True:
            action_return = action
            first = False
        state = state.takeAction(action)
        #print(f"--- %s takeaction  seconds ---" % float(time.time() - start_time_takeAction))
    #print('rollout finish!!')
    print(f"--- %s rollout  seconds ---" % float(time.time() - start_time))
    return state.getReward(),action_return

def epsilongreedy(state,epsilon,value_model,manual_features_file,bert_features_file):
    Qsa = {}
    first = True
    action_return = None
    while not state.isTerminal():
        try:
            if random.uniform(0,1)<epsilon or state.currentRound<=3:
                action = random.choice(state.getPossibleActions())
                if first==True:
                    action_return =action
                    first = False
                #
            else:
                for possible_action in range(0, 7):  # [0,1,2..6]
                    demo_state = push_combi_to_interaction(interaction=state.board, rounds=state.currentRound,
                                                           action=possible_action, interaction_id=state.interaction_id,
                                                           this_round_payoff=0)
                    demo_state_embadded = create_embadding_for_qdn(demo_state[demo_state.index <= state.currentRound - 1] ,manual_features_file,False)
                    Qsa[possible_action] = evalu(demo_state_embadded[[f'features_round_{i}' for i in range(1, state.currentRound + 1)]], value_model,'Q')[0][-1].item()
                sorted_x = sorted(Qsa.items(), key=operator.itemgetter(1))
                sorted_x.reverse()
                action = sorted_x[0][0]
                if first==True:
                    action_return =action
                    first = False
                #print(f"--- %s rollout greedy seconds ---" % float(time.time() - start_time))
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))
        state = state.takeAction(action)

    return state.getReward(),action_return

def randomPolicy(state):
    #start_time1 = time.time()
    while not state.isTerminal():
        try:
            action = random.choice(state.getPossibleActions())
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))
        state = state.takeAction(action)
    #print(f"--- %s seconds ---" % float(time.time() - start_time1))
    return state.getReward()


class treeNode():
    def __init__(self, state, parent,value_model,manual_features_file,bert_features_file):
        self.state = state
        self.isTerminal = state.isTerminal()
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 1
        self.totalReward = 0
        self.Qsa = {0:0,1:0,2:0,3:0,4:0,5:0,6:0}
        self.Nsa = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}
        self.children = {}
        import pickle
        #loaded_model = pickle.load(open('tacl-7-3/svr_q.sav', 'rb'))
        for possible_action in range(0,7):#[0,1,2..6]
            demo_state = push_combi_to_interaction(interaction=self.state.board, rounds=self.state.currentRound,
                                                              action=possible_action, interaction_id=self.state.interaction_id,
                                                              this_round_payoff=0)
            demo_state_embadded = create_embadding_for_qdn(demo_state[demo_state.index<=self.state.currentRound-1],manual_features_file,False)
            self.Qsa[possible_action] = evalu(
                    demo_state_embadded[[f'features_round_{i}' for i in range(1, self.state.currentRound + 1)]], value_model,
                    'Q')[0][-1].item()/(10.0)
            #print(demo_state_embadded[f'features_round_{self.state.currentRound}'].values[0])
            #self.Qsa[possible_action] = loaded_model.predict(list(demo_state_embadded[f'features_round_{self.state.currentRound}'].values))[0]/10.0



class mcts():
    def __init__(self, explorationConstant,timeLimit=None, iterationLimit=None,
                 rolloutPolicy=random_based_policy):#epsilongreedy#randomPolicy
        if timeLimit != None:
            if iterationLimit != None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.timeLimit = timeLimit
            self.limitType = 'time'
        else:
            if iterationLimit == None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iterationLimit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iterationLimit
            self.limitType = 'iterations'
        self.explorationConstant = explorationConstant
        self.rollout = rolloutPolicy

    def search(self, initialState,value_model,manual_features_file,bert_features_file,bert_model):
        self.root = treeNode(initialState, None,value_model,manual_features_file,bert_features_file)
        count=0
        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                self.executeRound(value_model,manual_features_file,bert_features_file,bert_model)
                count+=1
        else:
            for i in range(self.searchLimit):
                self.executeRound(value_model)

        bestChild = self.getBestChild(self.root, 0.0)
        return self.getAction(self.root, bestChild), count

    def executeRound(self,value_model,manual_features_file,bert_features_file,bert_model):
        node = self.selectNode(self.root,value_model,manual_features_file,bert_features_file,bert_model)
        reward, action_in_node = self.rollout(node.state,bert_model)#,value_model,manual_features_file,bert_features_file)
        #print(reward)
        self.backpropogate(node, reward,action_in_node)

    def selectNode(self, node,value_model,manual_features_file,bert_features_file,bert_model):
        while not node.isTerminal:
            if node.isFullyExpanded:
                node = self.getBestChild(node, self.explorationConstant)
            else:
                return self.expand(node,value_model,manual_features_file,bert_features_file,bert_model)
        return node

    def expand(self, node,value_model,manual_features_file,bert_features_file,bert_model):
        #print(node.state.currentRound)
        actions = node.state.getPossibleActions()
        for action in actions:
            if action not in node.children.keys():
                newNode = treeNode(node.state.takeAction(action,bert_model), node,value_model,manual_features_file,bert_features_file)
                node.children[action] = copy.deepcopy(newNode)#
                node.children[action].parent = node
                if len(actions) == len(node.children):
                    node.isFullyExpanded = True
                return node.children[action]

        raise Exception("Should never reach here")

    def backpropogate(self, node, reward,action_in_node):
        next_childe = None
        while node is not None:
            node.numVisits += 1
            if next_childe!=None:
                action = next_childe.state.board.at[next_childe.state.currentRound-2,'group_sender_answer_index']-1
                node.Nsa[action] += 1
                node.Qsa[action] = node.Qsa[action] + (1/node.Nsa[action])*(reward-node.Qsa[action])
            else:
                if action_in_node!=None:
                    node.Nsa[action_in_node] += 1
                    node.Qsa[action_in_node] = node.Qsa[action_in_node] + (1 / node.Nsa[action_in_node]) * (reward - node.Qsa[action_in_node])

            node.totalReward += reward
            next_childe = node
            node = node.parent

    def getBestChild(self, node, explorationValue):
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():
            review = child.state.board.at[node.state.currentRound-1,'group_sender_answer_index']-1
            #nodeValue = child.totalReward / child.numVisits + explorationValue * math.sqrt(math.log(node.numVisits) / child.numVisits)
            nodeValue = node.Qsa[review] + explorationValue * math.sqrt(
              math.log(sum(node.Nsa.values())) / node.Nsa[review])
            #print(nodeValue)
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        return random.choice(bestNodes)

    def getAction(self, root, bestChild):
        for action, node in root.children.items():
            if node is bestChild:
                return action
