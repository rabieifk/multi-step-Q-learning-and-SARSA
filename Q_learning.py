import networkx as nx
import numpy as np
import pandas as pd
import random
from array import *

#Time:

class MapBuilder:

    def __init__(self):
        self.data_df = pd.read_csv('data.csv')
        number_of_edges = len(self.data_df)
        self.G = nx.DiGraph()

        for i in range(0, number_of_edges):
            start = self.data_df.iloc[i]['start']
            end = self.data_df.iloc[i]['end']
            Energy = self.data_df.iloc[i]['Energy']
            Time = self.data_df.iloc[i]['Time']

            self.G.add_edge(start, end, Energy=Energy, Time=Time)

    def next_State(self, state):
        if (state == 11):
            print('This state is terminal state and has no successor!')
        edges = [n for n in self.G.neighbors(state)]
        return edges

    def get_Reward(self, start, end):
        edge = self.data_df.loc[(self.data_df['start'] == (start)) & (self.data_df['end'] == (end))]
        Energy = edge.iloc[0]['Energy']
        Time = edge.iloc[0]['Time']
        return [np.random.normal(Energy, 0.5), np.random.normal(Time, 0.5)]

    def initial_state(self):
        return 1

    def terminal_state(self):
        return 11


mapping = MapBuilder()
# b = mapping.next_State(1)
b = mapping.get_Reward(1, 2)

i_state = 1
states = 11
episode = 0
ALL_POSSIBLE_ACTIONS = 21
currentState = 1
nextState = 1
currentAction = 1
nextAction = 1
epsilon = 0.5
temp_array = array('i', [1, 1, 1, 1])


# array1 = mapping.next_State(currentState)


# episod_gen(1)

# reward = mapping.get_Reward(1,2)
#
#
# def ChooseAction(currentState):
#    op_action = np.argmax(Q)
#    e_greedy_choice = np.random.uniform(1-epsilon, 1)
#    nextState = mapping.next_State(currentState)
#    if e_greedy_choice:
#        nextState = nextState[op_action]
#    else:
#       nextState = np.random.randint(0, len(nextState)-1)
#    currentState = nextState
# oldValue = 0
# alpha = 0.2
# gamma = 0.9
# value = 0
#
# nextState = mapping.next_State(currentState)
# action = np.random.randint(0, len(nextState)-1)
#
# while currentState != 11:
#
#    nextState = mapping.next_State(currentState)
#    nextState = nextState[action]
#    instant_reward = -mapping.get_Reward(currentState, nextState)[1]
#    value = instant_reward +
#    print (instant_reward)
#    Q[(currentState, action)] = oldValue + alpha * (value - oldValue)


class Qlearning:
    def __init__(self, initial_state, epsilon=0.01, alpha=0.2, gamma=0.9):
        # self.Q = np.matrix(np.zeros([11,11]))
        self.Q = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.initial_state = initial_state

    def getQ(self, state, action):
        return self.Q.get((state, action), 0.0)

    def chooseState_softmax(self, state):
        if state != 11:
            nxS = mapping.next_State(state)
            #print ("Hi", nxS)
            if nxS[0] != 11:
                Q_table = np.zeros(len(nxS))
                #q = [self.getQ(state, a) for a in mapping.next_State(state)]
                for t in range(len(nxS)):
                    Q_table[t] = self.Q[(state, nxS[t])]
                    Q_table[t] = np.exp(Q_table[t])
                    #print ("Q:", nxS[t], self.Q[(state, nxS[t])])
                sumofQ = sum(Q_table)
                for r in range(len(Q_table)):
                    Q_table[r] = Q_table[r]/sumofQ
                #print ("Q_table:", Q_table)
                i = np.argmax(Q_table, axis=0)
                #next_State = mapping.next_State(state)[i]
                Q_temp = Q_table
                temp = np.zeros(len(Q_temp))
                #for u in range(len(Q_temp)):
                #    print ("U:", u)
                #    i = np.argmax(Q_temp, axis=0)
                #    print ("i:", i, np.amax(Q_temp), Q_temp)
                #    temp[u] = Q_temp[i]
                #    print ("temp:", temp)
                #    Q_temp[i] = 0
                #print (temp)
                probability = random.uniform(0, 1)
                for l in range(len(Q_table)):
                    if l == 0:
                        if probability < Q_table[l]:
                            k = np.where(Q_table == Q_table[l])
                            i = k[0][0]
                            #print ("1:", k, i)
                    elif l == len(Q_table)-1:
                        if Q_table[l-1] <= probability < 1 :
                            k = np.where(Q_table == Q_table[l])
                            i = k[0][0]
                            #print ("2:", k,i)
                    else:
                        if Q_table[l-1] <= probability < Q_table[l]:
                            k = np.where(Q_table == Q_table[l])
                            i = k[0][0]
                            #print ("3:", k, i)

                next_State = mapping.next_State(state)[i]
                #print (next_State)
                return next_State
            else:
                return nxS[0]
        else:
            return state

    def chooseState(self, state):
        if random.random() < epsilon:
            next_State = random.choice(mapping.next_State(state))
            #next_State = mapping.next_State(state)[random.randint(0, len(mapping.next_State(state))-1)]
        else:
            q = [self.getQ(state, a) for a in mapping.next_State(state)]
            maxQ = max(q)
            count = q.count(maxQ)
            if count > 1:
                best = [i for i in range(len(mapping.next_State(state))) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)

            next_State = mapping.next_State(state)[i]
            #print (next_State)
        return next_State

    def learnQ(self, state, nextState, reward, value):
        oldv = self.Q.get((state, nextState), None)
        if oldv is None:
            self.Q[(state, nextState)] = reward
        else:
            self.Q[(state, nextState)] = oldv + self.alpha * (value - oldv)

    def learn(self, state1, nextState, reward, state2, action2):
        if state2 != 11:
            qnext = max([self.getQ(state2, action2)])
            self.learnQ(state1, nextState, reward, reward + self.gamma * qnext)
        elif state2 == 11:
            self.learnQ(state1, nextState, reward, reward)


# mapping = MapBuilder()
qlearning = Qlearning(currentState, 0.01, 0.2, 0.9)
b = qlearning.chooseState(currentState)



# print (b)

def update(current_state):
    if current_state == 11:
        return current_state
    elif current_state != 11:
        #print ("current_state:", current_state)
        nextState = qlearning.chooseState(current_state)
        nextState = qlearning.chooseState_softmax(current_state)
        if nextState != 11:
            #print ("nextState:", nextState)
            instant_reward = -mapping.get_Reward(current_state, nextState)[1]
            #print ("instant_reward:", instant_reward)
            nx_nxState = qlearning.chooseState(nextState)
            #print ("nx_nxs:", nx_nxState)
            qlearning.learn(current_state, nextState, instant_reward, nextState, nx_nxState)
            print (qlearning.Q)
            return nextState
        else:
            #print "The of Episode."
            return nextState



def episode_generation(array1):
    if array1[0] != 11:
        for x in array1:
            # for x in range(0, len(array1)-1):
            print ("Array1:", array1)
            print ("x", x)
            currentState = x
            array1.remove(currentState)
            temp_array = mapping.next_State(currentState)
            print ("currentState:", currentState, "Array1 new:", array1)
            episode_generation(temp_array)
    elif array1[0] == 11:
        print "**********end of the episode************"
        return 0


def episod_gen(state):
    #print (state)
    if state != 11:
        # update(state)
        nextState = mapping.next_State(state)
        if nextState[0] == 11:
            qlearning.Q[(nextState[0])] = 0
            #print ("end of episode", sarsa.Q)
        else:
            #print (nextState)
            for i in range(0, len(nextState)):
                qlearning.Q[(state, nextState[i])] = -mapping.get_Reward(state, nextState[i])[1]
                episod_gen(nextState[i])
                #print (nextState[i], sarsa.Q)
    else:
        qlearning.Q[(state)] = 0
        #print ("The end of episode", sarsa.Q)
        return 0

training = 1000
sum16811 = 0
sum12311 = 0
sum_Time = {}
path = []

def getT(state, action, Table):
    return Table.get((state, action), 0.0)
episod_gen(currentState)
print (qlearning.Q)
for i in range(training):
    currentState = 1
    path = []
    print ("**One Episode**")
    while currentState != 11:
        nextState = update(currentState)
        path = np.append(path, [currentState, nextState])
        print (path)
        #print (currentState, nextState)
        #sum_Time[(currentState, nextState)] = sum_Time[(currentState, nextState)] + sarsa.Q[(currentState, nextState)]
        currentState = nextState
    #time = [getT(currentState, a, sum_Time) for a in mapping.next_State(currentState)]
    #max_sumT = max(time)
    #m = time.index(max_sumT)
    sum12311 = (sum12311 + qlearning.Q[(1, 2)] + qlearning.Q[(2, 3)] + qlearning.Q[(3, 11)]) / 2
    sum16811 = (sum16811 + qlearning.Q[(1, 6)] + qlearning.Q[(6, 8)] + qlearning.Q[(8, 11)]) / 2

#print (m)
print (sum12311, sum16811)

#print (sarsa.chooseState_softmax(2))
