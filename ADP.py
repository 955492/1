import numpy as np
import scipy.stats as st

a_rate = [[4.633333333
,2.633333333
,2.733333333
,1.966666667
,1.633333333
,1.666666667
,1.466666667
,2.166666667
,3.533333333
,6.033333333
,7.633333333
,9.366666667
,8.666666667
,9.233333333
,7.8
,6.233333333
,6.2
,6.633333333
,6.333333333
,7.433333333
,7.9
,6.1
,6.7
,4.833333333],
[3.333333333
,2.433333333
,1.966666667
,1.166666667
,0.9
,1.033333333
,0.866666667
,1.366666667
,2.166666667
,3.5
,4.966666667
,6.133333333
,6.5
,7.233333333
,6.033333333
,5.8
,5.733333333
,6.033333333
,8.1
,6.533333333
,6.166666667
,5.4
,4.866666667
,3.866666667
],
[0.6
,0.5
,0.666666667
,0.466666667
,0.466666667
,0.633333333
,0.433333333
,0.6
,0.9
,1.833333333
,1.7
,1.3
,1.133333333
,0.833333333
,0.9
,0.966666667
,0.733333333
,0.766666667
,0.866666667
,0.833333333
,1.133333333
,0.833333333
,0.733333333
,1.066666667
],
[0.033333333
,0
,0
,0
,0
,0.066666667
,1.233333333
,10.56666667
,9.866666667
,8.733333333
,8.333333333
,5.533333333
,2.833333333
,2.2
,1.366666667
,0.3
,0.6
,0.6
,0.933333333
,1.066666667
,1.033333333
,0.633333333
,0.133333333
,0.1
],
[0.166666667
,0.333333333
,0.466666667
,0.366666667
,0.266666667
,0.2
,0.333333333
,1.1
,0.9
,0.6
,0.766666667
,0.6
,0.333333333
,0.7
,0.433333333
,0.4
,0.433333333
,0.2
,0.333333333
,0.466666667
,0.266666667
,0.566666667
,0.2
,0.333333333
]]
s_time = [67.883,19.487,45.164,67.149,121.309]
warddic = {0:[1,4,3],
           1:[4,3,2],
           2:[0,1,3],
           3:[2,1,4],
           4:[3,2,1], }
tran = [[0,0,0,0,0],
        [0,0,0.2788,0,0.0563],
        [0,0,0,0.3659,0],
        [0,0,0,0,0],
        [0,0,0,0.2678,0]]

C = 6

B = [[0,35,0,45,40],[0,0,45,40,35],[35,40,0,45,0],[0,40,35,0,45],[0,45,40,35,0]]

D = 60

class pf():
    def __init__(self,N):
        self.isd = [[[],[],[],[],[]],
        [[],[],[],[],[]],
        [[[],[],[],[],[]],[[],[],[],[],[]],[[],[],[],[],[]],[[],[],[],[],[]],[[],[],[],[],[]]]]
        self.N = N
        self._reset()

    def _reset(self):
        self.state = [[[],[],[],[],[]],
        [[],[],[],[],[]],
        [[[],[],[],[],[]],[[],[],[],[],[]],[[],[],[],[],[]],[[],[],[],[],[]],[[],[],[],[],[]]]]
        self.hour = 1
        self.r = 0
    
    def end(self):
        return self.hour == 168
    
    #The action space
    def action_space(self):
        f_space = [] 
        f = [[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]]
        dict_o = {0:[],1:[],2:[],3:[],4:[]}
        dict_t = {0:[],1:[],2:[],3:[],4:[]}
        for i in range(5):
            for j in warddic[i]:
                for xij in range(3):
                    f[0][i][j] = xij
                    if f not in f_space:
                        dict_0[i] = 
                    

    #The one-step-cost                
    def reward(self,action):
        r = 0
        for i in range(5):
            r += C * len(self.state[0][i])
            for k in range(5):
                r += action[0][i][k] * B[i][k]
                r += action[1][i][k] * D
        self.r = r
    
    #Check if the action is valid
    def valid_action(self,action):
        for i in range(5):
            #Check waiting pool
            if len(self.state[0][i]) < len(action[0][i][0]) + len(action[0][i][1]) + len(action[0][i][2]) + len(action[0][i][3]) + len(action[0][i][4]):
                return False
            for j in range(5):
                #Check wards
                if len(action[0][i][j]) + len(self.state[1][j]) + len(self.state[2][j][0]) + len(self.state[2][j][1]) + len(self.state[2][j][2]) + len(self.state[2][j][3]) + len(self.state[2][j][4]) > self.N[j]:
                    return False
           
    #The transition probability to state s
    def p(self,s,action):
        #State transfer under given action
        state = self.state
        for j in range(5):
            for k in range(5):
                number1 = action[0][j][k]
                if number1 != 0:
                    for x in range(number1):
                        state[1][k].append(self.state[0][j][0][1])
                        state[0][j].pop(0)
                        if len(state[0][j]) == 0:
                            break
                        number2 = action[1][j][k]
                #Sign transfer patients
                if number2 != 0:
                    for x in range(number2):
                        state[0][k].append([0,state[2][j][k][0]])
                        state[2][j][k].pop(0)
                        if len(state[2][j][k]) == 0:
                            break
        s = [[],[],
        [[],[],[],[],[]]]
        for i in range(2):
            for j in range(5):
                s[i].append(len(state[i][j]))

        for i in range(5):
            for j in range(5):
                s[2][i].append(len(state[2][i][j]))

        P = 1
        for i in range(5):
            d = len(s[1][i]) - len(self.state[1][i])
            a = len(s[0][i]) - len(self.state[0][i]) 
            P *= st.poisson.pmf(a,a_rate[i][t%24])
            P *= st.geom.pmf(d,s_time[i])
            for j in range(5):
                if tran[i][j] != 0:
                    c = s[2][i][j] - self.state[2][i][j]
                    P *= st.binom(c,d,tran[i][j])
        return p
    
    def pre_step(self,action):
        #State transfer under given action
        state = self.state
        for j in range(5):
            for k in range(5):
                number1 = action[0][j][k]
                if number1 != 0:
                    for x in range(number1):
                        state[1][k].append(self.state[0][j][0][1])
                        state[0][j].pop(0)
                        if len(state[0][j]) == 0:
                            break
                number2 = action[1][j][k]
                #Sign transfer patients
                if number2 != 0:
                    for x in range(number2):
                        state[0][k].append([0,state[2][j][k][0]])
                        state[2][j][k].pop(0)
                        if len(state[2][j][k]) == 0:
                            break
        s = [[],[],
        [[],[],[],[],[]]]
        for i in range(2):
            for j in range(5):
                s[i].append(len(state[i][j]))

        for i in range(5):
            for j in range(5):
                s[2][i].append(len(state[2][i][j]))
        return s

    def _step(self,action):
        #State transfer under given action
        for j in range(5):
            for k in range(5):
                number1 = action[0][j][k]
                if number1 != 0:
                    for x in range(number1):
                        self.state[1][k].append(self.state[0][j][0][1])
                        self.state[0][j].pop(0)
                        if len(self.state[0][j]) == 0:
                            break
                number2 = action[1][j][k]
                #Sign transfer patients
                if number2 != 0:
                    for x in range(number2):
                        self.state[0][k].append([0,self.state[2][j][k][0]])
                        self.state[2][j][k].pop(0)
                        if len(self.state[2][j][k]) == 0:
                            break
        #State transfer under do nothing action
        for i in range(5):
            count = 0
            #Generate transfer patients and check out patients
            while count < len(self.state[1][i]):
                if self.state[1][i][count] != 0:
                    self.state[1][i][count] -= 1
                #The serving is finished
                if self.state[1][i][count] == 0:
                    for j in range(5):
                        if tran[i][j] > np.random.rand():
                            self.state[2][i][j].append(st.geom.ppf(np.random.rand(),1/s_time[j]))
                    self.state[1][i].pop(count)
                    count -= 1
                count += 1
            #Increase waiting time 
            for j in range(len(self.state[0][i])):
                self.state[0][i][j][0] += 1
            come = int(st.poisson.ppf(np.random.rand(),a_rate[i][self.hour%24]))
            #Set a limit to the waiting pool
            if come > 90 + (self.N[i] - len(self.state[1][i]) - len(self.state[2][i][0]) - len(self.state[2][i][1]) - len(self.state[2][i][2]) - len(self.state[2][i][3]) - len(self.state[2][i][4]) ) - len(self.state[0][i]):
                come = 90 + (self.N[i] - len(self.state[1][i]) - len(self.state[2][i][0]) - len(self.state[2][i][1]) - len(self.state[2][i][2]) - len(self.state[2][i][3]) - len(self.state[2][i][4]) ) - len(self.state[0][i])
            #Generate arrival patients
            for j in range(come):
                self.state[0][i].append([0,st.geom.ppf(np.random.rand(),1/s_time[i])])
            
        #Sign transfer patient
        for i in range(5):
            for j in range(5):
                if len(self.state[2][j][i]) != 0:
                    if len(self.state[1][j]) + len(self.state[2][i]) + len(self.state[2][i][0]) + len(self.state[2][i][1]) + len(self.state[2][i][2]) + len(self.state[2][i][3]) + len(self.state[2][i][4]) < self.N[i]:
                        if np.random.rand() > 0.5:
                            self.state[1][i].append(self.state[2][j][i][0])
                            self.state[2][j][i].pop(0)
            #Sign primary patient               
            if len(self.state[1][i]) + len(self.state[0][i]) + len(self.state[2][i][0]) + len(self.state[2][i][1]) + len(self.state[2][i][2]) + len(self.state[2][i][3]) + len(self.state[2][i][4]) <= self.N[i]:
                for j in range(len(self.state[0][i])):
                    if len(self.state[0][i]) == 0:
                            break
                    self.state[1][i].append(self.state[0][i][0][1])
                    self.state[0][i].pop(0)
            else:
                for j in range(self.N[i] - (len(self.state[1][i]) + len(self.state[2][i][0]) + len(self.state[2][i][1]) + len(self.state[2][i][2]) + len(self.state[2][i][3]) + len(self.state[2][i][4]))):
                    if len(self.state[0][i]) == 0:
                            break
                    self.state[1][i].append(self.state[0][i][0][1])
                    self.state[0][i].pop(0)

        self.hour += 1
        self.reward(action)
    
    def _render(self):
        s = [[],[],
        [[],[],[],[],[]]]
        for i in range(2):
            for j in range(5):
                s[i].append(len(self.state[i][j]))

        for i in range(5):
            for j in range(5):
                s[2][i].append(len(self.state[2][i][j]))

        return s , self.r


def policy_evaluattion(env,policy,gamma = 0.9):
    s_space = env.s_space()
    f_space = env.f_space()
    min_reward = 0
    for f in f_space:
        reward_ = (1 - gamma) * env.reward(f_space)
        if min_reward == 0:
            min_reward = reward_
            policy = f
        for s in s_space():
            P = env.p(state)
            reward_ -= gamma * P * model(s)
        if reward_ <= min_reward:
            min_reward = reward
            policy = f
    return f

        
def policy_improvement(env,policy):


    



p = pf([])