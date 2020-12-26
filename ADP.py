import numpy as np
import scipy.stats as st
from itertools import combinations
import CNN

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
        self.a_space = self.action_space()
        self.nA = len(self.a_space)
    
    #Initialized the environment
    def _reset(self):
        self.state = [[[],[],[],[],[]],
        [[],[],[],[],[]],
        [[[],[],[],[],[]],[[],[],[],[],[]],[[],[],[],[],[]],[[],[],[],[],[]],[[],[],[],[],[]]]]
        self.hour = 1
        self.r = 0
    
    #The stop codition
    def end(self):
        return self.hour == 168

    #The one-step-cost                
    def reward(self,action):
        r = 0
        for i in range(5):
            r += C * len(self.state[0][i])
            for k in range(5):
                r += action[0][i][k] * B[i][k]
                r += action[1][i][k] * D
        self.r = r
    
    #List all possible action. We set a overflow limit of 1 and transfer limit of 1 
    def action_space(self):
        a_space = []
        dict_o = {0:[],1:[],2:[],3:[],4:[]}
        dict_t = {0:[[0,0,0,0,0]],1:[[0,0,0,0,0]],2:[[0,0,0,0,0]],3:[[0,0,0,0,0]],4:[[0,0,0,0,0]]}
        for i in range(5):
            o = [0,0,0,0,0]
            dict_o[i].append(o)
            if len(self.state[0][i]) != 0:
                for j in range(3):
                    if len(self.state[1][warddic[i][j]]) + len(self.state[2][i][0]) + len(self.state[2][i][1]) + len(self.state[2][i][2]) + len(self.state[2][i][3]) +len(self.state[2][i][4]) >= self.N[warddic[i][j]]:
                        continue
                    else:
                        o = [0,0,0,0,0]
                        o[warddic[i][j]] = 1    
                        dict_o[i].append(o)
            for j in range(5):
                if tran[i][j] != 0:
                    if len(self.state[2][i][j]) == 0:
                        continue
                    if len(self.state[0][j]) >= 90:
                        continue
                    t = [0,0,0,0,0]
                    t[j] =  1
                    dict_t[i].append(t)
        a1_s = []
        a2_s = []    
        for i in dict_o[0]:
            for j in dict_o[1]:
                for k in dict_o[2]:
                    for h in dict_o[3]:
                        for m in dict_o[4]:
                            f = [i,j,k,h,m]
                            a1_s.append(f)
        for i in dict_t[0]:
            for j in dict_t[1]:
                for k in dict_t[2]:
                    for h in dict_t[3]:
                        for m in dict_t[4]:
                            g = [i,j,k,h,m]
                            a2_s.append(g)
        for i in a1_s:
            for j in a2_s:
                a = [i,j]
                a_space.append(a)

        return a_space                     
    
    #Calculate all possible next state
    def state_space(self,s):
        s_space = []
        d = [[],[],[],[],[]]
        t = {0:'N',1:{0:[[0,0]]},2:{0:[0]},3:'N',4:{0:[0]}}
        for i in range(5):
            #First calculate possible leaving patient and transfer patient
            if s[1][i] + sum(s[2][i]) <= self.N[i]:
                if s[1][i] != 0:
                    d_ = 0
                    while st.binom.pmf(d_,s[1][i],1/s_time[i]) >= 0.01:
                        d[i].append(d_)
                        if i == 1 and d_ > 0:
                            t[1][d_] = []
                            t_2 = 0
                            while st.binom.pmf(t_2,d_,tran[1][2]) >= 0.01:
                                t_4 = 0
                                if d_ - t_2 > 0:
                                    while st.binom.pmf(t_4,d_ - t_2,tran[1][4]) >= 0.01:
                                        t[1][d_].append([t_2,t_4])
                                        t_4 += 1
                                        if t_4 > d_ - t_2:
                                            break
                                t_2 += 1 
                                if t_2 > d_:
                                    break
                        if i == 2 and d_ > 0:
                            t[2][d_] = [0]
                            t_ = 1
                            while st.binom.pmf(t_,d_,tran[2][3]) >= 0.01:
                                t[2][d_].append(t_)
                                t_ += 1
                                if t_ > d_:
                                    break
                        if i == 4 and d_ > 0:
                            t[4][d_] = []
                            t_ = 0
                            while st.binom.pmf(t_,d_,tran[4][3]) >= 0.01:
                                t[4][d_].append(t_)
                                t_ += 1
                                if t_ > d_:
                                    break
                        d_ += 1
                        if d_ > s[1][i]:
                            break
                if s[1][i] == 0:
                    d[i].append(0)
            
            s_d = {0:[],1:[],2:[],3:[],4:[]}
            for i in range(5):
                for leave in d[i]:
                    if i == 1:
                        for j in t[1][leave]:
                            tp = sum(j)
                            if s[1][i] + sum(s[2][i]) + tp - leave >= self.N[i]:
                                a = 0
                            if s[1][i] + sum(s[2][i]) + tp - leave < self.N[i]:
                                a = 0
                                while st.poisson.pmf(a,a_rate[i][self.hour%24]) >= 0.01:
                                    tr2 = j[0]+s[2][1][2]
                                    tr4 = s[2][1][4] + j[1]
                                    tr = [0,0,0,tr2,0,tr4]
                                    if a + s[0][i] + s[1][i] + sum(s[2][i]) + tp - leave <= self.N[i]:
                                        wait = 0
                                        ward = s[1][i] - leave + tp + sum(s[2][i]) + a + s[0][i]
                                    else:
                                        wait = s[0][i] + a - leave + tp
                                        ward = self.N[i] - sum(tr)  
                                    s_d[i].append([wait,ward,tr])
                                    a += 1
                                    if a + s[0][i] - leave + tp > 90:
                                        break
                    if t[i] == 'N':
                        tp = 0
                        if s[1][i] + sum(s[2][i]) + tp - leave >= self.N[i]:
                            a = 0
                        if s[1][i] + sum(s[2][i]) + tp - leave < self.N[i]:
                            a = 0
                            while st.poisson.pmf(a,a_rate[i][self.hour%24]) >= 0.01:
                                tr = [0,0,0,0,0]
                                if a + s[0][i] + s[1][i] + sum(s[2][i]) + tp - leave <= self.N[i]:
                                    wait = 0
                                    ward = s[1][i] - leave + tp + sum(s[2][i]) + a + s[0][i]
                                else:
                                    wait = s[0][i] + a - leave + tp
                                    ward = self.N[i] - sum(tr)  
                                s_d[i].append([wait,ward,tr])
                                a += 1
                                if a + s[0][i] - leave + tp > 90:
                                    break
                    if i == 2 or i == 4:
                        for j in t[i][leave]:
                            tp = j
                            if s[1][i] + sum(s[2][i]) + tp - leave >= self.N[i]:
                                a = 0
                            if s[1][i] + sum(s[2][i]) + tp - leave < self.N[i]:
                                a = 0
                                while st.poisson.pmf(a,a_rate[i][self.hour%24]) >= 0.01:
                                    if i == 2:
                                        tr = [0,0,0,0,s[2][2][4] + j]
                                    if i == 4:
                                        tr = [0,0,0,s[2][4][3] + j,0]
                                    if a + s[0][i] + s[1][i] + sum(s[2][i]) + tp - leave <= self.N[i]:
                                        wait = 0
                                        ward = s[1][i] - leave + tp + sum(s[2][i]) + a + s[0][i]
                                    else:
                                        wait = s[0][i] + a - leave + tp
                                        ward = self.N[i] - sum(tr)  
                                    s_d[i].append([wait,ward,tr])
                                    a += 1
                                    if a + s[0][i] - leave + tp > 90:
                                        break


            
            for i in s_d[0]:
                for j in s_d[1]:
                    for k in s_d[2]:
                        for h in s_d[3]:
                            for l in s_d[4]:
                                state = [i[0],j[0],k[0],h[0],l[0]],[i[1],j[1],k[1],h[1],l[1]],[i[2],j[2],k[2],h[2],l[2]]
                                s_space.append(state)

        return s_space

    #Check if the action is valid
    def valid_action(self,action):
        for i in range(5):
            #Check waiting pool
            if len(self.state[0][i]) < action[0][i][0] + action[0][i][1] + action[0][i][2] + action[0][i][3] + action[0][i][4]:
                return False
            for j in range(5):
                #Check wards
                if action[0][i][j] + len(self.state[1][j]) + len(self.state[2][j][0]) + len(self.state[2][j][1]) + len(self.state[2][j][2]) + len(self.state[2][j][3]) + len(self.state[2][j][4]) > self.N[j]:
                    return False
        return True
           
    #The transition probability to state s
    def P(self,s,action):
        state = self.pre_step(action)
        P = 1
        for i in range(5):
            d = s[1][i] - state[1][i]
            a = s[0][i] - state[0][i]
            P *= st.poisson.pmf(a,a_rate[i][self.hour%24])
            P *= st.binom.pmf(d,state[1][i],1/s_time[i])
            for j in range(5):
                if tran[i][j] != 0:
                    c = s[2][i][j] - state[2][i][j]
                    P *= st.binom.pmf(c,d,tran[i][j])
        return P
    
    def pre_step(self,action):
        #State transfer under given action
        s = [[],[],
        [[],[],[],[],[]]]
        for i in range(5):
            xi = len(self.state[0][i])
            yi = len(self.state[1][i])
            for j in range(5):
                xi -= action[0][i][j]
                yi += action[0][j][i]
                zij = len(self.state[2][i][j]) - action[1][i][j]
                s[2][i].append(zij)
            s[0].append(xi)
            s[1].append(yi)

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
                        print(x)
                        print(self.state,j,k)
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

        return s 

def find_best(env):
    min_v = 1000000
    best_action = [[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]]
    if env.action_space() == [best_action]:
        return best_action
    for action in env.action_space():
        #print(action)
        pre_s = env.pre_step(action)
        v = env.reward(action)
        for state in env.state_space(pre_s):
            v = 1
            v += env.P(state,action) * model.predict(state)
            if v <= min_v:
                min_v = v
                best_action = action
    
    return action


n = [100,90,90,100,90]
p = pf(n)

for t in range(68):
    print(t)
    action = find_best(p)
    print(action)
    p._step(action)
    #print(p._render())

