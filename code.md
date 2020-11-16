import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

action_list = [(0, 0), (0, 2), (0, 3), (1, 0), (1, 2), (1, 3)]
rho = 0.8
gamma = 0.9
ld1 = 1.3 * rho
ld2 = 0.4 * rho
mu1 = 1
mu2 = 1 / 2
mu3 = 1
N = 10
h = [3, 1]


def P_list(s, a):
    if a == 0:
        beta = ld1 + ld2
        return [(min(s[0] + 1, N), s[1], ld1 / beta), (s[0], min(s[1] + 1, N), ld2 / beta)]
    elif a == 1:
        beta = ld1 + ld2 + mu2
        return [(min(s[0] + 1, N), s[1], ld1 / beta), (s[0], min(s[1] + 1, N), ld2 / beta),
            (max(s[0] - 1, 0), s[1], mu2 / beta)]
    elif a == 2:
        beta = ld1 + ld2 + mu3
        return [(min(s[0] + 1, N), s[1], ld1 / beta), (s[0], min(s[1] + 1, N), ld2 / beta),
            (s[0], max(s[1] - 1, 0), mu3 / beta)]
    elif a == 3:
        beta = ld1 + ld2 + mu1
        return [(min(s[0] + 1, N), s[1], ld1 / beta), (s[0], min(s[1] + 1, N), ld2 / beta),
            (max(s[0] - 1, 0), s[1], mu1 / beta)]
    elif a == 4:
        beta = ld1 + ld2 + mu1 + mu2
        return [(min(s[0] + 1, N), s[1], ld1 / beta), (s[0], min(s[1] + 1, N), ld2 / beta),
            (max(s[0] - 1, 0), s[1], (mu1 + mu2) / beta)]
    elif a == 5:
        beta = ld1 + ld2 + mu1 + mu3
        return [(min(s[0] + 1, N), s[1], ld1 / beta), (s[0], min(s[1] + 1, N), ld2 / beta),
            (max(s[0] - 1, 0), s[1], mu1 / beta), (s[0], max(s[1] - 1, 0), mu3 / beta)]
    else:
        raise Exception('no such action')


# policy iteration
iterations = 0
value_function = np.zeros((11, 11))
policy_function = np.ones((11, 11))
while True:
    policy_function_new = policy_function.copy()
    
    value_function_new = value_function.copy()
    for i in tqdm(range(N + 1)):
        for j in range(N + 1):
            value_function_new[i, j] = - (h[0] * i + h[1] * j) + gamma * sum(
                [nP * value_function[ni, nj] for ni, nj, nP in
                 P_list((i, j), policy_function[i, j])])
    iterations += 1
    if np.max(np.abs(value_function_new - value_function)) < 0.01:
        value_function = value_function_new
        break
    else:
        value_function = value_function_new
    for i in range(N + 1):
        for j in range(N + 1):
            policy_function_new[i, j] = np.argmax([sum([nP * value_function[ni, nj]
                                                        for ni, nj, nP in P_list((i, j), a)]) for a in range(6)])
    iterations += 1
    if np.max(np.abs(policy_function_new - policy_function)) < 0.01:
        policy_function = policy_function_new
        break
    else:
        policy_function = policy_function_new

print('policy iteration %d' % iterations)
print('value function:', value_function)
print('policy function:', policy_function)
# value iteration
iterations = 0
value_function = np.zeros((11, 11))
policy_function = np.ones((11, 11))
while True:
    value_function_new = value_function.copy()
    for i in tqdm(range(N + 1)):
        for j in tqdm(range(N + 1)):
            value_function_new[i, j] = - (h[0] * i + h[1] * j) + gamma * max(
                [sum([nP * value_function[ni, nj] for ni, nj, nP in
                      P_list((i, j), a)]) for a in range(6)])
    iterations += 1
    if np.max(np.abs(value_function_new - value_function)) < 0.01:
        value_function = value_function_new
        break
    else:
        value_function = value_function_new
for i in tqdm(range(N + 1)):
    for j in tqdm(range(N + 1)):
        policy_function[i, j] = np.argmax([sum([nP * value_function[ni, nj] for
                                                ni, nj, nP in P_list((i, j), a)]) for a in range(6)])
print('value iteration %d' % iterations)
print('value function:', value_function)
print('policy function:', policy_function)


def sample_action(s, policy, eps):
    rand = np.random.binomial(1, eps)
    return np.random.choice(range(6)) if rand == 1 else policy[s[0], s[1]]


# Q learning
eps = 0.3
alpha = 0.005
iterations = 0
rollout = 50
Q_function = np.random.randn(11, 11, 6)
policy_function = np.ones((11, 11), dtype=int)
best_diff = 999
for _ in range(200000):
    Q_function_new = Q_function.copy()
    S = np.random.randint(0, N + 1, 2)
    episodes = [S]
    rewards = []
    eps = max(eps - 0.000005, 0.001)
    for _ in range(rollout):
        A = sample_action(S, policy_function, eps)
    dynamic = P_list(tuple(S), A)
    Snext = dynamic[np.random.choice(range(len(dynamic)), p=[pp for _, _, pp in
                                                             dynamic])][0:2]
    episodes.append(Snext)

    Q_function_new[int(S[0]), S[1], A] = Q_function_new[int(S[0]), S[1], A] + alpha * (
            - (h[0] * S[0] + h[1] * S[1]) + gamma * np.max(Q_function_new[Snext[0], Snext[1], :]) -
            Q_function_new[S[0], S[1], A])
    rewards.append(-(h[0] * S[0] + h[1] * S[1]))
    iterations += 1
    S = Snext
    diff = np.max(np.abs(Q_function_new - Q_function))
    if diff < 0.001:
        Q_function = Q_function_new
        break
    else:
        Q_function = Q_function_new
for i in tqdm(range(N + 1)):
    for j in range(N + 1):
        policy_function[i, j] = np.argmax(Q_function[i, j, :])
# if (iterations%50000)==0:
# best_diff=diff
# print('steps:',iterations//rollout)
# print('diff:',diff)
# print('epsilon:',eps)
# print('reward:',np.mean(rewards))
# print('policy',policy_function)


print('Q iteration %d' % iterations)
print('Q function:', Q_function)
print('policy function:', policy_function)


def sample_action(s, Q, T):
    pi = np.exp(Q[s[0], s[1], :] / T)
    pi = pi / pi.sum()
    return np.random.choice(range(6), p=pi)


# boltzmann exploration
T = 5
alpha = 0.005
iterations = 0
rollout = 50
Q_function = np.zeros((11, 11, 6))
policy_function = np.ones((11, 11), dtype=int)
for _ in range(200000):
    Q_function_new = Q_function.copy()
    S = np.random.randint(0, N + 1, 2)
    episodes = [S]
    rewards = []
    T = max(T - 0.000025, 0.5)
    for _ in range(rollout):
        A = sample_action(S, Q_function, T)
    dynamic = P_list(tuple(S), A)
    Snext = dynamic[np.random.choice(range(len(dynamic)), p=[pp for _, _, pp in
                                                             dynamic])][0:2]
    episodes.append(Snext)

    Q_function_new[int(S[0]), S[1], A] = Q_function_new[int(S[0]), S[1], A] + alpha * (
            - (h[0] * S[0] + h[1] * S[1]) + gamma * np.max(Q_function_new[Snext[0], Snext[1], :]) -
            Q_function_new[S[0], S[1], A])
    rewards.append(-(h[0] * S[0] + h[1] * S[1]))
    iterations += 1
    S = Snext
    diff = np.max(np.abs(Q_function_new - Q_function))
    if diff < 0.001:
        Q_function = Q_function_new
        break
    else:
        Q_function = Q_function_new
for i in tqdm(range(N + 1)):
    for j in range(N + 1):
        policy_function[i, j] = np.argmax(Q_function[i, j, :])
print('Q iteration %d' % iterations)
print('Q function:', Q_function)
print('policy function:', policy_function)


def threshold_policy(s, b):
    if s == (0, 0):
        return range(6)
    elif (s[0] == 0) and (s[1] > 0):
        return [2, 5]
    elif (s[0] > 0) and (s[0] < b) and (s[1] == 0):
        return [3, 4, 5]
    elif (s[0] > 0) and (s[0] < b) and (s[1] > 0):
        return [4, 5]
    else:
        return [4]


# threshold policy
gamma = 0.9
for b in range(1, 11):
    iterations = 0
    value_function = np.zeros((11, 11))
    policy_function = np.ones((11, 11))

    value_function_new = value_function.copy()
    for i in range(N + 1):
        for j in range(N + 1):
            value_function_new[i, j] = - (h[0] * i + h[1] * j) + gamma * max(
                [sum([nP * value_function[ni, nj] for ni, nj, nP in
                      P_list((i, j), a)]) for a in threshold_policy((i, j), b)])
    iterations += 1
    if np.max(np.abs(value_function_new - value_function)) < 0.01:
        value_function = value_function_new
        break
    else:
        value_function = value_function_new
    for i in range(N + 1):
        for j in range(N + 1):
            action_set = threshold_policy((i, j), b)

policy_function[i, j] = action_set[np.argmax([sum([nP * value_function[ni, nj] for
                                                   ni, nj, nP in P_list((i, j), a)]) for a in action_set])]
print('b=', b)
print('value iteration %d' % iterations)
print('cost:', -value_function)
