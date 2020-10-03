"""
Approccio Random Agent, Q-Learning, SARSA

Q-Learning: l'agente impara con il QL per num_episodes e ogni 
300 episodi di training la policy viene testata su 1000 test episodes.

SARSA: l'agente impara con il SARSA per num_episodes e ogni 
300 episodi di training la policy viene testata su 1000 test episodes.


@author: Luigi Camporeale, Christian Madera, Giuseppe Pracella
"""
import numpy as np 
import gym
import matplotlib.pyplot as plt
import pandas as pd
import time
import os

from IPython.display import clear_output


def eps_greedy(Q, s, eps=0.1):
    '''
    Epsilon greedy policy
    '''
    if np.random.uniform(0,1) < eps:
        # sceglie un'azione casuale
        return np.random.randint(Q.shape[1])
    else:
        # sceglie un'azione della greedy policy
        return greedy(Q, s)


def greedy(Q, s):
    '''
    Greedy policy
    restituisce l'indice corrispondente al massimo valore action-state
    '''
    return np.argmax(Q[s])


def run_episodes(env, Q, num_episodes=100, to_print=False):
    '''
    Avvia 1000 episodi al fine di valutare la policy
    '''
    tot_rew = []
    state = env.reset()
    total_rewards = 0
    total_epochs = 0
    total_penalties = 0

    for _ in range(num_episodes):
        done = False
        game_rew = 0
        epochs, penalties, reward = 0, 0, 0

        while not done:
            # seleziona un'azione greedy
            next_state, rew, done, _ = env.step(greedy(Q, state))
            
            if reward == -10:
                penalties += 1
                
            epochs += 1
            state = next_state
            game_rew += rew 
            if done:
                state = env.reset()
                tot_rew.append(game_rew)
                
        total_penalties += penalties
        total_epochs += epochs

    if to_print:
        print(f"Results after {num_episodes} episodes:")
        print('Mean score: %.3f of %i games!'%(np.mean(tot_rew), num_episodes))
        print(f"Average timesteps per episode: {total_epochs / num_episodes}")
        print(f"Average penalties per episode: {total_penalties / num_episodes}")


    return np.mean(tot_rew)

def Q_learning(env, lr=0.01, num_episodes=10000, eps=0.3, gamma=0.95, eps_decay=0.00005):
    nA = env.action_space.n
    nS = env.observation_space.n
    start_time = time.time()

    # Inizializza la matrice Q
    # Q: matrice nS*nA dove ogni riga rappresenta uno stato ed ogni colonna rappresenza una differente azione
    Q = np.zeros((nS, nA))
    games_reward = []
    test_rewards = []
    epochs_list = []
    penalties_list = []
    exploration_rate = []

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        tot_rew = 0
        penalties = 0
        epochs = 0
        
        if ep%1000==0:
            exploration_rate.append(eps)
            print("Exploration rate", eps)
        
        # il valore epsilon subisce un decadimento fino a quando non raggiunge la soglia di 0.01
        if eps > 0.01:
            eps -= eps_decay

        # loop fino a quando l'ambiente si ferma
        while not done:
            # seleziona un'azione seguendo la eps-greedy policy
            action = eps_greedy(Q, state, eps)

            # fai un passo nell'ambiente
            next_state, rew, done, _ = env.step(action) 
            
            if rew == -10:
                penalties += 1
            
            # il Q-learning aggiorna il valore state-action (prende in considerazione il massimo valore Q per il prossimo stato)
            Q[state][action] = Q[state][action] + lr*(rew + gamma*np.max(Q[next_state]) - Q[state][action])

            state = next_state
            epochs += 1
            tot_rew += rew
            if done:
                break
        
        games_reward.append(tot_rew)

        epochs_list.append(epochs)
        penalties_list.append(penalties)
        
        # Testa la policy ogni 300 episodi e stampa i risultati
        if (ep % 300) == 0:
            test_rew = run_episodes(env, Q, 1000, True)
            print("Episode:{:5d}  Eps:{:2.4f}  Reward:{:2.4f}".format(ep, eps, test_rew))
            test_rewards.append(test_rew)
            
            
    execution_time = time.time() - start_time
    #print("--- %s secondi ---" % execution_time)             
    return Q, games_reward, test_rewards, exploration_rate, execution_time, epochs_list, penalties_list


def SARSA(env, lr=0.01, num_episodes=10000, eps=0.3, gamma=0.95, eps_decay=0.00005):
    nA = env.action_space.n
    nS = env.observation_space.n
    start_time = time.time()

    # Inizializza la matrice Q
    # Q: matrice nS*nA dove ogni riga rappresenta uno stato ed ogni colonna rappresenza una differente azione
    Q = np.zeros((nS, nA))
    games_reward = []
    test_rewards = []
    epochs_list = []
    penalties_list = []
    exploration_rate = []

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        tot_rew = 0
        penalties = 0
        epochs = 0

        if ep%1000==0:
            exploration_rate.append(eps)
            print("Exploration rate", eps)
            
        # il valore epsilon subisce un decadimento fino a quando non raggiunge la soglia di 0.01
        if eps > 0.01:
            eps -= eps_decay


        action = eps_greedy(Q, state, eps) 

        # loop fino a quando l'ambiente si ferma
        while not done:
            next_state, rew, done, _ = env.step(action) # Take one step in the environment

            if rew == -10:
                penalties += 1
            
            # sceglie la prossima azione (necessaria per l'aggiornamento SARSA)    
            next_action = eps_greedy(Q, next_state, eps) 
            
            # aggiornamento SARSA
            Q[state][action] = Q[state][action] + lr*(rew + gamma*Q[next_state][next_action] - Q[state][action])

            state = next_state
            epochs += 1
            action = next_action
            tot_rew += rew
            if done:
                break
            
        games_reward.append(tot_rew)

                
        epochs_list.append(epochs)
        penalties_list.append(penalties)

        # testa la policy ogni 300 episodi e stampa i risultati
        if (ep % 300) == 0:
            test_rew = run_episodes(env, Q, 1000, True)
            print("Episode:{:5d}  Eps:{:2.4f}  Reward:{:2.4f}".format(ep, eps, test_rew))
            test_rewards.append(test_rew)
            
            
    execution_time = time.time() - start_time
    #print("--- %s seconds ---" % execution_time)               
    return Q, games_reward, test_rewards, exploration_rate, execution_time, epochs_list, penalties_list

def Random_Agent(env, num_episodes=10000):
    nA = env.action_space.n
    nS = env.observation_space.n
    start_time = time.time()

    games_reward = []
    epochs_list = []
    penalties_list = []

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        tot_rew = 0
        penalties = 0
        epochs = 0
    
        # loop fino a quando l'ambiente si ferma
        while not done:
            # seleziona un'azione casuale
            action = env.action_space.sample()

            # fai un passo nell'ambiente
            next_state, rew, done, _ = env.step(action) 
            
            if rew == -10:
                penalties += 1

            epochs += 1
            tot_rew += rew
            if done:
                games_reward.append(tot_rew)
          
        epochs_list.append(epochs)
        penalties_list.append(penalties)
            
    execution_time = time.time() - start_time
    #print("--- %s secondi ---" % execution_time)             
    return games_reward, execution_time, epochs_list, penalties_list


if __name__ == '__main__':
    env = gym.make('Taxi-v3')
    
    Q_qlearning, games_reward_ql, test_rewards_ql, exploration_rate_ql, execution_time_ql, epochs_list_ql, penalties_list_ql = Q_learning(env, lr=.1, num_episodes=5000, eps=0.4, gamma=0.95, eps_decay=0.001)

    Q_sarsa, games_reward_sa, test_rewards_sa, exploration_rate_sa, execution_time_sa, epochs_list_sa, penalties_list_sa = SARSA(env, lr=.1, num_episodes=5000, eps=0.4, gamma=0.95, eps_decay=0.001)
    
    games_reward_ra, execution_time_ra, epochs_list_ra, penalties_list_ra = Random_Agent(env, num_episodes= 5000)
    
    print('**********Q-LEARNING Q-TABLE**********')
    print(Q_qlearning)
    
    print('**********SARSA Q-TABLE***************')
    print(Q_sarsa)
    
    
    '''
    Grafici
    
    '''
    
    df = pd.DataFrame({"Random Agent Rewards": games_reward_ra, 
             "Random Agent Steps": epochs_list_ra,
             "Random Agent Penalties": penalties_list_ra})
    df_ma = df.rolling(100, min_periods = 100).mean()
    df_ma.iloc[1:5000].plot()  
    
    df = pd.DataFrame({"Q-Learning Steps": epochs_list_ql, 
                       "SARSA Steps": epochs_list_sa})
    df_ma = df.rolling(100, min_periods = 100).mean()
    df_ma.iloc[1:5000].plot()
    
    df = pd.DataFrame({"Q-Learning Rewards": games_reward_ql, 
                       "SARSA Rewards": games_reward_sa})
    df_ma = df.rolling(100, min_periods = 100).mean()
    df_ma.iloc[1:5000].plot()
    
    df = pd.DataFrame({"Q-Learning Test": test_rewards_ql, 
             "SARSA Test": test_rewards_sa})
    df_ma = df.rolling(10, min_periods = 10).mean()
    df_ma.iloc[1:1000].plot()
    
    df = pd.DataFrame({'Algoritmi':['Q-Learning', 'SARSA', 'Random Agent'], 'Execution Time':[execution_time_ql, execution_time_sa, execution_time_ra]})
    ax = df.plot.bar(x='Algoritmi', y='Execution Time', rot=0, color='#F5C54F')
 