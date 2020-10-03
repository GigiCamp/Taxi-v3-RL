"""
Approccio SARSA/Genetico

L'agente impara con il SARSA per num_training_episodes,
quando tutti gli agenti hanno terminato il loro training, questi
vengono valutati per num_test_episode.

La scelta dei genitori per ciascuna  generazione si basa sulla stocastic beam search,
ovvero più un agente è adatto e più probabilità ha di essere scelto. La nuova generazione
è prodotta dai due migliori. I genitori scelti faranno parte della nuova 
generazione al fine di non rischiaredi peggiorare il caso migliore.

Il Crossover effettuato è del tipo One-Point ed è fatto riga per riga.

@author: Luigi Camporeale, Christian Madera, Giuseppe Pracella
"""
import gym
import time
import random
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import clear_output

def eps_greedy(Q, s, eps=0.1):
    '''
    Epsilon greedy policy
    '''
    if np.random.uniform(0,1) < eps:
        # sceglie un'azione casuale
        return env.action_space.sample()
    else:
        # sceglie un'azione della greedy policy
        return greedy(Q, s)


def greedy(Q, s):
    '''
    Greedy policy
    restituisce l'indice corrispondente al massimo valore action-state rispetto all'agent corrente
    '''
    return np.argmax(Q[agent, state, :])

def select_best_parent(score):
    '''
    Si ricercano i due parent che presentano lo score più elevato
    
    '''
    max1=0
    max2=0
    x1=0
    x2=0
    for x in score.keys():
        if score[x]>max1:
            max2=max1
            x2=x1
            max1=score[x]
            x1=x
    return x1,x2

def select_random_parent(score):
    s=sum(list(score.values()))
    if s==0:
        p=np.random.choice(list(score.keys()), 2)
    else:
        p=np.random.choice(list(score.keys()), 2, [i/s for i in list(score.values())])
    return p[0],p[1]


def crossover(q_a,q_b):
    '''
    Viene effettuato il One-Point Crossover. Viene scelto un punto
    di crossover casuale c e le code dei due parent vengono invertite
    per ottenere dei nuovi off-springs.
    
    '''
    a,b=np.shape(q_a)
    c = random.randint(0,a-1)

    nq_a=np.zeros((a,b))
    nq_b=np.zeros((a,b))

    for i in range(0,a):
            if i<c:
                nq_a[i]=q_a[i]
                nq_b[i]=q_b[i]
            else:
                nq_a[i]=q_b[i]
                nq_b[i]=q_a[i]
    return nq_a,nq_b

env = gym.make('Taxi-v3')
action_space_size=env.action_space.n
state_space_size = env.observation_space.n

population_size=10

Q = np.random.rand(population_size,state_space_size, action_space_size)

num_generation=10
num_training_episodes = 5000
num_test_episode=1000
max_steps_per_episode = 1000

learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1.0
max_exploration_rate = 1.0
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

score=[]
index=0
max_value=0

for gen in range(num_generation):
    print("Generazione",gen)
    rewards_all_episodes = {}
    for agent in range(population_size):
        print("Agent",agent)
        exploration_rate_list = []
        exploration_rate = 1
        for ep in range(num_training_episodes):
            state = env.reset()
            done = False
            tot_rew = 0
        
            exploration_rate_list.append(exploration_rate)
            # il valore epsilon subisce un decadimento fino a quando non raggiunge la soglia di 0.01
            if exploration_rate > 0.01:
                exploration_rate -= exploration_decay_rate

            # seleziona un'azione in base alla eps-greedy policy
            action = eps_greedy(Q, state, exploration_rate)

            # loop fino a quando l'ambiente si ferma
            while not done:
                # fai un passo nell'ambiente
                next_state, rew, done, info = env.step(action)  
                
                # sceglie la prossima azione (necessaria per l'aggiornamento SARSA)
                next_action = eps_greedy(Q, next_state, exploration_rate) 
            
                # aggiornamento SARSA
                Q[agent, state, action] = Q[agent, state, action] * (1 - learning_rate) + learning_rate * (rew + discount_rate * Q[agent, next_state, next_action])

                state = next_state
                action = next_action
                tot_rew += rew
                if done:
                    #games_reward.append(tot_rew)
                    break

    for agent in range(population_size):
        for episode in range(num_test_episode):
            state = env.reset()
            done = False
            rewards_current_episode = 0
            for step in range(max_steps_per_episode):

                action = np.argmax(Q[agent,state,:])

                new_state, reward, done, info = env.step(action)
                state = new_state
                rewards_current_episode += reward
                if done == True:
                    break
            if agent in rewards_all_episodes:
                rewards_all_episodes[agent]=rewards_current_episode+rewards_all_episodes[agent]
            else:
                rewards_all_episodes[agent]=rewards_current_episode

    print(rewards_all_episodes)
    
    new_q_table=Q.copy()
    for i in range(int((population_size-2)/2)):
        g1,g2=select_random_parent(rewards_all_episodes)
        new_q_table[i], new_q_table[i+int(population_size/2)]=crossover(Q[int(g1),:,:].copy(),Q[int(g2),:,:].copy())
    g1,g2=select_best_parent(rewards_all_episodes)
    new_q_table[population_size-2]=Q[int(g1),:,:].copy()
    new_q_table[population_size-1]=Q[int(g2),:,:].copy()
    Q=new_q_table
    
    # si effettua una ricerca del massimo per trovare il tasso di successo maggiore e la generazione corrispondente
    maxx=0
    for i in rewards_all_episodes.keys():
        if maxx<rewards_all_episodes[i]:
            maxx=rewards_all_episodes[i]
    score.append(maxx)
    if maxx>max_value:
        max_value=maxx
        index=gen

print(rewards_all_episodes)

plt.plot([i for i in range(1,num_generation+1)],[s/num_test_episode for s in score],label="success rate")
print(index,max_value)
plt.plot(index+1, max_value/num_test_episode, 'ro',color='red',label="max success rate")
plt.legend()
plt.savefig("genetico_sarsa.png")