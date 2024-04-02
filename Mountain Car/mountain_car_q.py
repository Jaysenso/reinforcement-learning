import gymnasium as gym 
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training = True, render = False) :

    ## initialize the environment and reset its starting position
    env = gym.make('MountainCar-v0', render_mode = "human" if render else None)


    ## Define hyper-parameters
    segments = 20
    learning_rate_a = 0.9        ## alpha or learning rate
    discount_rate_g = 0.9        ## gamma or discount 
    epsilon = 1                  ## probability of epsilon (1 - 100% random action) - agent will choose randomly between exploration or exploitation
    epsilon_decay_rate = 2 / episodes ## agent will explore randomly at the start and starts exploiting the longer we run the simulation
    rng = np.random.default_rng()   ## random number generator
   
    rewards_per_episodes = np.zeros(episodes)
    
    ## Divide observation space (pos and velocity into segments)
    pos_space = np.linspace(env.observation_space.low[0],env.observation_space.high[0], segments)
    velocity_space = np.linspace(env.observation_space.low[1] ,env.observation_space.high[1], segments)

    ## [TRAINING] initialize the Q-table (20 x 20 x 3 array)
    if(is_training) :
        q_table = np.zeros((len(pos_space), len(velocity_space), env.action_space.n))
    ## [EVALUATION] load a pre-trained Q-table to evaluate agent
    else :
        f = open('mountain_car.pk1', 'rb')
        q_table = pickle.load(f)
        f.close()

    
    for i in range(episodes) :
        state = env.reset()[0]                              ## Starting position and velocity always 0
        print(state[0],state[1])
        state_pos = np.digitize(state[0], pos_space)        ## we throw them into their respective bins to discretize the state
        state_vel = np.digitize(state[1], velocity_space)

        done = False                                        ##True when reached goal
        total_reward = 0

        print(f"Episode : {i}")
        while not done and total_reward > -1000:
            ## [EXPLORATION] agent will randomly selects an action (0 = drive left, 1 = stay neutral, 2 = drive right)
            if is_training and rng.random() < epsilon :
                action = env.action_space.sample()
            ## [EXPLOITATION] agent will choose the best action to take in the current state based off the Q-Table
            else :
                action = np.argmax(q_table[state_pos, state_vel, :])

            ## agent will then take the chosen action and executed in the environment, transiting to the new state
            new_state,reward,done,_,_ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], velocity_space)

            ## position is clipped to [-1.2, 0.6] and velocity clipped to [-0.07, 0.07]
                
            ## introduce a reward policy (give more rewards when the velocity or position of the car is within a certain range)
            print(new_state[0],new_state[1])
            
            if new_state[0] < -0.6:
                reward += 10
            
            ## with the new_state, we will update our Q-Table (bellman Equation)
            if(is_training) :
                q_table[state_pos, state_vel, action] += learning_rate_a * (
                    reward + discount_rate_g * np.max(q_table[new_state_p, new_state_v, :]) - q_table[state_pos, state_vel, action]
                )
            
        ## we then transit current state to the new state
            state = new_state
            state_pos = new_state_p
            state_vel = new_state_v
            total_reward += reward

            print(f"Steps : {total_reward} , rewards : {reward}")

        ## we decay epsilon after every episode 
        epsilon = max(epsilon - epsilon_decay_rate, 0)
        rewards_per_episodes[i] = total_reward
    


    if(i % 2500 == 0) :
        print(f"Episode : {i} reward : {reward}")

          
    env.close()

    ## Export Q-Table for agent 's evaluation
    if is_training :
        f = open('mountain_car.pk1', 'wb')
        pickle.dump(q_table,f)
        f.close()

    ## we print out the mean rewards per 100 episodes
    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episodes[max(0, t-100): (t+1)])
    plt.plot(mean_rewards)


if __name__ == "__main__":

    # run(5000, is_training = False, render = True)
    run(500, is_training = True, render = False)
    #run(500, is_training = False, render = True)
