import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ThetaSequence:
    def __init__(self, seq: np.ndarray, n_elements: int):
        self.seq = seq
        self.n_elem = n_elements
        self.mean = np.mean(seq)
        self.var = np.var(seq)
        self.max_theta = np.max(seq)
        self.std_dev = np.std(seq)
def generate_theta_sequence_set(n_elem, n_sets):
    theta_seq_set = []
    for _ in range(n_sets):
        theta_seq = ThetaSequence(np.random.uniform(0, 1, size=n_elem), n_elem)
        theta_seq_set.append(theta_seq)
    return np.array(theta_seq_set)
def specific_theta_generate(mean, std_dev, theta1=np.random.uniform(0, 1)):
    coff_a = 2
    coff_b = 2 * theta1 - 6 * mean
    coff_c = 6 * (mean ** 2) - 3 * (std_dev ** 2) + (2 * theta1 ** 2) - 6 * mean * theta1
    count = 0
    while count < 1000:
        count += 1
        delta = coff_b ** 2 - 4 * coff_a * coff_c
        # print("delta problem")
        if delta >= 0:
            theta2 = (-coff_b + np.sqrt(delta)) / (2 * coff_a)
            theta3 = (-coff_b - np.sqrt(delta)) / (2 * coff_a)
            if 0 <= theta2 <= 1 and 0 <= theta3 <= 1:
                return ThetaSequence(np.array([theta1, theta2, theta3]),3)
        theta1 = np.random.uniform(0, 1)
        coff_b = 2 * theta1 - 6 * mean
        coff_c = 6 * (mean ** 2) - 3 * (std_dev ** 2) + (2 * theta1 ** 2) - 6 * mean * theta1
    else:
        return None
def theta_generate_static_mean_rolling_std_dev(mean, start_std_dev, end_std_dev, span):
    theta_set = []
    for rolling_std_dev in np.arange(start_std_dev, end_std_dev, span):
        theta_group = specific_theta_generate(mean, rolling_std_dev)
        if not theta_group == None:
            theta_set.append(theta_group)
    return np.array(theta_set)
def theta_generate_static_std_dev_rolling_mean(std_dev, start_mean, end_mean, span):
    theta_set = []
    for rolling_mean in np.arange(start_mean, end_mean, span):
        theta_group = specific_theta_generate(rolling_mean, std_dev)
        if not theta_group == None:
            theta_set.append(theta_group)
    return np.array(theta_set)


## Above are interface
y_delta_greedy_static_mean = []
y_delta_greedy_static_std_dev = []

num_epsilons = 101 
epsilon_values = np.linspace(0,1, num_epsilons)
num_std_devs = 41   
std_devs = np.linspace(0, np.sqrt(1/6), num_std_devs)
num_trials = 5000
rewards = np.zeros((num_epsilons, num_std_devs))
oracle_values = np.zeros((num_epsilons, num_std_devs))

mean=[0.25,0.5,0.75]
for mean_var in mean:
    theta = theta_generate_static_mean_rolling_std_dev(mean_var, 0.001, 0.401, 0.01)
    for i, epsilon in enumerate(epsilon_values):
        for j in range(len(theta)):
            count=[0,0,0]
            theta_estimates=[0,0,0]
            total_reward = 0
            reward=0
            arm=0
            for _ in range(num_trials):
                if np.random.rand() < epsilon:
                    arm = np.random.choice([0, 1, 2])
                else:
                    arm = np.argmax(theta_estimates)
                if np.random.rand() < theta[j].seq[arm]:
                    reward = 1
                else:
                    reward = 0
                count[arm] += 1
                theta_estimates[arm] += (1/count[arm]) * (reward - theta_estimates[arm])
                total_reward += reward
            oracle_value = num_trials * theta[j].max_theta
            rewards[i, j] = total_reward
            oracle_values[i, j] = oracle_value
    rewards_differences = np.abs(rewards - oracle_values)
    
    res_reward  = [num for num in rewards_differences[10] if abs(num) > 1]
    y_delta_greedy_static_mean.append(res_reward)
    

    
    epsilon_values_2d = np.repeat(epsilon_values[:, np.newaxis], num_std_devs, axis=1)
    std_devs_2d = np.tile(std_devs[np.newaxis, :], (num_epsilons, 1))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(epsilon_values_2d.flatten(), std_devs_2d.flatten(), rewards_differences.flatten(),
                        c=rewards_differences.flatten(), cmap='viridis', depthshade=True)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Absolute Difference from Oracle Value')
    ax.set_xlabel('Epsilon')
    ax.set_ylabel('Standard Deviation')
    ax.set_zlabel('Absolute Difference from Oracle Value')
    
    plt.show()

    
num_epsilons = 101 
epsilon_values = np.linspace(0,1, num_epsilons)
num_std_devs = 100   
std_devs = np.linspace(0, np.sqrt(1/6), num_std_devs)
num_trials = 5000
rewards = np.zeros((num_epsilons, num_std_devs))
oracle_values = np.zeros((num_epsilons, num_std_devs))
var=[0.102,0.204,0.306]
for var_var in var:
    theta = theta_generate_static_std_dev_rolling_mean(var_var, 0, 1, 0.01)
    for i, epsilon in enumerate(epsilon_values):
        for j in range(len(theta)):
            count=[0,0,0]
            theta_estimates=[0,0,0]
            total_reward = 0
            reward=0
            arm=0
            for _ in range(num_trials):
                if np.random.rand() < epsilon:
                    arm = np.random.choice([0, 1, 2])
                else:
                    arm = np.argmax(theta_estimates)
                if np.random.rand() < theta[j].seq[arm]:
                    reward = 1
                else:
                    reward = 0
                count[arm] += 1
                theta_estimates[arm] += (1/count[arm]) * (reward - theta_estimates[arm])
                total_reward += reward
            oracle_value = num_trials * theta[j].max_theta
            rewards[i, j] = total_reward
            oracle_values[i, j] = oracle_value
    rewards_differences = np.abs(rewards - oracle_values)

    res_reward  = [num for num in rewards_differences[10] if abs(num) > 1]
    y_delta_greedy_static_mean.append(res_reward)


    epsilon_values_2d = np.repeat(epsilon_values[:, np.newaxis], num_std_devs, axis=1)
    std_devs_2d = np.tile(std_devs[np.newaxis, :], (num_epsilons, 1))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(epsilon_values_2d.flatten(), std_devs_2d.flatten(), rewards_differences.flatten(),
                        c=rewards_differences.flatten(), cmap='viridis', depthshade=True)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Absolute Difference from Oracle Value')
    ax.set_xlabel('Epsilon')
    ax.set_ylabel('mean')
    ax.set_zlabel('Absolute Difference from Oracle Value')
    plt.show()

#共400次个点

