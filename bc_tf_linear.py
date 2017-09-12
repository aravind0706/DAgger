import tensorflow as tf
import numpy as np
import pickle
import gym
import argparse
import tf_util
import load_policy
import matplotlib.pyplot as plt
import os

"""
Usage: python bc_tf_linear.py expert_data_s_a_Humanoid-v1.pkl Humanoid-v1 --obs_dim 376 --act_dim 17 --num_epochs 20 --num_updates 100 --num_expert_rollouts 120 --plot --render --table
       python bc_tf_linear.py expert_data_s_a_Reacher-v1.pkl Reacher-v1 --obs_dim 11 --act_dim 2 --num_epochs 20 --num_updates 100 --num_expert_rollouts 25 --plot --render --table
       python bc_tf_linear.py expert_data_s_a_Hopper-v1.pkl Hopper-v1 --obs_dim 11 --act_dim 3 --num_epochs 20 --num_updates 100 --num_expert_rollouts 25 --plot --render --table
"""


class Network(object):

    def __init__(self, inp_dim = 376, out_dim = 17):

        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.W1 = tf.Variable(tf.truncated_normal((inp_dim, out_dim), stddev = 0.1))
        self.b1 = tf.Variable(tf.constant(.1, shape = (out_dim,)))
        #self.W2 = tf.Variable(tf.truncated_normal((100,100), stddev = 0.1))
        #self.b2 = tf.Variable(tf.constant(.1, shape = (100,)))
        #self.W3 = tf.Variable(tf.truncated_normal((100,out_dim), stddev = 0.1))
        #self.b3 = tf.Variable(tf.constant(.1, shape = (out_dim,)))
        self.obs = tf.placeholder(tf.float32, [None, inp_dim])
        self.act = tf.placeholder(tf.float32, [None, out_dim])
        self.optimizer = tf.train.AdamOptimizer(0.005)
        self._build_graph()

    def _build_graph(self):
        self.act_pred = tf.matmul(self.obs, self.W1) + self.b1
        #h2 = tf.nn.tanh(tf.matmul(h1, self.W2) + self.b2)
        #self.act_pred = tf.matmul(h1, self.W3) + self.b3
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.act_pred - self.act), reduction_indices = [1]))
        self.train_step = self.optimizer.minimize(self.loss)

    def train(self, obs, acts, sess):

        sess.run(self.train_step, feed_dict = {self.obs: obs, self.act: acts})

    def validate_loss(self, obs, acts, sess):
        
        loss = sess.run(self.loss, feed_dict = {self.obs: obs, self.act: acts})
        return np.sqrt(loss)

    def predict(self, obs, sess):
        
        #obs = obs.reshape(-1, 1)
        acts_predicted = sess.run(self.act_pred, feed_dict = {self.obs: obs})
        return acts_predicted

def random_batch_sampling(dataset,  obs_dim, act_dim, batch_size = 128):
    obs_batch = np.zeros((batch_size, obs_dim))
    acts_batch = np.zeros((batch_size, act_dim))
    max_idx = dataset.shape[0]
    for i in range(batch_size):
        idx = np.random.RandomState().randint(0, max_idx)
        obs_batch[i] = dataset[idx, :obs_dim]
        acts_batch[i] = dataset[idx, obs_dim:]

    return [obs_batch, acts_batch]

def rollout_network(env, network, sess, mean, stddev, num_rollouts, render):

    max_time_steps = env.spec.timestep_limit
    returns = []
    for _ in range(num_rollouts):
        obs = env.reset() 
        if render:
            env.render()
        done = False
        traj_return = 0
        counter = 0
        while not done:
            action = network.predict((obs.reshape((1,-1))- mean)/stddev, sess)
            obs, r, done, i = env.step(action)
            traj_return += r
            counter += 1
            if render:
                env.render()
            if counter > max_time_steps:
                break
        returns.append(traj_return)
    return returns


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_data', type = str)
    parser.add_argument('envname', type = str)
    parser.add_argument('--act_dim', type = int)
    parser.add_argument('--obs_dim', type = int)
    parser.add_argument('--num_epochs', type = int, default = 20, help = 'Number of training epochs')
    parser.add_argument('--num_updates', type = int, default = 100, help = 'Number of minibatch updates in an epoch')
    parser.add_argument('--num_rollouts', type= int, default = 20, help = 'Number of rollouts for averaging statistics')
    parser.add_argument('--plot', action = 'store_true', help = 'Whether to plot performance vs epoch')
    parser.add_argument('--render', action = 'store_true', help = 'Whether to render test rollouts')
    parser.add_argument('--table', action = 'store_true', help = 'Whether to print table of best statistic')
    parser.add_argument('--num_expert_rollouts', type = int, default = 120, help = 'Number of expert rollouts for dataset')
    args = parser.parse_args()
    os.system('python run_expert.py experts/' + args.envname + '.pkl ' + args.envname + ' --num_rollouts ' + str(args.num_expert_rollouts))
    exp_data = pickle.load(open(args.expert_data, 'rb'))
    env = gym.make(args.envname)
    obs_dim = args.obs_dim
    act_dim = args.act_dim
    #exp_data = pickle.load(open('expert_data_s_a_Humanoid-v1.pkl', 'rb'))
    exp_obs = exp_data['observations']
    exp_acts = exp_data['actions'].reshape((-1, act_dim))
    exp_obs_mean = np.mean(exp_obs, axis = 0)
    exp_obs_std = np.std(exp_obs, axis = 0) + 1e-3
    exp_obs = (exp_obs - exp_obs_mean) / exp_obs_std
    exp_o_a = np.column_stack((exp_obs, exp_acts)) 
    for _ in range(5):
        np.random.shuffle(exp_o_a)
    
    # Train the network with the current dataset , do 1500 batch updates 
    returns_mean = [] # Store the average return after each DAgger iteration in this list for it to be plotted
    returns_std = []
    with tf.Session() as sess:
        network = Network(inp_dim = args.obs_dim, out_dim = args.act_dim)
        sess.run(tf.global_variables_initializer())
        #tf_util.initialize()
    # Now rollout the trained network to collect more data points and append it to the dataset
        for num_iter in range(args.num_epochs):
            for _ in range(args.num_updates):
	        obs_batch, acts_batch = random_batch_sampling(exp_o_a, obs_dim, act_dim)
	        network.train(obs_batch, acts_batch, sess)
            traj_returns = rollout_network(env, network, sess, exp_obs_mean, exp_obs_std, args.num_rollouts, args.render)
            traj_returns = np.array(traj_returns)
            traj_returns_mean = np.mean(traj_returns)
            traj_returns_std = np.std(traj_returns)
            returns_mean.append(traj_returns_mean)
            returns_std.append(traj_returns_std)
            print("BC Iter", (num_iter +1), "Mean Return", traj_returns_mean, "StdDev of Return", traj_returns_std)
            #returns.append({'mean': traj_returns_mean, 'sdev': traj_returns_std})
    returns = {'mean': returns_mean, 'std': returns_std}
    pickle.dump(returns, open('bc_returns_linear' + args.envname + '.pkl', 'wb'))
    returns_mean = np.array(returns_mean)
    returns_std = np.array(returns_std)
    
    if args.render:
        plt.figure()
        xaxis = np.arange(1, args.num_epochs+1)
        plt.errorbar(xaxis, returns_mean, yerr = returns_std)
        plt.xlabel('BC Iter')
        plt.ylabel('Mean Return')
        plt.title('Behavior Cloning Learning Curve (Linear) ' + args.envname)
        plt.savefig('BC_Curve_Linear'+ args.envname + '.png')
    
    if args.table:
    #### Print Table for best average Return#####
        print("Table for best average return")
        idx = np.argmax(returns_mean)
        print("Average Return", returns_mean[idx], "Standard Deviation", returns_std[idx])

if __name__ == '__main__':
    main()
