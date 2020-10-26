import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from RNNs import recurrent_networks
import sys, os, scipy.signal

####################################################################################
#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]    
####################################################################################

num_steps    = 100
def generate_task(k=0.05):
    '''
    train task: bandit with fixed reward schedules
    Input: k, the probability of false feedback
    Output: the rewards (-1/1), and the probability of observing a reward
    '''
    rand_int     = np.random.randint(2)
    rand_int     = np.random.randint(2) 
    proba_r      = np.zeros(num_steps) 
    proba_r[:]   = (k) * (rand_int == 0) + (1. - k) * (rand_int == 1)
    rewards      = np.zeros([num_steps, 2])
    while True:
        random_numb  = np.random.rand(num_steps)        
        rewards[:,0] = (proba_r < random_numb) * 1.
        rewards[:,1] = (proba_r >= random_numb) * 1.
        if np.abs(rewards[:,1].mean() - ((0. + k) * (rand_int == 0) + (1. - k) * (rand_int == 1))) < 0.01 :
            break        
    return 2 * rewards - 1, proba_r

def generate_reversal_task():
    '''
    test task: bandit with a reversal in the middle
    Output: the rewards (-1/1), and the probability of observing a reward
    '''    
    proba_r         = np.zeros([100])
    rand_int        = np.random.randint(2) 
    default         = rand_int * 0.95 + (1 - rand_int) * 0.05
    proba_r[:50]    = default; 
    proba_r[50:]    = 1 - default;
    rewards         = np.zeros([100, 2]); random_numb = np.random.rand(100)
    rewards[:,0]    = (proba_r < random_numb) * 1.
    rewards[:,1]    = (proba_r >= random_numb) * 1.
    return 2 * rewards - 1, proba_r

# class to generate the bandit tasks
class conditioning_bandit():
    def __init__(self):
        self.reset()
        
    def set_restless_prob(self):
        self.bandit         = self.restless_rewards[self.timestep]
        
    def reset(self, reversal=False, false_positive=0.05):
        self.timestep          = 0
        if reversal==False:
            noisy_rew, _           = generate_task(k=false_positive)
        else:
            noisy_rew, _           = generate_reversal_task()
        self.restless_rewards  = noisy_rew
        self.set_restless_prob()
        
    def pullArm(self,action):
        if self.timestep >= (len(self.restless_rewards) - 1): done = True
        else: done = False
        return self.bandit[int(action)], done, self.timestep

    def update(self):
        self.timestep += 1
        self.set_restless_prob()
    
class AC_Network():
    def __init__(self, trainer, noise, coefficient):
        '''
        Returns the graph. 
        Takes as input: trainer, a tensorflow optimizer
                        noise, with computation noise (noise=1) or decision entropy (noise=0)
                        coefficient, coefficient for the computation noise or decision entropy

        '''
        if noise : regularize = 0
        else: regularize = coefficient
            
        #Input and visual encoding layers
        self.prev_rewardsch        = tf.placeholder(shape=[None,1], dtype=tf.float32)
        self.prev_actions          = tf.placeholder(shape=[None], dtype=tf.int32)
        self.prev_actions_onehot   = tf.one_hot(self.prev_actions, 2, dtype=tf.float32)
        self.timestep              = tf.placeholder(shape=[None,1], dtype=tf.float32)
        input_                     = tf.concat([self.prev_rewardsch, self.prev_actions_onehot],1)

        self.actions             = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot      = tf.one_hot(self.actions, 2, dtype=tf.float32)        

        #Recurrent network for temporal dependencies        
        nb_units = 48
        lstm_cell       = recurrent_networks.RNN(3, nb_units, noise)
        h_init          = np.zeros((1, nb_units), np.float32)
        self.state_init = [h_init]        
        self.h_in       = tf.placeholder(tf.float32, [1, nb_units])        
        self.h_noise    = tf.placeholder(tf.float32, [None, nb_units])        
        self.state_in   = self.h_in
        all_noises      = self.h_noise

        if noise: 
            all_inputs         = tf.concat((input_, all_noises), axis=1)
            rnn_in             = tf.transpose(tf.expand_dims(all_inputs, [0]),[1,0,2])
        else:
            rnn_in = tf.transpose(tf.expand_dims(input_, [0]),[1,0,2])
            
        states, self.added_noises_means    = tf.scan(lstm_cell.step, rnn_in, initializer=(self.state_in, 0.))
        self.states_means                  = tf.reduce_mean(tf.math.abs(states), axis=(-1,-2))
        
        lstm_h         = states[:,0]
        self.state_out = states[:1,0]
        rnn_out        = lstm_h

        self.policy = slim.fully_connected(rnn_out,2, activation_fn=tf.nn.softmax,
            weights_initializer=normalized_columns_initializer(0.01), biases_initializer=None)        
            
        #Get ops for loss functions and gradient updating.
        self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)
                
        self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

        #Loss functions
        self.entropy     = - tf.reduce_sum(self.policy * tf.log(self.policy + 1e-7))
        self.policy_loss = - tf.reduce_sum(tf.log(self.responsible_outputs + 1e-7) * self.advantages)
        print('regularization is {0}'.format(regularize))
        self.loss        = self.policy_loss - self.entropy * regularize
        
        self.loss_entropy = self.entropy * regularize

        #Get gradients from network using losses
        local_vars            = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.gradients        = tf.gradients(self.loss,local_vars)
        self.var_norms        = tf.global_norm(local_vars)         
        self.apply_grads      = trainer.apply_gradients(zip(self.gradients,local_vars))
                
class Worker():
    def __init__(self, game, trainer, model_path, coefficient, model_name, noise):
        self.model_path            = model_path
        self.trainer               = trainer
        self.episode_rewards       = []
        self.episode_lengths       = []
        self.addnoises_mean_values = []
        self.hidden_mean_values    = []
        self.episode_reward_reversal = []
        model_name                += '_network_{}'.format('RNN')
        self.summary_writer        = tf.summary.FileWriter("REINFORCEtrainings/" + str(model_name))
        self.coefficient    = coefficient
        self.ac_network = AC_Network(trainer, noise, coefficient)
        self.env      = game
        
    def train(self, rollout, sess, gamma, bootstrap_value):
        '''
        train method
        '''        
        rollout           = np.array(rollout)
        actions           = rollout[:,0]
        rewards_ch        = rollout[:,1]
        timesteps         = rollout[:,2]
        h_noises          = rollout[:,3]

        prev_actions      = [2] + actions[:-1].tolist()    # initialize one-hot vector representing the previous chosen action of episode to 0
        prev_rewards_ch   = [0] + rewards_ch[:-1].tolist() # initialize previous observed reward of episode to 0
        
        self.rewards_plus  = np.asarray(rewards_ch.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]

        rnn_state = self.ac_network.state_init[0]
        feed_dict = {self.ac_network.prev_rewardsch:np.vstack(prev_rewards_ch), self.ac_network.prev_actions:prev_actions,
            self.ac_network.h_noise:np.vstack(h_noises),                     
            self.ac_network.actions:actions, self.ac_network.timestep:np.vstack(timesteps),
            self.ac_network.advantages:discounted_rewards, self.ac_network.h_in:rnn_state}            
        p_l,e_l,v_n,_ = sess.run([self.ac_network.policy_loss,
            self.ac_network.entropy,
            self.ac_network.var_norms,
            self.ac_network.apply_grads],
            feed_dict=feed_dict)
        return p_l / len(rollout),e_l / len(rollout), 0.,v_n

    def test(self, sess):
        '''
            test method in reversal learning A* task
        '''        
        episode_reward     = 0
        d, a, t, rch       = False, 2, 0, 0
        rnn_state          = self.ac_network.state_init[0]
        self.env.reset(reversal=True)

        while d == False:
            h_noise = np.array(np.random.normal(size=self.ac_network.state_init[0].shape) * self.coefficient, dtype=np.float32)
            #Take an action using probabilities from policy network output.
            feed_dict = {self.ac_network.prev_rewardsch:[[rch]], self.ac_network.prev_actions:[a],
                            self.ac_network.timestep:[[t]], self.ac_network.h_in:rnn_state, self.ac_network.h_noise:h_noise}

            a_dist,rnn_state_new,added_noise,state_mean = sess.run([self.ac_network.policy,self.ac_network.state_out,
                                                          self.ac_network.added_noises_means, self.ac_network.states_means], 
                                                          feed_dict=feed_dict)
            
            a                   = np.random.choice(a_dist[0],p=a_dist[0])
            a                   = np.argmax(a_dist == a)
            rnn_state           = rnn_state_new[:2]
            rch,d,t             = self.env.pullArm(a)
            episode_reward     += ((rch + 1)/2. * 100) /100.
            if not d:
                self.env.update()
        return episode_reward
        
    def work(self, gamma, sess, saver, train):
        '''
        This is the main function
        Takes as input: gamma, the discount factor
                        sess, a Tensorflow session
                        saver, a Tensorflow saver
                        train boolean, do we train or not?
        The function will train the agent on the A task. To do so, the agent plays an A episode, and at the end of the episode, 
        we use the experience to perform a gradient update. When computation noise is assumed in the RNN, the noise realizations are 
        saved in the buffer and then fed to the back-propagation process.
        '''
        episode_count = 0
        while True:
            episode_buffer, state_mean_arr, added_noise_arr = [], [], []
            episode_reward, episode_step_count = 0, 0
            d, a, t, rch       = False, 2, 0, 0 #initialization parameters (in particular, the previous action is initialized to a null one-hot vector, a=2)
            rnn_state          = self.ac_network.state_init[0]
            self.env.reset()
            
            while d == False:
                h_noise = np.array(np.random.normal(size=self.ac_network.state_init[0].shape) * self.coefficient, dtype=np.float32)

                #Take an action using probabilities from policy network output.
                feed_dict = {self.ac_network.prev_rewardsch:[[rch]], self.ac_network.prev_actions:[a],
                                self.ac_network.timestep:[[t]], self.ac_network.h_in:rnn_state, self.ac_network.h_noise:h_noise}
                                    
                a_dist,rnn_state_new,added_noise,state_mean = sess.run([self.ac_network.policy,self.ac_network.state_out,
                                                              self.ac_network.added_noises_means, self.ac_network.states_means], 
                                                              feed_dict=feed_dict)
                a                   = np.random.choice(a_dist[0],p=a_dist[0])
                a                   = np.argmax(a_dist == a)
                rnn_state           = rnn_state_new[:2]
                rch,d,t             = self.env.pullArm(a)
                episode_reward     += ((rch + 1)/2. * 100) /(num_steps)
                episode_step_count += 1
                state_mean_arr.append(state_mean)
                added_noise_arr.append(added_noise)
                episode_buffer.append([a,rch,t,h_noise,d])
                if not d:
                    self.env.update()
            
            self.addnoises_mean_values.append(np.mean(added_noise_arr))
            self.hidden_mean_values.append(np.mean(state_mean_arr))
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_step_count)

            # Update the network using the experience buffer at the end of the episode.
            if len(episode_buffer) != 0 and train == True:
                p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,0.0)

            # Periodically save summary statistics.
            if episode_count % 50 == 0 and episode_count != 0:
                self.episode_reward_reversal.append(self.test(sess)) # reward on A* task
                if episode_count % 500 == 0 and train == True:
                    saver.save(sess, self.model_path+'/model-'+str(episode_count)+'.cptk')
                    print("Saved Model")
                        
                if episode_count % 5e4 == 0: # stopping criterion
                    return None                      

                mean_reward    = np.mean(self.episode_rewards[-50:])
                mean_noiseadd  = np.mean(self.addnoises_mean_values[-50:])
                mean_hidden    = np.mean(self.hidden_mean_values[-50:])
                mean_reversal  = np.mean(self.episode_reward_reversal[-1])
                summary = tf.Summary()
                summary.value.add(tag='Perf/reward', simple_value=float(mean_reward))
                summary.value.add(tag='Perf/reversal_Reward', simple_value=float(mean_reversal))
                summary.value.add(tag='Info/noise_added', simple_value=float(mean_noiseadd))
                summary.value.add(tag='Info/hidden_activity', simple_value=float(mean_hidden))
                summary.value.add(tag='Parameters/biases_transition', simple_value=np.abs(sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[3])).mean())
                summary.value.add(tag='Parameters/matrix_transition', simple_value=np.abs(sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[1])).mean())                
                summary.value.add(tag='Parameters/matrix_input', simple_value=np.abs(sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[2])).mean())                                
                if train == True:
                    summary.value.add(tag='Losses/policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/var Norm', simple_value=float(v_n))
                self.summary_writer.add_summary(summary, episode_count)
                self.summary_writer.flush()
                                
            episode_count += 1

try:
    index = int(sys.argv[1]) - 1
except:
    index = 0
    
coefficient_list = np.array([1.])
noise            = True
nb_coefficients  = len(coefficient_list)
idx_simul        = int(index/nb_coefficients) # simulation id
idx_coeff        = index - idx_simul * nb_coefficients # coefficient id
print('simulation id {0}, coefficient id {1}, coefficient val {2}'.format(idx_simul, idx_coeff, coefficient_list[idx_coeff]))

gamma      = .5
load_model, train = False, True
model_name = 'model_id{0}_coeff{1}_noise{2}'.format(idx_simul, str(coefficient_list[idx_coeff]).replace('.', '_'), noise*1)

model_path = './save_models_here/' + model_name
tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path) 

trainer = tf.train.RMSPropOptimizer(learning_rate=1e-4)
worker  = Worker(conditioning_bandit(),trainer,model_path,coefficient_list[idx_coeff], model_name, noise) 
saver   = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    # train
    worker.work(gamma,sess,saver,train)
