import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from RNNs import recurrent_networks
import sys, os

####################################################################################
#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer
####################################################################################

# generating training task
num_steps    = 5
def generate_task(num_tasks):
    '''
    Input: the number of tasks to generate
    Output: the rewards (-1/1), the probability of getting a reward and
            an integer giving the `context` meaning which cue is presented
    '''
    rand_ints = np.random.randint(1, 9, size=num_tasks)
    rand_ints = rand_ints + 1 * (rand_ints >= 5)
    proba_r   = rand_ints * 0.1
    rewards    = np.zeros([num_tasks, 2])
    random_numb = np.random.rand(num_tasks)
    rewards[:,0] = (proba_r < random_numb) * 1.
    rewards[:,1] = (proba_r >= random_numb) * 1.
    return 2 * rewards - 1, proba_r, (rand_ints - 1 - 1 * (rand_ints > 5))

num_steps_test = 5
def generate_test_task(num_tasks):
    '''
    Input: the number of tasks to generate
    Output: the rewards (-1/1), the probability of getting a reward and
            an integer giving the `context` meaning the sequence of cues presented
    '''    
    rand_ints = np.random.randint(1, 9, size=(num_tasks, num_steps_test)) #1 + np.random.randint(2, size=(num_tasks, num_steps_test)) * 7 #  
    rand_ints    = rand_ints + 1 * (rand_ints >= 5)
    proba_r      = np.mean(rand_ints * 0.1, axis=-1)
    rewards      = np.zeros([num_tasks, 2])
    random_numb  = np.random.rand(num_tasks)
    rewards[:,0] = (proba_r < random_numb) * 1.
    rewards[:,1] = (proba_r >= random_numb) * 1.
    return 2 * rewards - 1, proba_r, (rand_ints - 1 - 1 * (rand_ints > 5))

# the probabilistic tasks
class probabilistic_task():
    def __init__(self):
        self.reset()

    # resets the `num_tasks` tasks
    def reset(self, num_tasks=100, cond=0):
        self.num_tasks         = num_tasks
        self.timestep          = 0
        if cond==0: # if train
            noisy_rew, proba_r, context  = generate_task(num_tasks)
        elif cond==1: # if test
            noisy_rew, proba_r, context  = generate_test_task(num_tasks)
        else:
            assert(False)
        self.probabilistic_rewards = noisy_rew
        self.context              = context
        self.proba_r              = proba_r

    # return the rewards the `num_tasks` tasks given actions `actions`
    def pullArm(self,actions):        
        return self.probabilistic_rewards[range(self.num_tasks), np.array(actions, dtype=np.int)]

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
        self.context               = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.context_onehot        = tf.one_hot(self.context, 8, dtype=tf.float32)
        input_                     = tf.concat([self.context_onehot],1)

        self.actions        = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, 2, dtype=tf.float32)

        #Recurrent network for temporal dependencies        
        nb_units        = 48
        self.nb_units   = nb_units
        lstm_cell       = recurrent_networks.RNN(8, nb_units, noise)
        h_init          = np.zeros((1, nb_units), np.float32)
        self.state_init = [h_init]        
        self.state_in   = tf.placeholder(tf.float32, [None, nb_units])  
        self.h_noise    = tf.placeholder(tf.float32, [None, None, nb_units])
        all_noises      = self.h_noise

        if noise: 
            all_inputs         = tf.concat((input_, all_noises), axis=-1)
            rnn_in             = tf.transpose(all_inputs, [1,0,2])
        else:
            rnn_in = tf.transpose(input_,[1,0,2])

        states, self.added_noises_means = tf.scan(lstm_cell.step,
                                                  rnn_in,
                                                  initializer=(self.state_in, 0.))

        self.states_means                  = tf.reduce_mean(tf.math.abs(states), axis=(0,-1))

        self.state_out = states[-1,:]

        self.policy = slim.fully_connected(self.state_out,2, activation_fn=tf.nn.softmax,
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
        self.episode_reward_test   = []
        self.performance_oracle    = []
        self.episode_reward_test_oracle = []
        model_name                += '_noise_{0}_network_{1}'.format(noise * 1, 'RNN')
        self.summary_writer        = tf.summary.FileWriter("REINFORCEtrainings/" + str(model_name))
        self.coefficient    = coefficient
        self.ac_network = AC_Network(trainer, noise, coefficient)
        self.env      = game

    def train(self,rollout,sess):
        '''
        train method in associative learning A task
        '''
        actions           = rollout[0][0]
        rewards_ch        = rollout[0][1]
        contexts          = rollout[0][2]
        h_noises          = rollout[0][3]
        rnn_state         = rollout[0][4]

        feed_dict = {self.ac_network.h_noise:h_noises,                     
            self.ac_network.actions:actions, self.ac_network.context:contexts,
            self.ac_network.advantages:rewards_ch, self.ac_network.state_in:rnn_state}

        p_l,e_l,v_n,_ = sess.run([self.ac_network.policy_loss,
            self.ac_network.entropy,
            self.ac_network.var_norms,
            self.ac_network.apply_grads],
            feed_dict=feed_dict)
        return p_l / len(rollout),e_l / len(rollout), 0.,v_n

    def test(self, sess, num_tasks=100):
        '''
        test method in weather prediction A* task
        '''        
        self.env.reset(num_tasks=num_tasks, cond=1)

        rnn_state          = np.tile(self.ac_network.state_init[0], (num_tasks, 1))
        h_noise  = np.array(np.random.normal(size=(num_tasks, num_steps_test + 1, self.ac_network.nb_units)) * self.coefficient, dtype=np.float32)
        contexts = np.concatenate((self.env.context, -np.ones(num_tasks)[:,np.newaxis]), axis=-1)

        #Take an action using probabilities from policy network output.        
        feed_dict = {self.ac_network.context:contexts, 
                     self.ac_network.state_in:rnn_state, 
                     self.ac_network.h_noise:h_noise}

        a_dist,rnn_state_new,added_noise,state_mean = sess.run([self.ac_network.policy,self.ac_network.state_out,
                                                      self.ac_network.added_noises_means, self.ac_network.states_means], 
                                                      feed_dict=feed_dict)
        a                   = 1 * (np.random.rand(num_tasks) > np.cumsum(a_dist, axis=-1)[:,0])
        rch                 = self.env.pullArm(a)

        return rch.mean(), self.env.pullArm((self.env.proba_r > 0.5) * 1).mean()

    def work(self,gamma,sess,saver,train,num_tasks=100):
        episode_count = 0
        while True:
            episode_buffer     = []
            episode_reward     = 0
            episode_step_count = 0
                    
            rnn_state          = np.tile(self.ac_network.state_init[0], (num_tasks, 1))
            self.env.reset(num_tasks)
            state_mean_arr     = []
            added_noise_arr    = []  
            h_noise  = np.array(np.random.normal(size=(num_tasks, num_steps + 1, self.ac_network.nb_units)) * self.coefficient, dtype=np.float32)

            contexts = np.concatenate((np.tile(self.env.context, (num_steps, 1)).T, -np.ones(num_tasks)[:,np.newaxis]), axis=-1)

            #Take an action using probabilities from policy network output.
            feed_dict = {self.ac_network.context:contexts,
                         self.ac_network.state_in:rnn_state,
                         self.ac_network.h_noise:h_noise}
            a_dist,rnn_state_new,added_noise,state_mean = sess.run([self.ac_network.policy,self.ac_network.state_out,
                                                          self.ac_network.added_noises_means, self.ac_network.states_means], 
                                                          feed_dict=feed_dict)
            a                   = 1 * (np.random.rand(num_tasks) > np.cumsum(a_dist, axis=-1)[:,0])
            rch                 = self.env.pullArm(a)
            episode_step_count += 1
            episode_buffer.append([a,rch,contexts,h_noise,rnn_state])

            self.performance_oracle.append(self.env.pullArm((self.env.proba_r > 0.5) * 1))
            self.addnoises_mean_values.append(added_noise.mean())
            self.hidden_mean_values.append(np.mean(state_mean.mean()))
            self.episode_rewards.append(rch.mean())
            self.episode_lengths.append(episode_step_count)

            # Update the network using the experience buffer at the end of the episode.
            if len(episode_buffer) != 0 and train == True:
                p_l,e_l,g_n,v_n = self.train(episode_buffer,sess)

            # Periodically save summary statistics
            if episode_count % 50 == 0 and episode_count != 0:
                rch_test, rch_test_oracle = self.test(sess)
                self.episode_reward_test.append(rch_test) # reward on A* task
                self.episode_reward_test_oracle.append(rch_test_oracle) # reward upperbound on A*
                if episode_count % 500 == 0 and train == True:
                    saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
                    print("Saved Model")

                if episode_count % 5e4 == 0: # stopping criterion
                    return None                      

                mean_reward = np.mean(self.episode_rewards[-50:])
                mean_noiseadd = np.mean(self.addnoises_mean_values[-50:])
                mean_hidden   = np.mean(self.hidden_mean_values[-50:])
                summary = tf.Summary()
                summary.value.add(tag='Perf/Reward_train_homo', simple_value=float(mean_reward))                
                summary.value.add(tag='Perf/Reward_oracle_homo', simple_value=float(np.mean(self.performance_oracle[-50:])))
                summary.value.add(tag='Test/Reward_test_hetero', simple_value=float(self.episode_reward_test[-1]))
                summary.value.add(tag='Test/Reward_oracle_hetero', simple_value=float(self.episode_reward_test_oracle[-1]))
                summary.value.add(tag='Info/Noise_added', simple_value=float(mean_noiseadd))
                summary.value.add(tag='Info/Hidden_activity', simple_value=float(mean_hidden))  
                summary.value.add(tag='Parameters/biases_transition', simple_value=np.abs(sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[3])).mean())
                summary.value.add(tag='Parameters/matrix_transition', simple_value=np.abs(sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[1])).mean())                
                summary.value.add(tag='Parameters/matrix_input', simple_value=np.abs(sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[2])).mean())                                
                if train == True:
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l)/num_tasks)
                    summary.value.add(tag='Perf/Entropy', simple_value=float(e_l)/num_tasks)
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                self.summary_writer.add_summary(summary, episode_count)
                self.summary_writer.flush()

            episode_count += 1

try:
    index = int(sys.argv[1]) - 1
except:
    index = 1

coefficient_list = np.array([1.])
noise               = True
nb_coefficients  = len(coefficient_list)
idx_simul           = int(index/nb_coefficients) # simulation id
idx_coeff           = index - idx_simul * nb_coefficients # coefficient id
print('simulation id {0}, coefficient id {1}, coefficient val {2}'.format(idx_simul, idx_coeff, coefficient_list[idx_coeff]))

gamma      = .5
load_model = False
train      = True
model_name = 'model_id{0}_coeff{1}_noise{2}'.format(idx_simul, str(coefficient_list[idx_coeff]).replace('.', '_'), noise*1)

model_path = './save_models_here/' + model_name
tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path) 

trainer = tf.train.RMSPropOptimizer(learning_rate=1e-4)
worker  = Worker(probabilistic_task(),trainer,model_path,coefficient_list[idx_coeff], model_name, noise) 
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

