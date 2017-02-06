############################################
#                 clone.py                 #
#   Author: William Guss                   #
#           wguss@ml.berkeley.edu          #
#                                          #
#   A behavioural cloning implementation   #
#                                          #
############################################
import tensorflow as tf 
import numpy as np
import tf_util
import prettytensor as pt 
from replay_buffer import ReplayBuffer

REPLAY_START_SIZE=1000
BATCH_SIZE=64


class BasicClone:
	"""
	A basic behavioural cloning implementation using
	a decorrolated replay buffer.
	"""

	def __init__(self, state_shape, action_shape, sess, env, name, load=False):
		self.sess = sess
		self.env = env
		self.state_shape = state_shape
		self.action_shape = action_shape
		self.name = name


		self.memory = ReplayBuffer(name, load)
		self.state_ph, self.action_tensor = self.construct()
		self.expert_ph, self.train_op = self.enroll()

	def construct(self):
		"""
		Constructs the feedforward graph for the behavioral clone.
		"""
		with tf.variable_scope("{}_clone".format(self.name)):
			state_ph = tf.placeholder(tf.float32, [None] + self.state_shape)
			action_tensor = (pt.wrap(state_ph)
				             .flatten()
				             .fully_connected(300, activation_fn=tf.nn.relu)
				             .fully_connected(400, activation_fn=tf.nn.relu),
				             .fully_connected(sum(self.action_shape, activation_fn=tf.identity))
				             .reshape([-1] + self.action_shape))
		return state_ph, action_tensor


	def enroll(self):
		"""
		Constructs the training op for the behavioral clone.
		"""
		
		pass


	def act(self, state):
		"""
		Takes an action given a particular state.
		Args:
			state -- A sample from state space which is not 
				     batched.
		"""
		batched_action = self.sess.run(self.action_tensor, {
			self.state_ph: [state]
			})

		return batched_action[0]

	def perceive(self, state, expert_action, train=False):
		"""
		Perceives a state and cooresponding expert action pair.
		Args:
			state -- A sample from state sapce which is not batched.
			expert_action -- The cooresponding expert action.
			train -- If the basic clone should train immediately.
			         By default, train must be called manually.
		"""
		self.memory.put((state, expert_action))

		if train:
			self.train(1)

	def train(self, num_iters):
		"""
		Trains the clone for an number of iterations on its perceived
		expert actions.
		Args:
			num_iters -- The number of iterations to train for.
		"""
		if self.memory.size() > REPLAY_START_SIZE:
			batch = self.memory.get(BATCH_SIZE)
			states, expert_actions = zip(*batch)


			self.sess.run(self.train_op, {
				self.state_ph: np.array(states)
				self.expert_ph: np.array(expert_actions)
			})
