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
import os 
from replay_buffer import ReplayBuffer

REPLAY_START_SIZE=64
BATCH_SIZE=64
LEARNING_RATE=1e-3


class BasicClone:
	"""
	A basic behavioural cloning implementation using
	a decorrolated replay buffer.
	"""

	def __init__(self, state_shape, action_shape, sess, name, load=False):
		self.sess = sess
		self.state_shape = state_shape
		self.action_shape = action_shape
		self.name = name

		self.save_dir = "./clones/{}/".format(self.name)
		if not os.path.exists(self.save_dir):
			os.mkdir(self.save_dir)


		self.memory = ReplayBuffer(name)

		with tf.variable_scope("{}_clone".format(self.name)) as clone_scope:

			self.state_ph, self.action_tensor = self.construct()
			self.expert_ph, self.train_op = self.enroll()

			scope_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=clone_scope.name)
			self.saver = tf.train.Saver(scope_vars)
			if load:
				self.load()
			else:
				init = tf.variables_initializer(scope_vars)
				self.sess.run(init)


	def construct(self):
		"""
		Constructs the feedforward graph for the behavioral clone.
		"""
		with tf.variable_scope("model"):
			state_ph = tf.placeholder(tf.float32, [None] + self.state_shape,name="state")
			action_tensor = (pt.wrap(state_ph)
							 .reshape([-1] + [sum(self.state_shape)])
				             .fully_connected(300, activation_fn=tf.nn.relu, l2loss=0.00001)
				             .fully_connected(400, activation_fn=tf.nn.relu, l2loss=0.00001)
				             .fully_connected(sum(self.action_shape), activation_fn=tf.identity))
		return state_ph, action_tensor


	def enroll(self):
		"""
		Constructs the training op for the behavioral clone.
		"""

		with tf.variable_scope("trainer"):
			expert_ph = tf.placeholder(tf.float32, [None] + self.action_shape,name="expert")
			self.loss = 0.5*tf.reduce_mean(tf.square(expert_ph - self.action_tensor))
			train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)

		return expert_ph, train_op

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

	def perceive(self, state, expert_action, train=True):
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
			return self.train(1)
			

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


			_, loss = self.sess.run([self.train_op, self.loss], {
				self.state_ph: np.array(states),
				self.expert_ph: np.array(expert_actions)
			})

			return loss


	def save(self):
		self.saver.save(self.sess, self.save_dir+ "model.ckpt")
		self.memory.save()

	def load(self):
		self.saver.restore(self.sess, self.save_dir+ "model.ckpt")
		self.memory.load()
