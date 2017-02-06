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


class BasicClone:
	"""
	A basic behavioural cloning implementation using
	a decorolated replay buffer.
	"""

	def __init__(self, state_shape, action_shape, sess, env):
		self.sess = sess
		self.env = env
		self.state_shape = state_shape
		self.action_shape = action_shape

		self.input_ph, self.output_tensor = self.construct()
		self.train_op = self.enroll()

	def construct(self):
		"""
		Constructs the feedforward graph for the behavioral clone.
		"""
		# Create input placeholders.
		pass

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

		pass

	def perceive(self, state, expert_action, train=False):
		"""
		Perceives a state and cooresponding expert action pair.
		Args:
			state -- A sample from state sapce which is not batched.
			expert_action -- The cooresponding expert action.
			train -- If the basic clone should train immediately.
			         By default, train must be called manually.
		"""


		if train:
			self.train()

	def train(self, num_iters):
		"""
		Trains the clone for an number of iterations on its perceived
		expert actions.
		Args:
			num_iters -- The number of iterations to train for.
		"""
		pass
