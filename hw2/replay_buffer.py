############################################
#             replay_buffer.py             #
#   Author: William Guss                   #
#           wguss@ml.berkeley.edu          #
#                                          #
#   The replay buffer for the clone.       #
#                                          #
############################################

import numpy as np
import lmdb

class ReplayBuffer:
	"""
	The replay buffer used to decorolate batches of expert
	policy percepts.
	Args:
		name -- The name of the replay buffer used for saving.
		load -- Whether or not the replay buffer should reload an existing store.
	"""
	def __init__(self, name, load=False):
		self.max_percepts = max_percepts
		self.name = name
		self.store = np.array([])

		if load:
			self.load()



	def put(self, percept):
		"""
		Wraps the storage mechanism for the replay buffer.
		"""
		self.store += [percept]

	def get(self, num_samples):
		"""
		Gets NUM_SAMPLES samples from the replay buffer at random.
		"""
		return np.random.choice(self.store, num_samples)

	def load(self):
		"""
		Loads the replay buffer using self.NAME.
		"""
		self.store = np.load('./replay/{}.npz'.format(self.name)).tolist()

	def save(self):
		"""
		Saves the replay buffer in a compressed format.
		"""
		np.savez('./replay/{}.npz'.format(self.name), np.array(self.store))

