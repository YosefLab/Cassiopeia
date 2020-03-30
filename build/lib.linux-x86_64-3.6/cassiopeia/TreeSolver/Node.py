import hashlib 
import numpy as np

class Node:
	"""
	An abstract class for all nodes in a tree. 

	Attributes:
		- name: name of node (this will either be some internal identifier or the cellBC)
		- char_vec: the array of character states, ordered by character.
		- char_string: a string representation of the char_vec, delimited by '|'. Used for quick comparisons between node character states.
		- pid: process id (useful for disambiguating between identical character states traversed on different parts of the tree)
		- is_target: boolean value indicating whether or not these nodes are targets or not.
		- support: float value indicating support of node

	Methods:
		- get_character_string: utility function for getting character string
		- get_name: utility for getting the name of the node
		- get_character_vec: utility for getting the character vector
		- get_edit_distance: calculate the edit distance between two nodes
		- get_modified_hamming_dist: calculate hamming distance between nodes, taking into account mutagenesis process
		- get_mut_length: get a 0 - 1 value indicating the proportion of non-missing characters are different between two nodes
	
	"""

	def __init__(self, name, character_vec = [], is_target = True, pid = None, support = None):
		"""
		Initiate a new Node.

		:param name:
			Name of the node
		:param character_vec:
			A list of character states, of length C. All Nodes in a tree should have the same number of characters.
		:param char_string:
			A string concatenation of the character vector, deliminted by "|"
		:param pid:
			Process ID, necessary for discriminating nodes with identical state that can appear on both sides of the tree. The process 
			ID is generated from the internal node that serves as the root of an ILP run. 
		:param is_target:
			Boolean that discriminiates between target and non-target Nodes.
		:param support:
			Float indicating support of node

		:return:
			None.

		"""


		self.name = name
		self.char_vec = [str(c) for c in character_vec]
		self.char_string = '|'.join([str(c) for c in character_vec])
		self.pid = pid
		self.is_target = is_target
		self.support = support


	def get_character_string(self):
		"""
		Utility to get the character string.

		:return:
			The character string, delimited by "|"
		"""

		return self.char_string

	def get_name(self):
		"""
		Utility to get the name of the node.

		:return:
			Name of the node (str).
		"""

		return self.name

	def get_character_vec(self):
		"""
		Utility to get the character vector. 

		:return:
			A list of strings corresponding to the state of each character.
		"""

		return self.char_vec

	def get_mut_length(self, node2, priors=None):
		"""
		Utility to calculate the number of mutations separating two nodes from one another

		:param node2:
			Node to compare against.
		:param priors:
			A dictionary representing the priors of each character state.
		:return:
			A count of the number of mutations separating the nodes.

		"""

		cs1, cs2 = self.get_character_string(), node2.get_character_string()
		x_list, y_list = cs1.split("|"), cs2.split("|")

		count = 0
		for i in range(0, len(x_list)):
			if x_list[i] == y_list[i]:
				continue
			elif y_list[i] == "-":
				count += 0

			elif x_list[i] == '0':
				if not priors:
					count += 1
				else:
					count += -np.log(priors[i][str(y_list[i])])
			else:
				return -1
		return count

	def get_modified_hamming_dist(self, node2, priors=None):
		"""
		Score the 'modified' hamming distance, where the following holds:

			- If the two states disagree and neither is 0, add 2
			- If one state is '0' and the other is not, add 1
			- Else, add 0.

		:param node2:
			Node to compare to.
		:param priors:
			A dictionary storing the prior probability of each character state arising. This is used to weight node states in agreement.

		:return:
			A score, normalized by the number of characters that were observed in both nodes.
		"""

		cs1, cs2 = self.get_character_string(), node2.get_character_string()
		x_list, y_list = cs1.split("|"), cs2.split("|")
		
		count = 0
		for i in range(0, len(x_list)):
			
			if x_list[i] == y_list[i]:
				count += 0

			elif x_list == '-' or y_list[i] == '-':
				count += 0

			elif x_list == '0' or y_list[i] == '0':
				count += 1

			else:
				count += 2

		return count
			
	def get_edit_distance(self, node2):
		"""
		Get the edit distance of the two nodes. Similar to get_mut_length() but instead normalizes by the number of characters shared
		between the two nodes, and does not take into account priors. 

		:param node2:
			Node to compare.

		:return:
			An float representing the edit distance.
		"""

		cs1, cs2 = self.get_character_vec(), node2.get_character_vec()

		count = 0
		num_present = 0

		for i in range(0, len(cs1)):
			if cs1[i] == '-' or cs2[i] == '-':
				continue
		
			num_present += 1
			if cs1[i] != cs2[i]:
				count += 1

		if num_present == 0:
			return 0

		return count / num_present

	def __print__(self):

		print(self.name, self.char_string)





