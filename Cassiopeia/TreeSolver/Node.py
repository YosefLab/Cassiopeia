import hashlib 

class Node:
	"""
	An abstract class for all nodes in a tree. Unless created manually, these nodes are created in Cassiopeia.TreeSolver.lineage_solver.solver_utils in the `node_parent`
	function. If the node parent already exists (tested by checking for equality with respect to the character states and process id) then we do not create a new node.

	Attributes:
		- name: name of node (this will either be some internal identifier or the cellBC)
		- char_vec: the array of character states, ordered by character.
		- char_string: a string representation of the char_vec, delimited by '|'. Used for quick comparisons between node character states.
		- pid: process id (useful for disambiguating between identical character states traversed on different parts of the tree)
		- is_target: boolean value indicating whether or not these nodes are targets or not.

	Methods:
		- get_character_string: utility function for getting character string
		- get_name: utility for getting the name of the node
		- get_character_vec: utility for getting the character vector
		- get_edit_distance: calculate the edit distance between two nodes
	
	"""

	def __init__(self, name, character_vec = [], is_target = True, pid = None):

		self.name = name
		self.char_vec = [str(c) for c in character_vec]
		self.char_string = '|'.join([str(c) for c in character_vec])
		self.pid = pid
		self.is_target = is_target 


	def get_character_string(self):
		return self.char_string

	def get_name(self):
		return self.name

	def get_character_vec(self):
		return self.char_vec

	def get_edit_distance(self, node2, priors=None):

		cs1, cs2 = self.get_character_string(), node2.get_character_string()
		x_list, y_list = cs1.split("|"), cs2.split("|")
		
		count = 0
		for i in range(0, len(x_list)):
			if x_list[i] == y_list[i]:
				pass
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

	def __print__(self):

		print(self.name, self.char_string)

	# def __eq__(self, other):

	# 	if isinstance(other, Node):
	# 		return (self.char_string, self.pid) == (other.char_string, other.pid)
	# 	return False

	# def __ne__(self, other):

	# 	if isinstance(other, Node):
	# 		return (self.char_string, self.pid) != (other.char_string, other.pid)
	# 	return False

	# def __gt__(self, other):

	# 	if isinstance(other, Node):
	# 		return (self.char_string, self.pid) > (other.char_string, other.pid)

	# 	raise Exception("Both items must be Nodes.")

	# def __ge__(self, other):

	# 	if isinstance(other, Node):
	# 		return (self.char_string, self.pid) >= (other.char_string, other.pid)

	# 	raise Exception("Both items must be Nodes.")

	# def __lt__(self, other):

	# 	if isinstance(other, Node):
	# 		return (self.char_string, self.pid) < (other.char_string, other.pid)

	# 	raise Exception("Both items must be Nodes.")

	# def __le__(self, other):

	# 	if isinstance(other, Node):
	# 		return (self.char_string, self.pid) <= (other.char_string, other.pid)

	# 	raise Exception("Both items must be Nodes.")


	# def __hash__(self):

	# 	#return hash((self.pid, self.get_character_string()))
	# 	return int(hashlib.md5((self.get_character_string() +  "_" + self.name + "_" + str(self.pid)).encode('utf-8')).hexdigest(), 16)




