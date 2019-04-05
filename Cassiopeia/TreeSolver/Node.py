import hashlib 

class Node:

	def __init__(self, name, character_vec, is_target = True, pid = None):

		self.name = name
		self.char_vec = [str(c) for c in character_vec]
		self.char_string = '|'.join([str(c) for c in character_vec])
		self.pid = pid
		self.is_target = is_target 

	def vect_to_string(self):

		return '|'.join(self.character_vec)

	def get_character_string(self):
		return self.char_string

	def get_name(self):
		return self.name

	def get_character_vec(self):
		return self.char_vec

	def get_edit_distance(self, node2):

		cs1, cs2 = self.get_character_string(), node2.get_character_string()
		x_list, y_list = cs1.split("|"), y_list.split("|")

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

	def __eq__(self, other):

		if isinstance(other, Node):
			return (self.name, self.char_string, self.pid) == (other.name, other.char_string, other.pid)
		return False

	def __hash__(self):

		return int(hashlib.md5((self.get_character_string() + self.name).encode('utf-8')).hexdigest(), 16)




