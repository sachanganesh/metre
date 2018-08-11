import numpy as np

"""
	Sachandhan Ganesh
	June 2018

	Motif Discovery Algorithms
"""

class BFMotifSearch(object):
	def __init__(self, alignment, alphabet=['A', 'C', 'G', 'T']):
		self.alphabet  = [s.upper() for s in alphabet]					# ensure all letters in alphabet is uppercase
		self.alignment = np.core.defchararray.capitalize(alignment)		# ensure all letters in alignment are uppercase
		self.num_seq   = alignment.shape[0]								# number of sequences in alignment
		self.seq_len   = alignment.shape[1]								# length of each aligned sequences

	def get_alignment(self):
		return self.alignment

	def set_alignment(self, alignment):
		self.alignment = alignment
		self.num_seq   = len(alignment)
		self.seq_len   = len(alignment[0])

	"""
		description: compute a (Position Frequency) matrix that describes the frequency of each letter occuring at a position in the alignment
		shape: (alphabet_len, seq_len)
	"""
	def get_profile(self, alignment=None):
		if alignment is None:
			alignment = self.alignment
			num_seq   = self.num_seq
			seq_len   = self.seq_len
		else:
			num_seq, seq_len = alignment.shape

		profile = np.zeros((len(self.alphabet), seq_len))

		for seq_pos in range(seq_len):
			for seq in range(num_seq):
				for i, lt in enumerate(self.alphabet):
					if alignment[seq][seq_pos] == lt:
						profile[i][seq_pos] += 1

		return profile

	"""
		description: find the consensus string that is composed of the most frequently occurring letter at each position
		shape: (seq_len)
	"""
	def get_consensus(self, profile):
		num_lts, seq_len = profile.shape

		consensus = np.full(seq_len, '')

		for seq_pos in range(seq_len):
			majority_ind = 0

			for lt in range(1, num_lts):
				if profile[lt][seq_pos] > profile[majority_ind][seq_pos]:
					majority_ind = lt

			consensus[seq_pos] = self.alphabet[majority_ind]

		return consensus

	"""
		description: get a sub-matrix from the alignment matrix
		shape: (num_seq, subseq_len)
	"""
	def get_subalignment(self, positions, num_seq, subseq_len, alignment=None):
		if alignment is None:
			alignment = self.alignment

		subalignment = np.full((num_seq, subseq_len), '')

		for i, motif_data in enumerate(zip(alignment[:num_seq], positions[:num_seq])):
			seq, pos = motif_data
			subalignment[i] = np.array([seq[pos : pos + subseq_len]])

		return subalignment

	"""
		description: get the motif at a given position in the alignment matrix
		shape: (num_seq, motif_len)
	"""
	def get_motif(self, positions, motif_len):
		return self.get_subalignment(positions, num_seq=self.num_seq, subseq_len=motif_len)

	"""
		description: get the score for the motifs seen in the first num_seq sequences in the alignment
		shape: 1
	"""
	def partial_score(self, motif_positions, num_seq, motif_len):
		motif_alignment = self.get_subalignment(motif_positions, num_seq=num_seq, subseq_len=motif_len)

		profile = self.get_profile(motif_alignment)

		score = np.sum(np.max(profile, axis=0))
		return score

	"""
		description: get the score for motifs in all the sequences in the alignment
		shape: 1
	"""
	def score(self, motif_positions, motif_len):
		return self.partial_score(motif_positions, num_seq=self.num_seq, motif_len=motif_len)

	"""
		description: get the next combinatorial set of motif positions
		shape: (num_seq, seq_len - motif_len + 1)
	"""
	def next_positions(self, positions, motif_len, max_pos):
		positions = np.copy(positions)

		for i in reversed(range(len(positions))):
			if positions[i] < max_pos - 1:
				positions[i] += 1
				return positions

			positions[i] = 0

		return positions

	"""
		description: try every possible combination of motif positions and retain the best combination
		shape: (num_seq, seq_len - motif_len + 1), 1
	"""
	def bruteforce_search(self, motif_len):
		init_pos = np.full(self.num_seq, 0)
		pos = np.copy(init_pos)

		max_pos = self.seq_len - motif_len

		best_score = self.score(pos, motif_len)
		best_motif = pos

		num_iter = -1

		while True:
			pos = self.next_positions(pos, motif_len, max_pos)
			new_score = self.score(pos, motif_len)

			if new_score > best_score:
				best_score = new_score
				best_motif = pos

			if np.array_equal(pos, init_pos):
				return best_motif, best_score

		return best_motif, best_score

	"""
		description: get the next combinatorial set of positions while also varying the number of positions considered
		shape: (num_seq, seq_len - motif_len + 1), 1
	"""
	def next_vertex(self, vertex, num_seq, max_seq, motif_len, max_val):
		vertex = np.copy(vertex)

		if num_seq < max_seq - 1:
			vertex[num_seq + 1] = 0
			return vertex, num_seq + 1
		else:
			for i in reversed(range(max_seq)):
				if vertex[i] < max_val - 1:
					vertex[i] += 1
					return vertex, i

		return vertex, -1

	"""
		description: try every possible combination of motif positions and retain the best combination (uses next_vertex instead of next_positions)
		shape: (num_seq, seq_len - motif_len + 1), 1
	"""
	def simple_search(self, motif_len):
		pos = np.full(self.num_seq, 0)
		max_pos = self.seq_len - motif_len

		best_score = self.score(pos, motif_len)
		best_motif = pos

		i = 0
		while i >= 0:
			if i < self.num_seq - 1:
				pos, i = self.next_vertex(pos, num_seq=i, max_seq=self.num_seq, motif_len=motif_len, max_val=max_pos)
			else:
				new_score = self.score(pos, motif_len)

				if new_score > best_score:
					best_score = new_score
					best_motif = pos

				pos, i = self.next_vertex(pos, num_seq=i, max_seq=self.num_seq, motif_len=motif_len, max_val=max_pos)
		return best_motif, best_score

	"""
		description: get the next vertex after skipping the given vertex's children combinations
		shape: (num_seq, seq_len - motif_len + 1), 1
	"""
	def bypass_subtree(self, vertex, num_seq, motif_len, max_val):
		vertex = np.copy(vertex)

		for i in reversed(range(num_seq)):
			if vertex[i] < max_val - 1:
				vertex[i] += 1
				return vertex, i

		return vertex, -1

	"""
		description: using the branch-and-bound approach, find the best combination of motif positions by trying only the combinations that would guaranteeably not give worse results
		shape: (num_seq, seq_len - motif_len + 1), 1
	"""
	def efficient_search(self, motif_len):
		pos = np.full(self.num_seq, 0)
		max_pos = self.seq_len - motif_len

		best_score = self.score(pos, motif_len)
		best_motif = pos

		i = 0
		while i >= 0:
			if i < self.num_seq - 1:
				partial_score = self.partial_score(pos, num_seq=i, motif_len=motif_len)
				best_possible_score = partial_score + (self.num_seq - i) * motif_len

				if best_possible_score < best_score:
					pos, i = self.bypass_subtree(pos, i, motif_len, max_val=max_pos)
				else:
					pos, i = self.next_vertex(pos, num_seq=i, max_seq=self.num_seq, motif_len=motif_len, max_val=max_pos)
			else:
				new_score = self.score(pos, motif_len)

				if new_score > best_score:
					best_score = new_score
					best_motif = pos

				pos, i = self.next_vertex(pos, num_seq=i, max_seq=self.num_seq, motif_len=motif_len, max_val=max_pos)

		return best_motif, best_score

	"""
		description: simple CONSENSUS-like heuristic search for the best combination of motif positions
		shape: (num_seq, seq_len - motif_len + 1), 1
	"""
	def heuristic_search(self, motif_len):
		best_motif = np.full(self.num_seq, 0)
		pos = np.copy(best_motif)

		max_pos = self.seq_len - motif_len

		for pos_a in range(max_pos + 1):
			pos[0] = pos_a

			for pos_b in range(max_pos + 1):
				pos[1] = pos_b

				new_score = self.partial_score(pos, num_seq=2, motif_len=motif_len)
				if new_score > self.partial_score(best_motif, num_seq=2, motif_len=motif_len):
					best_motif[0] = pos[0]
					best_motif[1] = pos[1]

		pos[0] = best_motif[0]
		pos[1] = best_motif[1]

		for i in range(2, self.num_seq):
			for pos_p in range(0, max_pos + 1):
				pos[i] = pos_p

				new_score = self.partial_score(pos, num_seq=i + 1, motif_len=motif_len)
				best_score = self.partial_score(best_motif, num_seq=i + 1, motif_len=motif_len)

				if new_score > best_score:
					best_motif[i] = pos[i]

			pos[i] = best_motif[i]

		return best_motif, self.score(best_motif, motif_len)
