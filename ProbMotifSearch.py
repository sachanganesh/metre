import numpy as np
from BFMotifSearch import BFMotifSearch
from matplotlib import pyplot as plt

class ProbMotifSearch(BFMotifSearch):
	def __init__(self, alignment, alphabet=['A', 'C', 'G', 'T']):
		super().__init__(alignment, alphabet)
	
	def scores_histogram(self, motif_length):
		pos = np.full(self.num_seq, 0)
		max_pos = self.seq_len - motif_length

		scores = np.array([])

		i = 0
		while i >= 0:
			if i < self.num_seq - 1:
				pos, i = self.next_vertex(pos, depth=i, max_depth=self.num_seq, motif_length=motif_length, max_val=max_pos)
			else:
				scores = np.append(scores, self.score(pos, motif_length))

				pos, i = self.next_vertex(pos, depth=i, max_depth=self.num_seq, motif_length=motif_length, max_val=max_pos)

		plt.hist(scores)
		plt.show()
		
		return scores

	def probability_motif_from_profile(self, motif, profile):
		consensus = self.get_consensus(profile)
		probability = 1
		
		for i, nt in enumerate(motif):
			nt_ind = self.alphabet.index(nt)
			probability *= profile[nt_ind][i]
		
		return probability
	
	def gibbs_search(self, motif_length, random_state=None, epochs=100):
		np.random.seed(random_state)
		
		best_result = (None, 0)
		
		for epoch in range(epochs):
			# randomly select initial motif positions
			max_pos = self.seq_len - motif_length
			pos = np.random.choice(max_pos + 1, size=self.num_seq, replace=True)

			best_score = 0
			new_score = self.score(pos, motif_length)

			while new_score > best_score:
				best_score = new_score
				best_pos = pos

				# randomly pick a sequence from alignment
				rm_seq = np.random.choice(self.num_seq)
				rm_pos = pos[rm_seq]

				# remove picked sequence from alignment
				tmp_alignment = np.delete(self.alignment, rm_seq, axis=0)
				tmp_pos = np.delete(pos, rm_seq)

				# form a profile based on motif positions
				motif_alignment = self.get_subalignment(tmp_pos, num_seq=self.num_seq - 1,
														subseq_length=motif_length, alignment=tmp_alignment)
				ppm = self.get_ppm(motif_alignment)

				# compute probabilities of motif positions for removed sequence
				pos_probs = np.zeros(max_pos + 1)
				for i in range(max_pos + 1):
					motif = self.alignment[rm_seq][i: i + motif_length]
					pos_probs[i] = self.probability_motif_from_profile(motif, ppm)

				# normalize probabilities to sum to 1
				prob_sum = np.sum(pos_probs)
				
				if prob_sum == 0: # account for when all probabilities are 0
					pos_probs = np.full(max_pos + 1, 1. / (max_pos + 1))
					prob_sum = 1
				
				pos_probs /= prob_sum

				# choose a new motif position for removed sequence based on computed distribution
				pos[rm_seq] = np.random.choice(max_pos + 1, p=pos_probs)

				new_score = self.score(pos, motif_length)

			if best_score > best_result[1]:
				best_result = (best_pos, best_score)
				
		return best_result