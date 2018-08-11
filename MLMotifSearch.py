import numpy as np
from BFMotifSearch import BFMotifSearch
import math

class MLMotifSearch(BFMotifSearch):
	def __init__(self, alignment, alphabet=['A', 'C', 'G', 'T']):
		super().__init__(alignment, alphabet)

	def get_ppm(self, alignment=None):
		if alignment is None:
			alignment = self.alignment
			num_seq   = self.num_seq
			seq_len   = self.seq_len
		else:
			num_seq, seq_len = alignment.shape

		# probability and laplace smoothing
		ppm = (self.get_profile(alignment) + 1) / (num_seq + len(self.alphabet))
		return ppm

	def get_pwm(self, alignment=None, background={'A':0.25, 'C':0.25, 'G':0.25, 'T':0.25}):
		if alignment is None:
			alignment = self.alignment
			num_seq   = self.num_seq
			seq_len   = self.seq_len
		else:
			num_seq, seq_len = alignment.shape

		ppm = self.get_ppm(alignment)
		pwm = np.zeros(ppm.shape)
		for i in range(len(self.alphabet)):
			for j in range(seq_len):
				pwm[i][j] = ppm[i][j] / background[self.alphabet[i]]

		pwm = np.log10(pwm) / np.log10(2)
		return pwm

	def get_num_subsequences(self, alignment):
		num_seq, seq_len = alignment.shape
		max_pos = seq_len - motif_length

		return num_seq * (max_pos + 1)

	def sample_subsequence(self, alignment, num_samples=None):
		num_seq, seq_len = alignment.shape

		max_pos = seq_len - motif_length
		visited = []

		if num_samples is None:
			num_samples = self.get_num_subsequences(alignment)

		while len(visited) < num_samples:
			i = np.random.choice(num_seq)
			j = np.random.choice(max_pos + 1)

			if (i, j) not in visited:
				visited.append((i, j))
				yield (i, j, alignment[i][j : j + motif_length])

		raise StopIteration

	def get_ppm_from_subsequence(self, subseq, lt_prob=0.5):
		other_prob = (1 - subseq_prob) / (len(self.alphabet) - 1)
		ppm = np.full((len(self.alphabet), len(subseq)), other_prob)

		for i, lt in enumerate(subseq):
			lt_ind = self.alphabet.index(lt)
			ppm[lt_ind][i] = subseq_prob

		return ppm

	def likelihood_of_motif(self, alignment, ppm, seq, pos, motif_length):
		prob_seq = 1

		for i in range(pos, pos + motif_length):
			lt = alignment[seq][i]
			lt_ind = self.alphabet.index(lt)

			prob_seq *= ppm[lt_ind][i]

	def likelihood_of_not_motif(self, seq, pos, motif_length, background):
		prob_seq = 1

		for i in range(pos, pos + motif_length):
			lt = alignment[seq][i]
			prob_seq *= background[lt]

		return prob_seq

	# def probability_of_sequence_from_motif(self, alignment, ppm, seq, pos, motif_length, background):
	# 	prob_seq = 1
	#
	# 	for lt in alignment[seq][:pos]:
	# 		prob_seq *= background[lt]
	#
	# 	for lt in alignment[seq][pos + motif_length:]:
	# 		prob_seq *= background[lt]
	#
	# 	prob_seq *= self.likelihood_of_motif(alignment, ppm seq, pos, motif_length)
	#
	# 	return prob_seq

	# motif probability matrix
	# also the E step
	def get_mpm(self, alignment, ppm, motif_length, background, lmbda):
		num_seq, seq_len = alignment.shape
		max_pos = seq_len - motif_length

		mpm = np.zeros((num_seq, max_pos + 1))

		# compute motif positional probabilities
		for i in range(num_seq):
			for j in range(max_pos + 1):
				z = self.likelihood_of_motif(alignment, ppm, i, j, motif_length) * lmbda
				mpm[i][j] = z / (z + self.likelihood_of_not_motif(seq, pos, motif_length, background) * (1 - lmbda))

		# normalize probabilities, must sum to 1
		for i in range(num_seq):
			prob_sum = np.sum(mpm[i])
			mpm[i] /= prob_sum

		return mpm

	# compute the ppm from the mpm
	# also the M step
	def get_ppm_from_mpm(self, alignment, mpm, motif_length):
		num_seq, seq_len = alignment.shape
		_, num_pos = mpm.shape

		# foreground probabilities
		ppm = np.zeros((len(self.alphabet), seq_len))

		for pos in range(num_pos):
			for lt_ind in range(len(self.alphabet)):
				for seq in range(num_seq):
					lt_positions = np.argwhere(alignment[seq] == self.alphabet[lt_ind]).flatten() - pos
					lt_positions = lt_pos[(lt_positions >= 0) & (lt_positions < num_pos)]

					ppm[lt_ind][pos] += np.sum(mpm[i][lt_positions])

			ppm[:][pos] = (ppm[:][pos] + 1) / (np.sum(ppm[:][pos]) + len(self.alphabet))

		# background probabilities
		background = {}
		pfm = self.get_profile(alignment)
		lt_freqs = np.sum(pfm, axis=1)

		for i, lt in enumerate(self.alphabet):
			f = 0
			for j in range(motif_length):
				f += pfm[i][j]

			background[lt] = lt_freqs[i] - f

		return ppm, background

	# compute the gamma and lambda values from the mpm
	# also the M step
	def get_lambda_from_mpm(self, mpm):
		num_seq, num_pos = mpm.shape
		gamma = np.sum(mpm) / num_seq

		return gamma / num_pos

	def EM(self, alignment, ppm, motif_length, background, lmbda):
		mpm = self.get_mpm(alignment, ppm, motif_length, background, lmbda)
		ppm, background = self.get_ppm_from_mpm(alignment, mpm, motif_length)
		lmbda = self.get_lambda_from_mpm(mpm)

		return mpm, ppm, background, lmbda

	def log_probability_of_subsequence_as_motif(self, alignment, ppm, seq, pos, motif_length):
		log_prob = 0

		for i in range(motif_length):
			lt = alignment[seq][pos + i]
			lt_ind = self.alphabet.index(lt)
			log_prob += math.log(ppm[lt_ind][i])

		return log_prob

	def log_probability_of_subsequence_as_background(self, alignment, seq, pos, motif_length, background):
		log_prob = 0

		for i in range(motif_length):
			lt = alignment[seq][pos + i]
			log_prob += math.log(background[lt])

		return log_prob


	def joint_log_likelihood(self, alignment, mpm, ppm, motif_length, background, lmbda):
		num_seq, seq_len = alignment.shape
		_, num_pos = mpm.shape

		likelihood = 0

		for i in range(num_seq):
			for j in range(num_pos):
				likelihood += (1 - mpm[i][j]) * self.log_probability_of_subsequence_as_background(alignment, i, j, motif_length, background) +
							  mpm[i][j] * self.log_probability_of_subsequence_as_motif(alignment, ppm, i, j, motif_length) +
							  (1 - mpm[i][j]) * math.log(1 - lmbda) + mpm[i][j] * math.log(lmbda)

		return likelihood

	def MEME(self, training_alignment, motif_length, background={'A':0.25, 'C':0.25, 'G':0.25, 'T':0.25}, alpha=0.8, epochs=1000):
		lmbda_inits = [math.sqrt(training_alignment.shape[0]) / self.get_num_subsequences(training_alignment)]
		while lmbda_inits[-1] > 1 / (2 * motif_length):
			lmbda_inits.append(lmbda_inits[-1] * 2)

		for i in range(epochs):
			best_motif_likelihood = float('-inf')
			best_motif = None

			for lmbda_i in lmbda_inits:
				best_subseq_likelihood = float('-inf')
				best_params = None
				Q = np.min([self.get_num_subsequences(alignment), math.log10(1 - alpha) / math.log10(1 - lmbda_i)])

				for seq, pos, subseq in self.sample_subsequence(training_alignment, num_samples=Q):
					lmbda = lmbda_i
					bg = background

					ppm = self.get_ppm_from_subsequence(subseq)
					mpm, ppm, bg, lmbda = self.EM(training_alignment, ppm, motif_length, bg, lmbda)

					likelihood = self.joint_log_likelihood(training_alignment, mpm, ppm, motif_length, bg, lmbda)
					if best_subseq_likelihood < likelihood:
						best_subseq_likelihood = likelihood
						best_params = (ppm, bg, lmbda)

				lmbda = lmbda_i

				old_likelihood = best_subseq_likelihood
				ppm, bg, lmbda = best_params

				delta_likelihood = float('inf')
				while delta_likelihood > 1e-6:
					mpm, ppm, bg, lmbda = self.EM(training_alignment, ppm, motif_length, bg, lmbda)

					new_likelihood = self.joint_log_likelihood(training_alignment, mpm, ppm, motif_length, bg, lmbda)
					delta_likelihood = new_likelihood - old_likelihood
					old_likelihood = new_likelihood

				if best_motif_likelihood < old_likelihood:
					best_motif_likelihood = old_likelihood
					best_motif = (ppm, bg, lmbda)
