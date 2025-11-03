# src/hmm_model.py
import numpy as np
import pickle
from typing import List, Set, Optional

ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
INDEX = {c: i for i, c in enumerate(ALPHABET)}
N_STATES = 26


class HMM:
    """
    First-order Hidden Markov Model trained on a word list.
    - Hidden states: letters a-z
    - Emissions: observed letter (or blank)
    - Provides posterior probability of each letter given the current mask.
    """

    def __init__(self, words: List[str]):
        self.pi = np.zeros(N_STATES)          # start probabilities
        self.trans = np.zeros((N_STATES, N_STATES))  # transition matrix
        self._train(words)

    # --------------------------------------------------------------------- #
    # Training
    # --------------------------------------------------------------------- #
    def _train(self, words: List[str]) -> None:
        for w in words:
            if len(w) < 2 or not w.isalpha():
                continue
            w = w.lower()
            self.pi[INDEX[w[0]]] += 1
            for i in range(len(w) - 1):
                a, b = INDEX[w[i]], INDEX[w[i + 1]]
                self.trans[a, b] += 1

        # Laplace smoothing + normalize
        self.pi += 1e-6
        self.pi /= self.pi.sum()
        for i in range(N_STATES):
            self.trans[i] += 1e-6
            self.trans[i] /= self.trans[i].sum()

    # --------------------------------------------------------------------- #
    # Forward-Backward algorithm
    # --------------------------------------------------------------------- #
    def forward_backward(self, length: int, mask: List[Optional[str]],
                         wrong_guessed: Set[str]) -> np.ndarray:
        """
        Returns gamma[t, i] = P(letter_i at position t | observations)
        """
        alpha = np.zeros((length, N_STATES))
        beta  = np.zeros((length, N_STATES))
        b     = np.ones((length, N_STATES))

        # Emission probabilities (constraints)
        for t in range(length):
            if mask[t] is not None:
                i = INDEX[mask[t]]
                b[t].fill(0.0)
                b[t, i] = 1.0
            else:
                for i, c in enumerate(ALPHABET):
                    if c in wrong_guessed:
                        b[t, i] = 0.0

        # Forward pass
        alpha[0] = self.pi * b[0]
        for t in range(1, length):
            for j in range(N_STATES):
                alpha[t, j] = b[t, j] * np.sum(alpha[t - 1] * self.trans[:, j])

        # Backward pass
        beta[length - 1] = 1.0
        for t in range(length - 2, -1, -1):
            for i in range(N_STATES):
                beta[t, i] = np.sum(self.trans[i] * b[t + 1] * beta[t + 1])

        gamma = alpha * beta
        # Normalize per position
        row_sums = gamma.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        gamma /= row_sums
        return gamma

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def get_letter_probs(self, mask: List[Optional[str]],
                         guessed: Set[str]) -> np.ndarray:
        """
        Returns a 26-dim probability vector over unguessed letters.
        """
        length = len(mask)
        revealed = {c for c in mask if c is not None}
        wrong = guessed - revealed

        gamma = self.forward_backward(length, mask, wrong)

        blank_pos = [t for t in range(length) if mask[t] is None]
        if not blank_pos:
            probs = np.ones(N_STATES) / N_STATES
        else:
            probs = gamma[blank_pos].sum(axis=0)
            probs /= probs.sum() + 1e-12

        # Zero out already guessed letters
        for i, c in enumerate(ALPHABET):
            if c in guessed:
                probs[i] = 0.0

        # Re-normalize
        total = probs.sum()
        if total == 0:
            probs[:] = 1.0 / N_STATES
        else:
            probs /= total
        return probs

    # --------------------------------------------------------------------- #
    # Persistence
    # --------------------------------------------------------------------- #
    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> 'HMM':
        with open(path, 'rb') as f:
            return pickle.load(f)