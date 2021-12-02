import numpy as np

class WordValidator:
    def __init__(self):
        with open('words.txt') as in_file:
            self.canonical_words = set([line.strip().lower() for line in in_file.readlines()])

    # Measure the string edit distance between two words
    def levenshtein_dist(self, word1: str, word2: str, stopping_point: int=100):
        distances = np.zeros((len(word1) + 1, len(word2) + 1)).astype(int)

        for t1 in range(len(word1) + 1):
            distances[t1][0] = t1
        for t2 in range(len(word2) + 1):
            distances[0][t2] = t2

        for t1 in range(1, len(word1) + 1):
            for t2 in range(1, len(word2) + 1):
                if (word1[t1-1] == word2[t2-1]):
                    distances[t1][t2] = distances[t1 - 1][t2 - 1]
                else:
                    a = distances[t1][t2 - 1]
                    b = distances[t1 - 1][t2]
                    c = distances[t1 - 1][t2 - 1]

                    min_val = min([a, b, c])
                    distances[t1][t2] = min_val + 1

            # Early stopping if we know this solution isn't as good as previous best
            if np.min(distances[t1]) > stopping_point:
                return stopping_point + 1

        return distances[len(word1)][len(word2)]

    def autocorrect_word(self, detected_word: str):
        detected_word = detected_word.lower()
        if detected_word in self.canonical_words:
            return {'words': [detected_word], 'distance': 0}

        # Only consider words with at least two characters in common with detected word
        detected_chars = set(detected_word)
        possible_matches = [word for word in self.canonical_words if len(detected_chars.intersection(set(word))) >= 2]

        # Find the closest match(es)
        best_words_so_far = []
        best_score_so_far = 100
        for possible_match in possible_matches:
            lev_score = self.levenshtein_dist(detected_word, possible_match, best_score_so_far)
            if lev_score == best_score_so_far:
                best_words_so_far.append(possible_match)
            if lev_score < best_score_so_far:
                best_words_so_far = [possible_match]
                best_score_so_far = lev_score

        return best_words_so_far, best_score_so_far
