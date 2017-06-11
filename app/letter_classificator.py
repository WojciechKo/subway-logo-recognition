import numpy as np
from collections import namedtuple
from moments import HuInvariants

class LetterClassificator:
    def __init__(self, model):
        self.model = model
        self.scores = { letter: self.normalize(invariants) for letter, invariants in self.model.invariants.items() }

    def normalize(self, invariants):
        return { invariant_degree: self.z_score( value, invariant_degree) for invariant_degree, value in invariants.items() }

    def classify(self, image):
        counts = np.bincount(image.flatten())
        fillness = counts[255] / (counts[0] + counts[255])
        if fillness > 0.70:
            return (None, 0)

        segment_invariants = HuInvariants(image).invariants()
        distances = { letter: self.distance(letter, segment_invariants) for letter, _invariants in self.model.invariants.items() }

        best_fit_letter = min(distances, key=lambda key: distances[key])
        best = {k: v for k, v in distances.items() if k == best_fit_letter}
        return list(best.items())[0]

    def distance(self, letter, invariants):
        base = self.model.invariants[letter]
        normalized = self.normalize(invariants)

        select = lambda invariants: [invariants[degree] for degree in [3, 4, 7]]
        np.set_printoptions(suppress=True)
        dist = np.sum(np.absolute(np.subtract(select(self.scores[letter]), select(normalized))))

        return dist

    def z_score(self, value, invariant_degree):
        invariants = [values[invariant_degree] for values in self.model.invariants.values()]
        min_i = min(invariants)
        max_i = max(invariants)

        return (value - min_i) / (max_i - min_i)
