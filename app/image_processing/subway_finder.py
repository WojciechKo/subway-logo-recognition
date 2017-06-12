import numpy as np
import itertools
import math

class SubwayFinder:
    def find_subways(self, grouped_classifications):
        if not all(letter in grouped_classifications for letter in ("S","U","B","W","A","Y")):
            print("Can not find all parts of logo")
            return []

        sorted_classifications = [grouped_classifications[letter] for letter in ("S","U","B","W","A","Y")]
        # import pdb; pdb.set_trace()
        combinations = list(itertools.product(*sorted_classifications))

        result = []

        errors = [self._mean_squared_error(combination) for combination in combinations]
        combinations_with_errors = sorted(zip(combinations, errors), key=lambda pair: pair[1])
        while(len(combinations_with_errors) > 0):
            best, _error = self._best_fitted_combination(combinations_with_errors)
            if best == None: break

            result.append(best)
            combinations_with_errors = self._remove_combination(combinations_with_errors, best)

        return result

    def _mean_squared_error(self, classifications):
        y = [np.mean(classification.segment.box[0]) for classification in classifications]
        x = [np.mean(classification.segment.box[1]) for classification in classifications]

        a, b = np.polyfit(x, y, 1)
        prediction = np.vectorize(lambda x: a*x + b)

        return np.mean((prediction(x) - y) ** 2)

    def _best_fitted_combination(self, combinations_with_errors):
        if len(combinations_with_errors) < 1 or combinations_with_errors[0][1] > 100:
            return(None, math.inf)

        return combinations_with_errors[0]

    def _remove_combination(self, combinations, best):
        rest_combinations = []
        best_segments = [classification.segment for classification in best]

        for combination, error in combinations:
            combination_segments = [classification.segment for classification in combination]

            if (combination_segments[0] != best_segments[0] and
                    combination_segments[1] != best_segments[1] and
                    combination_segments[2] != best_segments[2] and
                    combination_segments[3] != best_segments[3] and
                    combination_segments[4] != best_segments[4] and
                    combination_segments[5] != best_segments[5]):
                rest_combinations.append((combination, error))
        return rest_combinations
