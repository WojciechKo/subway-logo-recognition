import numpy as np

class SubwayFinder:
    def find_subways(self, combinations):
        result = []

        errors = [self._mean_squared_error(combination) for combination in combinations]
        comb_with_errors = sorted(zip(combinations, errors), key=lambda pair: pair[1])
        while(len(comb_with_errors) > 0):
            best, _error = self._best_fitted_combination(comb_with_errors)
            if best == None: break

            result.append(best)
            comb_with_errors = self._remove_combination(comb_with_errors, best)

        return result

    def _mean_squared_error(self, segments):
        y = [np.mean(segment.box[0]) for segment in segments]
        x = [np.mean(segment.box[1]) for segment in segments]

        a, b = np.polyfit(x, y, 1)
        prediction = np.vectorize(lambda x: a*x + b)

        return np.mean((prediction(x) - y) ** 2)

    def _best_fitted_combination(self, comb_with_errors):
        if len(comb_with_errors) < 1 or comb_with_errors[0][1] > 100:
            return(None, math.inf)

        return comb_with_errors[0]

    def _remove_combination(self, combinations, best):
        rest_combinations = []
        for combination, error in combinations:
            if combination[0] != best[0] and combination[1] != best[1] and  combination[2] != best[2] and  combination[3] != best[3] and combination[4] != best[4] and combination[5] != best[5]:
                rest_combinations.append((combination, error))
        return rest_combinations
