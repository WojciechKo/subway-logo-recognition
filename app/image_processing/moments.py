import numpy as np

class HuInvariants:
    def __init__(self, image):
        self.memo = {}
        self.image = image
        self.normalized_central_moments = NormalizedCentralMoments(image)

    def invariants(self):
        inv = { 1: self.invariant_1(),
                2: self.invariant_2(),
                3: self.invariant_3(),
                4: self.invariant_4(),
                5: self.invariant_5(),
                6: self.invariant_6(),
                7: self.invariant_7() }
        return inv

    def invariant_1(self):
        return self.n(2, 0) + self.n(0, 2)
    
    def invariant_2(self):
        first_part = pow((self.n(2, 0) - self.n(0, 2)), 2)
        second_part = 4 * pow(self.n(1, 1), 2)
        return first_part + second_part

    def invariant_3(self):
        first_part = pow((self.n(3, 0) - 3*self.n(1, 2)), 2)
        second_part = pow(3*self.n(2, 1) - self.n(0, 3), 2)
        return first_part + second_part

    def invariant_4(self):
        first_part = pow(self.n(3, 0) + self.n(1, 2), 2)
        second_part = pow(self.n(2, 1) + self.n(0, 3), 2)
        return first_part + second_part

    def invariant_5(self):
        first_part_1 = self.n(3, 0) - 3*self.n(1, 2)
        first_part_2 = self.n(3, 0) + self.n(1, 2)
        first_part_3 = pow(self.n(3, 0) + self.n(1, 2), 2) - 3*pow(self.n(2, 1) + self.n(0, 3), 2)
        first_part = first_part_1 * first_part_2 * first_part_3

        second_part_1 = 3*self.n(2, 1) - self.n(0, 3)
        second_part_2 = self.n(2, 1) + self.n(0, 3)
        second_part_3 = 3*pow(self.n(3, 0) + self.n(1, 2), 2) - pow(self.n(2, 1) + self.n(0, 3), 2)
        second_part = second_part_1 * second_part_2 * second_part_3

        return first_part + second_part

    def invariant_6(self):
        first_part_1 = self.n(2, 0) - self.n(0, 2)
        first_part_2 = pow(self.n(3, 0) + self.n(1, 2), 2) - pow(self.n(2, 1) + self.n(0, 3), 2)
        first_part = first_part_1 * first_part_2

        second_part_1 = 4*self.n(1, 1)
        second_part_2 = self.n(3, 0) + self.n(1, 2)
        second_part_3 = self.n(2, 1) + self.n(0, 3)
        second_part = second_part_1 * second_part_2 * second_part_3

        return first_part + second_part

    def invariant_7(self):
        first_part_1 = 3*self.n(2, 1) - self.n(0, 3)
        first_part_2 = self.n(3, 0) + self.n(1, 2)
        first_part_3 = pow(self.n(3, 0) + self.n(1, 2), 2) - 3*pow(self.n(2, 1) + self.n(0, 3), 2)
        first_part = first_part_1 * first_part_2 * first_part_3

        second_part_1 = self.n(3, 0) - 3*self.n(1, 2)
        second_part_2 = self.n(2, 1) + self.n(0, 3)
        second_part_3 = 3*pow(self.n(3, 0) + self.n(1, 2), 2) - pow(self.n(2, 1) + self.n(0, 3), 2)
        second_part = second_part_1 * second_part_2 * second_part_3

        return first_part - second_part

    def n(self, p, q):
        return self.normalized_central_moments.n(p, q)

class NormalizedCentralMoments:
    def __init__(self, image):
        self.memo = {}
        self.image = image
        self.central_moments = CentralMoments(image)

    def n(self, p, q):
        if (p, q) not in self.memo:
            self.memo[(p, q)] = self.calculate(p, q)
        return self.memo[(p, q)]
    
    def calculate(self, p, q):
        exp = ((p + q) / 2) + 1
        return self.central_moments.u(p, q) / pow(self.central_moments.u(0, 0), exp)

class CentralMoments:
    def __init__(self, image):
        self.memo = {(0, 1): 0, (1, 0): 0}
        self.image = image
        self.moments = Moments(image)

    def u(self, p, q):
        if (p, q) not in self.memo:
            self.memo[(p, q)] = self.calculate_central_moment(p, q)
        return self.memo[(p, q)]
    
    def calculate_central_moment(self, p, q):
        center_y, center_x = self.moments.center()
        result = 0.0

        it = np.nditer(self.image, flags=['multi_index'])
        while not it.finished:
            if it[0] == 255:
                y, x = it.multi_index
                result += (pow(x - center_x, p) * pow(y - center_y, q))
            it.iternext()

        return result

class Moments:
    def __init__(self, image):
        self.memo = {}
        self.image = image

    def M(self, p, q):
        if (p, q) not in self.memo:
            self.memo[(p, q)] = self.calculate_moment(p, q)
        return self.memo[(p, q)]
    
    def calculate_moment(self, p, q):
        result = 0.0

        it = np.nditer(self.image, flags=['multi_index'])
        while not it.finished:
            if it[0] == 255:
                y, x = it.multi_index
                result += (pow(x, p) * pow(y, q))
            it.iternext()

        return result
    
    def center(self):
        m_10 = self.M(1, 0)
        m_00 = self.M(0, 0)
        m_01 = self.M(0, 1)

        return (int(m_01/m_00), int(m_10/m_00))
