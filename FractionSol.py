import math

class Fraction:

    def __init__(self, numerator, denominator):
        if denominator == 0:
            raise ArithmeticError("Sorry, no denominators equal to 0, or else there is an Arithmetic Error.")
        self.numerator = numerator
        self.denominator = denominator
        self.compact()  # not including self. will cause an error (compact is not defined) to be shown.

    def show(self):
        print(self.numerator, "/", self.denominator)

    def update(self, numerator, denominator):
        if denominator == 0:
            raise ArithmeticError("Sorry, no denominators equal to 0, or else there is an Arithmetic Error.")
        self.numerator = numerator
        self.denominator = denominator
        self.compact()

    def compact(self):
        x = math.gcd(self.numerator, self.denominator)
        self.numerator = self.numerator // x
        self.denominator = self.denominator // x

    def __add__(self, fraction2):
        # a/b + c/d = (ad + bc)/bd
        result = Fraction(0, 1)
        result.numerator = self.numerator*fraction2.denominator + self.denominator*fraction2.numerator
        result.denominator = self.denominator*fraction2.denominator
        result.compact()
        return result

    def __sub__(self, fraction2):
        # a/b - c/d = (ad - bc)/bd
        result = Fraction(0, 1)
        result.numerator = self.numerator*fraction2.denominator - self.denominator*fraction2.numerator
        result.denominator = self.denominator*fraction2.denominator
        result.compact()
        return result

    def __mul__(self, fraction2):
        # a/b * c/d = ac/bd
        result = Fraction(0, 1)
        result.numerator = self.numerator*fraction2.numerator
        result.denominator = self.denominator*fraction2.denominator
        result.compact()
        return result

    def __truediv__(self, fraction2):
        # a/b / c/d = ad/bc
        if fraction2.numerator == 0:
            raise ArithmeticError("Sorry, no denominators equal to 0, or else there is an Arithmetic Error.")
        result = Fraction(0, 1)
        result.numerator = self.numerator*fraction2.denominator
        result.denominator = self.denominator*fraction2.numerator
        result.compact()
        return result

try:
    fr1 = Fraction(2, 3)
    fr2 = Fraction(0, 5)
    ketqua = fr1.__add__(fr2)
    ketqua.show()

    fr1 = Fraction(2, 3)
    fr2 = Fraction(0, 5)
    ketqua2 = fr1.__sub__(fr2)
    ketqua2.show()

    fr1 = Fraction(2, 3)
    fr2 = Fraction(0, 5)
    ketqua3 = fr1.__mul__(fr2)
    ketqua3.show()

    fr1 = Fraction(2, 3)
    fr2 = Fraction(0, 5)
    ketqua4 = fr1.__truediv__(fr2)
    ketqua4.show()
except Exception as err:
    print("Loi:", err)