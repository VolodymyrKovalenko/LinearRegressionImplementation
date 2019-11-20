import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class DataLengthValidationError(Exception):
    pass


class EmptySequenceValidationError(Exception):
    pass


class LinearRegressionCalculator:

    def __init__(self, x_data, y_data):

        self.validate_data(x_data, y_data)
        self.x_data = x_data
        self.y_data = y_data
        self.df = pd.DataFrame({'x': self.x_data, 'y': self.y_data})
        self.df['x_mean'] = self.df['x'].mean()
        self.df['y_mean'] = self.df['y'].mean()

    @staticmethod
    def validate_data(x_data, y_data):
        if not x_data or not y_data:
            raise EmptySequenceValidationError
        if len(x_data) != len(y_data):
            raise DataLengthValidationError

    @property
    def line_equation(self):

        m, c = self.calculate_line_coefficients()
        return f'y = {m}x + {c}'

    @property
    def line_coordinates(self):
        coordinates = []
        m, c = self.calculate_line_coefficients()
        for index, row in self.df.iterrows():
            x_point = row['x']
            y_point = m * x_point + c
            coordinates.append((x_point, y_point))

        return coordinates

    def calculate_line_coefficients(self):
        m = self.calculate_m()
        x0 = self.df['x_mean'][0]
        y0 = self.df['y_mean'][0]
        c = self.calculate_c(m, x0, y0)
        return m, c

    @property
    def r_square(self):
        sum_numerator = 0
        sum_denominator = 0

        coordinates = self.line_coordinates
        for index, row in self.df.iterrows():
            y_r = coordinates[index][1]
            sum_numerator += (y_r - row['y_mean']) ** 2
            sum_denominator += (row['y'] - row['y_mean']) ** 2
        r2 = sum_numerator / sum_denominator
        return r2

    def calculate_m(self):

        sum_numerator = 0
        sum_denominator = 0

        for index, row in self.df.iterrows():
            sum_numerator += (row['x'] - row['x_mean']) * (row['y'] - row['y_mean'])
            sum_denominator += (row['x'] - row['x_mean'])**2

        m = sum_numerator / sum_denominator
        return m

    @staticmethod
    def calculate_c(m, x0, y0):
        return y0 - (m*x0)


class LinearRegressionView(ABC):

    @abstractmethod
    def show_main_predicted_line(self, xs, ys):
        pass


class MatPlotLibLinearRegressionView(LinearRegressionView):

    def __init__(self):
        self.plt = plt

    def show_main_predicted_line(self, x_data, y_data):
        lin_regression_obj = LinearRegressionCalculator(x_data, y_data)

        coordinates = lin_regression_obj.line_coordinates
        x_values = [coordinate[0] for coordinate in coordinates]
        y_values = [coordinate[1] for coordinate in coordinates]

        self.plt.plot(x_values, y_values)
        self.plt.plot(x_data, y_data, 'ro')
        self.plt.show()


def main():
    x_arr = [1, 2, 3, 4, 5]
    y_arr = [3, 4, 2, 4, 5]

    lin_regression_view = MatPlotLibLinearRegressionView()
    lin_regression_view.show_main_predicted_line(x_arr, y_arr)


if __name__ == '__main__':
    main()



