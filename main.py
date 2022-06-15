import sys

from mathematicaparser.odevisualization.generators import generate_equations

if __name__ == "__main__":
    generate_equations(theory_name="four_point_system", equation_path="./examples/flow_equations/four_point_system/")
