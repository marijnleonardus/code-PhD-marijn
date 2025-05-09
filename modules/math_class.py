import numpy as np


class Math:
    @staticmethod
    def solve_quadratic_equation(a: float, b: float, c: float) -> tuple:
        """Solves a quadratic equation using the ABC formula.
    
        Args:
            a: float
            b: float
            c: float
    
        Returns:
            x1, x2 (tuple): containing the two solutions (x1, x2) rounded to one decimal place.
    
        Raises:
            ValueError: Raises an error if the discriminant is negative, indicating no real solutions."""
    
        # Calculate the discriminant (D) using the formula: D = b^2 - 4ac
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            raise ValueError("No real solutions exist for the given quadratic equation.")
    
        # Calculate the solutions using the ABC formula: x = (-b ± √D) / (2a)
        x1 = (-b + np.sqrt(discriminant))/(2*a)
        x2 = (-b - np.sqrt(discriminant))/(2*a)
    
        # Round the solutions to one decimal place
        x1 = round(x1, 2)
        x2 = round(x2, 2)
        return x1, x2
    

def main():
    try:
        # Solve the quadratic equation: 2x^2 + 5x - 3 = 0
        solution = Math.solve_quadratic_equation(2, 5, -3)
        print(f"The solutions are: x1 = {solution[0]}, x2 = {solution[1]}.")
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
