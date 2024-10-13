import numpy as np
from scipy.linalg import svd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from itertools import permutations

class Quadric:
    """
    Represents a general quadric surface.

    # TODO: Implement a canonicalize method that returns R, T so that if [X, Y, Z] = R * [x,y,z] + T, then the quadric is ax*2 + by*2 + cz*2 + d = 0. with d=0 or 1
    """

    def __init__(self):
        self.coefficients = None  # Will store the quadric coefficients

    def compute_gaussian_weights(self, X, Y, Z, variance=1):
        """
        Compute Gaussian weights for the points based on their distance from the origin.

        This will encourage the surface to fit the data near the center more accurately.
        
        Parameters:
            X, Y, Z (numpy.ndarray): Coordinates of the points.
            variance (float): Variance for the Gaussian weight function.
            
        Returns:
            numpy.ndarray: Array of weights.
        """
        # Compute squared distances from the origin
        distances_squared = X**2 + Y**2 + Z**2
        
        # Compute Gaussian weights
        weights = np.exp(-distances_squared / (2 * variance**2))
        
        return weights       

    def fit(self, points, weights=None):
        """
        Fits a general quadric surface to the data points.

        Parameters:
            points (numpy.ndarray): Nx3 array of points.

        Sets:
            self.coefficients (numpy.ndarray): Array of quadric coefficients.
        """
        # Extract X, Y, Z coordinates
        X = points[:, 0]
        Y = points[:, 1]
        Z = points[:, 2]

        # Build the design matrix
        D = np.column_stack([X**2, Y**2, Z**2, X * Y, X * Z, Y * Z, X, Y, Z, np.ones_like(X)])

        if weights is not None:
            D = np.sqrt(np.clip(weights[:, None], a_min=0.0001)) * D

        # Solve for the quadric coefficients using SVD
        _, _, Vt = svd(D)

        quadric_coefficients = Vt[-1, :]

        # Normalize coefficients
        quadric_coefficients /= np.linalg.norm(quadric_coefficients)

        self.coefficients = quadric_coefficients

    def evaluate(self, x, y, z):
        """
        Evaluates the quadric equation at the given point(s).

        Parameters:
            x, y, z (numpy.ndarray): Coordinates where to evaluate.

        Returns:
            numpy.ndarray: Value of the quadric equation at the given points.
        """
        A, B, C, D, E, F, G, H, I, J = self.coefficients

        return (A * x**2 + B * y**2 + C * z**2 +
                D * x * y + E * x * z + F * y * z +
                G * x + H * y + I * z + J)
    
    def extract_quadratic_matrix(self):
        """
        Extracts the quadratic matrix from the quadric coefficients and performs eigen-decomposition.

        Why?
        Because the eigenvectors of the largest eigenvalues in magnitude are the X and Y axes of the rotated frame.
        They are the directions where the principal curvature is most extreme.

        They span the plane which will be the domain of our functional quadratic surface

        Returns:
            tuple: (eigenvalues, eigenvectors)
        """
        # The linear coefficients are not relevant for this problem, so we can ignore them
        A, B, C, D, E, F, _, _, _, _ = self.coefficients

        # Construct the symmetric quadratic matrix
        Q = np.array([
            [A, D / 2, E / 2],
            [D / 2, B, F / 2],
            [E / 2, F / 2, C]
        ])

        return Q




class RotatedQuadratic:
    """
    Represents a rotated functional quadratic surface.
    """

    def __init__(self, a=0, b=0, rotation_matrix=np.eye(3), translation_vector=np.zeros(3)):
        """
        Represents a rotated quadratic surface.

        The quadratic surface in the rotated frame is defined by:
            z = a * x^2 + b * y^2

        The 3D coordinates in the original frame are obtained by applying the rotation and translation:
            [X, Y, Z]^T = R * [x, y, z]^T + T

        Parameters:
            a, b (float): Shape coefficients in the rotated frame.
            rotation_matrix (numpy.ndarray): 3x3 rotation matrix (R).
            translation_vector (numpy.ndarray): 3-element translation vector (T).
        """
        self.a = a  # Coefficient for x^2 in rotated frame
        self.b = b  # Coefficient for y^2 in rotated frame
        self.rotation_matrix = rotation_matrix  # 3x3 rotation matrix R
        self.translation_vector = translation_vector  # 3-element translation vector T
        self.quadric = None

    def __repr__(self):
        return f"RotatedQuadratic(a={self.a}, b={self.b}, rotation_matrix={self.rotation_matrix}, translation_vector={self.translation_vector})"

    @staticmethod
    def from_euler_angles(a, b, azimuth, elevation, roll, translation_vector=None):
        """
        Creates a rotated quadratic surface from Euler angles.

        Parameters:
            a, b (float): Coefficients for x^2 and y^2 in the rotated frame.
            azimuth, elevation, roll (float): Rotation angles in degrees.
            translation_vector (numpy.ndarray, optional): 3-element translation vector.

        Returns:
            RotatedQuadratic: Rotated quadratic surface.
        """
        # Create rotation matrix from Euler angles
        rotation = R.from_euler('zyx', [azimuth, elevation, roll], degrees=True)
        rotation_matrix = rotation.as_matrix()
        if translation_vector is None:
            translation_vector = np.zeros(3)
        return RotatedQuadratic(a, b, rotation_matrix, translation_vector)

    def apply(self, UV):
        """
        Applies the quadratic surface equation to an array of UV values and returns the corresponding XYZ points.

        Parameters:
            UV (numpy.ndarray): Nx2 array of (U, V) values in the parameter space. 

        Returns:
            numpy.ndarray: Nx3 array of (X, Y, Z) points on the quadratic surface with rotation and translation applied.
        """
        x = UV[:, 0]
        y = UV[:, 1]

        # Compute z in the rotated frame using the quadratic equation
        z = self.a * x**2 + self.b * y**2

        # Stack x, y, z into Nx3 array
        points_rotated = np.column_stack((x, y, z))

        # Apply rotation and translation: [X, Y, Z]^T = R * [x, y, z]^T + T
        points = (self.rotation_matrix @ points_rotated.T).T + self.translation_vector

        return points
    
    def untransformed(self, points):
        """
        Applies the inverse rotation and translation to an array of XYZ points and returns the corresponding xyz values.

        Parameters:
            points (numpy.ndarray): Nx3 array of (X, Y, Z) points in the rotated frame. 

        Returns:
            numpy.ndarray: Nx3 array of (x, y, z) values in the parameter spaces, where z=a*x**2+b*y**2
        """
        points_rotated = points - self.translation_vector

        # Apply inverse rotation: [x, y, z]^T = R^T * [X, Y, Z]^T
        points_unrotated = (self.rotation_matrix.T @ points_rotated.T).T

        return points_unrotated

    def fit(self, points, weights='gaussian', axis_order=None):
        """
        Fits a rotated quadratic surface to the data points.

        Parameters:
            points (numpy.ndarray): Nx3 array of points.
            weights (str or numpy.ndarray): Weights for fitting, default is 'gaussian'.
            axis_order (tuple or None): Specifies which axes to use for fitting (e.g., (0, 1, 2) for X, Y, Z).
                                        If None, tries all permutations and selects the one with the minimum error.

        Sets:
            self.rotation_matrix: Rotation matrix aligning the quadric with coordinate axes.
            self.translation_vector: Translation vector.
            self.a, self.b: Coefficients in rotated frame.

        Also Sets:    
            self.coefficients: Original Quadric coefficients, upon which the rotation is based.
        """
        # Fit general quadric
        self.quadric = Quadric()
        center = np.mean(points, axis=0)
        var = max([np.var(points[:, i]) for i in range(3)]) / 3.0
        centered_points = points - center

        if weights == 'gaussian':
            weights = self.quadric.compute_gaussian_weights(*points.T, variance=var)

        self.quadric.fit(centered_points, weights)

        # Extract Q and perform eigen-decomposition
        Q = self.quadric.extract_quadratic_matrix()
        eigenvalues, eigenvectors = np.linalg.eigh(Q)

        if axis_order is None:
            # Try all permutations of axes and choose the best fit based on error
            best_fit = None
            min_error = np.inf
            for perm in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
                rotation_matrix = self.get_rotation_matrix(eigenvalues, eigenvectors, axis_order=perm)
                rotated_points = self.rotate_points(centered_points, rotation_matrix)
                a, b, translations_rotated = self.fit_quadratic_in_rotated_frame(rotated_points)
                error = self.compute_fitting_error(rotated_points, a, b, translations_rotated)

                if error < min_error:
                    min_error = error
                    best_fit = (rotation_matrix, a, b, translations_rotated)

            self.rotation_matrix, self.a, self.b, translations_rotated = best_fit
        else:
            # Use specified axis_order for rotation and fitting
            self.rotation_matrix = self.get_rotation_matrix(eigenvalues, eigenvectors, axis_order=axis_order)
            rotated_points = self.rotate_points(centered_points, self.rotation_matrix)
            self.a, self.b, translations_rotated = self.fit_quadratic_in_rotated_frame(rotated_points)

        # Extract translation vector (so XYZ = R@xyz + T)
        self.translation_vector = self.rotation_matrix @ translations_rotated
        self.translation_vector += center

        return self


    def get_rotation_matrix(self, eigenvalues, eigenvectors, axis_order=(0, 1, 2)):
        """
        Constructs the rotation matrix based on the eigenvectors and the chosen axis_order.

        Parameters:
            eigenvalues (numpy.ndarray): Eigenvalues from the quadratic matrix.
            eigenvectors (numpy.ndarray): Eigenvectors from the quadratic matrix.
            axis_order (tuple): Specifies the axis mapping (e.g., (0, 1, 2) for X, Y, Z).

        Returns:
            numpy.ndarray: The rotation matrix.
        """
        # Sort eigenvalues and eigenvectors based on the specified axis order
        sorted_eigenvectors = eigenvectors[:, axis_order]

        # Construct the rotation matrix
        rotation_matrix = sorted_eigenvectors

        # Ensure the rotation matrix has determinant +1
        if np.linalg.det(rotation_matrix) < 0:
            rotation_matrix[:, 0] *= -1  # Flip the first axis

        return rotation_matrix
    
    def compute_fitting_error(self, rotated_points, a, b, translations_rotated):
        """
        Computes the fitting error between the rotated points and the fitted quadratic surface.

        Returns:
            float: The fitting error (residual).
        """
        x = rotated_points[:, 0]
        y = rotated_points[:, 1]
        z = rotated_points[:, 2]

        # Predicted z values
        z_pred = a * (x + translations_rotated[0])**2 + b * (y + translations_rotated[1])**2 + translations_rotated[2]

        # Compute the residual error
        error = np.linalg.norm(z - z_pred)

        return error

    def rotate_points(self, points, rotation_matrix=None):
        """
        Rotates the data points into the rotated coordinate system.

        Returns:
            numpy.ndarray: The rotated points.
        """
        if rotation_matrix is None:
            rotation_matrix = self.rotation_matrix

        rotated_points = (rotation_matrix.T @ points.T).T
        return rotated_points

    def fit_quadratic_in_rotated_frame(self, rotated_points):
        """
        Fits the quadratic function z + t_z = a*(X + t_x)^2 + b*(Y + t_y)^2 to the rotated data points,
        where t_x, t_y, t_z are the translations in the rotated frame.

        Returns:
            tuple: (a, b, t_x, t_y, t_z)
        """
        x = rotated_points[:, 0]
        y = rotated_points[:, 1]
        z = rotated_points[:, 2]

        # Build the design matrix including linear terms and constant term
        # (there is no skew/rotation term (xy) because we are in a rotated frame)
        D = np.column_stack([x**2, y**2, x, y, np.ones_like(x)])

        # Solve for the coefficients using least squares
        coeffs, _, _, _ = np.linalg.lstsq(D, z, rcond=None)
        a, b, d, e, f = coeffs

        # a x^2 + b y^2 + 2*c xy + d x + e y + f = z
        # but c is zero because we solved for rotation already!

        # Compute translations t_x and t_y
        t_x = -d / (2 * a) if a != 0 else 0
        t_y = -e / (2 * b) if b != 0 else 0

        # Compute t_z (translation along z)
        t_z = f - a * t_x**2 - b * t_y**2 

        return a, b, np.array((t_x, t_y, t_z))

    def mesh(self, ulim=(-1, 1), vlim=(-1, 1), resolution=100):
        """
        Generates a meshgrid of points on the quadric surface, applying the rotation and translation.

        Parameters:
            xlim (tuple): Limits for the x-axis in the rotated frame.
            ylim (tuple): Limits for the y-axis in the rotated frame.
            resolution (int): Number of points along each axis.
        """
        x_vals = np.linspace(ulim[0], ulim[1], resolution)
        y_vals = np.linspace(vlim[0], vlim[1], resolution)
        x_grid, y_grid = np.meshgrid(x_vals, y_vals)
        x_flat = x_grid.flatten()
        y_flat = y_grid.flatten()
        UV = np.column_stack((x_flat, y_flat))

        # Compute points on the surface
        XYZ = self.apply(UV)

        X_plot = XYZ[:, 0].reshape(x_grid.shape)
        Y_plot = XYZ[:, 1].reshape(y_grid.shape)
        Z_plot = XYZ[:, 2].reshape(x_grid.shape)

        return (X_plot, Y_plot, Z_plot)
    
    def plot(self, ulim=(-1, 1), vlim=(-1, 1), resolution=100, ax=None, **kwargs):
        """
        Plots the quadtatic surface mesh using matplotlib 3d. 

        Parameters:
            ulim (tuple): Limits for the x-axis in the rotated frame.
            vlim (tuple): Limits for the y-axis in the rotated frame.
            resolution (int): Number of points along each axis.
            ax (matplotlib.axes._subplots.Axes3DSubplot): Matplotlib 3d axes to plot on.
            kwargs (dict): Additional keyword arguments to pass to the ax.plot_surface method.
            
        """
        mesh = self.mesh(ulim, vlim, resolution)
        plot_kwargs = dict(cmap='viridis', alpha=0.6)
        plot_kwargs.update(kwargs)

        # Plot
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        ax.plot_surface(*mesh, **plot_kwargs)



    def plot_with_plotly(self, mesh, points, **kwargs):
        """
        Plots the quadratic surface mesh using Plotly 3D.

        Parameters:
            ulim (tuple): Limits for the x-axis in the rotated frame.
            vlim (tuple): Limits for the y-axis in the rotated frame.
            resolution (int): Number of points along each axis.
            kwargs (dict): Additional keyword arguments to pass to the go.Surface method.
            
        """
        import plotly.graph_objects as go

        X, Y, Z = mesh

        # Set plot parameters (kwargs passed to Plotly's Surface method)
        plot_kwargs = dict(colorscale='Viridis', opacity=0.6)
        plot_kwargs.update(kwargs)

        # Create a 3D surface plot using Plotly
        fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, **plot_kwargs)])

        # Add scatter plot for the original points
        scatter = go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers', marker=dict(size=1, color='red', opacity=0.8),
            name='Original Points'
        )
        fig.add_trace(scatter)

        # Update layout for better visualization
        fig.update_layout(scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'  # Keep the aspect ratio equal for all axes
        ))

        # Show the plot in the browser
        fig.show()

    
class QuadraticComparison:
    """
    Compares two quadratics (fitted and true) using the RotatedQuadratic class.

    It accounts for possible swaps between the a and b coefficients based on their magnitudes
    and compares both the rotation matrices and translation vectors.
    """

    def __init__(self, true_quadric, fitted_quadric):
        """
        Initializes the comparison.

        Parameters:
            true_quadric (RotatedQuadratic): The true quadratic surface.
            fitted_quadric (RotatedQuadratic): The fitted quadratic surface.
        """
        self.true_quadric = true_quadric
        self.fitted_quadric = fitted_quadric

        # Initialize comparison metrics
        self.coeff_error = None
        self.translation_error = None
        self.angle_error_degrees = None
        self.best_match = None

    def compare(self):
        """
        Compares the true and fitted quadratics by considering coefficient ordering,
        rotation alignment, and translation discrepancies.
        """

        # Compute coefficient differences
        # -> The order of a, b is undetermined
        # -> The sign of a, b is undetermined
        coeff_error = min([
            np.hypot( self.fitted_quadric.a - self.true_quadric.a,  self.fitted_quadric.b - self.true_quadric.b),
            np.hypot( self.fitted_quadric.b - self.true_quadric.a,  self.fitted_quadric.a - self.true_quadric.b),
            np.hypot(-self.fitted_quadric.a - self.true_quadric.a, -self.fitted_quadric.b - self.true_quadric.b),
            np.hypot(-self.fitted_quadric.b - self.true_quadric.a, -self.fitted_quadric.a - self.true_quadric.b)
        ])
        self.coeff_error = coeff_error

        # All possible axis permutations
        axis_permutations = list(permutations([0, 1, 2]))
        
        min_error = np.inf  # Initialize minimum error
        for perm in axis_permutations:

            # Reorder the axes of the true quadric's rotation matrix based on the permutation
            permuted_rotation = self.true_quadric.rotation_matrix[:, perm]

            # Flip axis sign if needed to ensure proper rotation (determinant +1)
            if np.linalg.det(permuted_rotation) < 0:
                permuted_rotation[:, 0] *= -1  # Flip the first axis
            
            # Compute the rotation difference
            rotation_diff = permuted_rotation @ self.fitted_quadric.rotation_matrix.T

           # Compute the trace of the rotation difference matrix
            trace_value = np.trace(rotation_diff)

            # Ensure the trace is within the valid range for arccos
            trace_value = np.clip(trace_value, -1.0, 3.0)

            # Compute the angle between the two rotation matrices
            rotation_angle = np.arccos((trace_value - 1) / 2)

            # Set the error to the angle in degrees
            rotation_error_degrees = np.degrees(rotation_angle)

            # Modulo 180
            rotation_error_degrees =  min(rotation_error_degrees, 180 - rotation_error_degrees)
            
            if rotation_error_degrees < min_error:
                # Store the best permutation
                min_error = rotation_error_degrees
                self.best_permutation = {
                    'rotation': permuted_rotation,
                    'angle_error_degrees': rotation_error_degrees,
                    'axis_order': perm
                }

        # Set final rotation and translation errors
        self.angle_error_degrees = self.best_permutation['angle_error_degrees']

        self.translation_error = np.linalg.norm(self.fitted_quadric.translation_vector - self.true_quadric.translation_vector)


    def print_results(self):
        """
        Prints the comparison results.
        """
        if self.coeff_error is None or self.angle_error_degrees is None or self.translation_error is None:
            self.compare()

        np.set_printoptions(precision=4, suppress=True)

        # Retrieve best match details
        true_a = self.true_quadric.a
        true_b = self.true_quadric.b
        R_true = self.true_quadric.rotation_matrix
        R_fitted = self.fitted_quadric.rotation_matrix

        # Report the comparison
        print(f"\nTrue coefficients (possibly adjusted): a = {true_a:.4f}, b = {true_b:.4f}")
        print(f"Fitted coefficients: a = {self.fitted_quadric.a:.4f}, b = {self.fitted_quadric.b:.4f}")

        print("\nTrue rotation matrix (possibly adjusted):")
        print(R_true)

        print("\nFitted rotation matrix:")
        print(R_fitted)

        print("\nTrue translation matrix (possibly adjusted):")
        print(self.true_quadric.translation_vector)

        print("\nFitted translation matrix:")
        print(self.fitted_quadric.translation_vector)

        # Display coefficient error
        print(f"\nCoefficient error (Euclidean norm): {self.coeff_error:.6f}")

        # Display translation error
        print(f"Translation error (Euclidean norm): {self.translation_error:.6f}")

        # Display rotation difference matrix
        print("\nRotation difference matrix (Adjusted True * Fitted^T):")
        rotation_diff = R_true @ R_fitted.T
        print(rotation_diff)

        # Display angle difference between rotations
        print(f"\nAngle difference between rotations: {self.angle_error_degrees:.4f} degrees")

def main():
    # Seed for reproducibility
    np.random.seed(42)

    # Define true parameters for the quadric
    a_true = -1
    b_true = 0.5
    translation_true = [0.5, -0.3, 1.2] # t_x, t_y, t_z
    rotation_angles = [10,20,30]  # Azimuth, Elevation, Roll in degrees

    # Create rotation matrix from Euler angles
    true_surface = RotatedQuadratic.from_euler_angles(a_true, b_true, *rotation_angles, translation_true)
   
    # Generate 3D points on the true surface
    uv = np.random.rand(1000, 2) * 2 - 1  # Random 2D points in [-1, 1]x[-1,1]
    print()
    print("UV points: (first 5)")
    print(uv[:5])

    points = true_surface.apply(uv)
    print()
    print("Generated 3D points: (first 5)")
    print(points[:5])

    # Add Gaussian noise to simulate measurement imperfections
    noise_level = 0.05
    points_noisy = points + noise_level * np.random.randn(*points.shape)

    # Initialize and fit the RotatedQuadratic model
    fitted_surface = RotatedQuadratic().fit(points_noisy)

    # # Initialize and perform comparison
    comparison = QuadraticComparison(true_quadric=true_surface, fitted_quadric=fitted_surface)
    comparison.compare()
    comparison.print_results()

    print()
    print("True Surface:\n",true_surface)
    print("Fit Surface:\n", fitted_surface)
    print()

    fitted_surface.plot_with_plotly(fitted_surface.mesh(), points_noisy)


if __name__ == "__main__":
    main()