import numpy as np
import pytest
from pytest import approx
from kappa.quadric_fitting import RotatedQuadratic, QuadraticComparison

@pytest.mark.parametrize(
        "a, b, azimuth, elevation, roll", [
        (0.5, -0.5,  0, 0, 0),    # No rotation
        (1.0, 0.5,  30, 45, 60),  # Rotated with significant angles
        (-1.0, 0.5, 10, 20, 30), # Negative a, small rotations
        (0.7, -0.8,  0, 90, 0),   # Only elevation rotation
        (1.2, -1.0,  90, 0, 90),  # Mixed rotations
        (0.4, -0.09, 136, 61, -98) # The Gorrilla
    ]
)
def test_fit_generated_quadratic(a, b, azimuth, elevation, roll):
    """
    Test fitting of a quadratic surface to generated points, accounting for possible coefficient permutations.
    """
    num_points = 100
    noise_level = 0.0  # No noise for simplicity

    # Generate points with deterministic noise (no noise here, but seed ensures repeatability)
    np.random.seed(42)

    true_quadric = RotatedQuadratic.from_euler_angles(a=a, b=b, azimuth=azimuth, elevation=elevation, roll=roll)
    uv = np.random.rand(1000, 2) * 2 - 1  # Random 2D points in [-1, 1]x[-1,1]
    points = true_quadric.apply(uv)

    # Fit the quadric to the generated points
    fitted_quadric = RotatedQuadratic().fit(points)

    # Compare true coefficients and rotation with the fitted ones
    comparison = QuadraticComparison(true_quadric, fitted_quadric)
    comparison.compare()
    
    # Optionally print the comparison results
    comparison.print_results()

    # Assert that coefficient error is below an acceptable threshold
    assert comparison.coeff_error < 0.1, "Coefficient error is larger than expected."
    
    # Assert that rotation error is below an acceptable threshold
    assert comparison.angle_error_degrees < 1.0, "Rotation angle error is larger than expected."
    
    # Assert that translation error is below an acceptable threshold
    assert comparison.translation_error < 0.1, "Translation error is larger than expected."


    # TODO: It would be nice to test the quality of fit e.g. by checking the mean
    #       residual, but this is difficult to do correctly. 

