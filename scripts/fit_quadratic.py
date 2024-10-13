import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import pandas as pd

from dotenv import load_dotenv; load_dotenv()
from kappa.quadric_fitting import RotatedQuadratic


  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit a quadratic surface to 3D points.")
    parser.add_argument("input", help="Path to input CSV file with X, Y, Z points.")
    parser.add_argument("output", nargs='?', default='-', help="Path to output JSON file for results (default:  stdout).")
    parser.add_argument("--show-plot", action="store_true", help="Show interactive 3D plot of the points.")
    args = parser.parse_args()


    data = pd.read_csv(args.input)
    points = data[['X', 'Y', 'Z']].to_numpy()
    surface = RotatedQuadratic().fit(points)

    a= surface.a
    b= surface.b
    azimuth, elevation, roll = R.from_matrix(surface.rotation_matrix).as_euler('zyx', degrees=True)

    # Compute the principal curvatures and other statistics
    k1 = 2 * a
    k2 = 2 * b
    mean_curvature = 0.5 * (k1 + k2)
    gaussian_curvature = k1 * k2

    print(f"Fitted coefficients: a={a}, b={b}")
    print(f"Principal curvatures:")
    print(f"  k1={k1}")
    print(f"  k2={k2}")
    print(f"Mean curvature={mean_curvature}")
    print(f"Gaussian curvature={gaussian_curvature}")



    # Create a dictionary for JSON output
    result = {
        "input_file": args.input,
        "fitted_coefficients": [a, b],
        "surface_translation": surface.translation_vector.tolist(),
        "surface_rotation": {
            "azimuth": azimuth,
            "elevation": elevation,
            "roll": roll,
            "rotation_matrix": surface.rotation_matrix.tolist()
        },
        "principal_curvatures": {
            "k1": k1,
            "k2": k2,
            "mean_curvature": mean_curvature,
            "gaussian_curvature": gaussian_curvature
        }
    }

      # Save results as JSON
    if args.output == '-':
        print(json.dumps(result, indent=4))
    else:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=4)

    canonical_xyz = surface.untransformed(points)
    umin, vmin, _ = canonical_xyz.min(0)
    umax, vmax, _ = canonical_xyz.max(0)

    # Show 3D plot if flag is set
    if args.show_plot:
        mesh = surface.mesh(ulim=(umin,umax), vlim=(vmin, vmax))
        surface.plot_with_plotly(mesh, points)
        