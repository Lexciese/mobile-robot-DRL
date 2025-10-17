import numpy as np

def generate_m_waypoints(points_per_segment=10):
    """
    Generates a list of waypoints that form the shape of the letter 'M'.

    The 'M' starts at (0, 0), goes up, diagonally down, diagonally up,
    and finally straight down.

    Args:
        points_per_segment (int): The number of points to generate for each
                                  of the four line segments of the 'M'.

    Returns:
        list: A list of waypoints, where each waypoint is in the format
              [[x], [y], [0]].
    """
    # Define the key vertices (corners) of the 'M' shape.
    # We start at the bottom-left.
    p1 = np.array([0, 0])   # Start point
    p2 = np.array([0, 8])   # Top-left
    p3 = np.array([4, 4])   # Middle-bottom dip
    p4 = np.array([8, 8])   # Top-right
    p5 = np.array([8, 0])   # End point, bottom-right

    all_points = []

    # Generate points for the first segment (upward line)
    # We use endpoint=False to avoid duplicating the corner points between segments.
    all_points.extend(np.linspace(p1, p2, points_per_segment, endpoint=False))

    # Generate points for the second segment (diagonal down)
    all_points.extend(np.linspace(p2, p3, points_per_segment, endpoint=False))

    # Generate points for the third segment (diagonal up)
    all_points.extend(np.linspace(p3, p4, points_per_segment, endpoint=False))

    # Generate points for the final segment (downward line)
    # We use endpoint=True here to include the very last point of the shape.
    all_points.extend(np.linspace(p4, p5, points_per_segment, endpoint=True))

    # Format the generated (x, y) points into the desired [[[x], [y], [0]]] structure
    waypoints = []
    for point in all_points:
        formatted_point = [[point[0]], [point[1]], [0]]
        waypoints.append(formatted_point)

    return waypoints

if __name__ == "__main__":
    # Generate the waypoints. You can change points_per_segment for more or less detail.
    # Using 5 points per segment will result in 4 * 5 = 20 total waypoints.
    m_shape_waypoints = generate_m_waypoints(points_per_segment=5)

    # --- Print the array in a readable format ---
    print("[")
    for i, wp in enumerate(m_shape_waypoints):
        # Format the numbers to one decimal place for clean output
        x_str = f"[{wp[0][0]:.1f}]"
        y_str = f"[{wp[1][0]:.1f}]"
        z_str = f"[{wp[2][0]}]"

        # Print the formatted waypoint line
        print(f"  [{x_str}, {y_str}, {z_str}]", end="")

        # Add a comma if it's not the last element
        if i < len(m_shape_waypoints) - 1:
            print(",")
        else:
            print() # Newline for the last element
    print("]")
