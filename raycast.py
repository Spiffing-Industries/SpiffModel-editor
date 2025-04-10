import numpy as np


def metaball_field(point, metaballs):
    """Computes the metaball field value at a given point."""
    field_value = 0.0
    for x,y,z, radius in metaballs:
        center = np.array([x,y,z])
        r = np.linalg.norm(point - center)
        if r < 1e-6:
            r = 1e-6  # prevent division by zero
        field_value += radius / (r**2)
    return field_value

def raymarch(origin, direction, metaballs, threshold=1.0, max_steps=100, max_distance=100.0, step_size=0.1):
    """
    Raymarching function to detect intersection with metaballs.

    Args:
        origin: 3D starting point of the ray.
        direction: Normalized 3D ray direction.
        metaballs: List of tuples (center, radius).
        threshold: Field threshold for surface definition.
        max_steps: Max steps to march along the ray.
        max_distance: Max distance before giving up.
        step_size: Distance to move each step.

    Returns:
        Intersection point if hit, None otherwise.
    """
    direction = direction / np.linalg.norm(direction)
    distance_traveled = 0.0

    for _ in range(max_steps):
        point = origin + direction * distance_traveled
        field = metaball_field(point, metaballs)

        if field >= threshold:
            return point  # Intersection found

        distance_traveled += step_size
        if distance_traveled > max_distance:
            break

    return None  # No intersection


if __name__ == "__main__":
    metaballs = [
        np.array([0.0, 0.0, 5.0,2.0]),
        np.array([1.5, 0.0, 5.0,2.0])
    ]

    origin = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])

    hit = raymarch(origin, direction, metaballs,threshold = 2.0)
    if hit is not None:
        print(f"Hit metaball surface at: {hit}")
    else:
        print("No hit")
