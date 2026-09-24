import math

def distance_field(width, height, obstacles, resolution):
    """Exact Euclidean distance (meters) from each cell to the nearest obstacle.

    Computed with the Felzenszwalb-Huttenlocher squared-distance transform:
    one exact 1-D pass per column, then one per row. Every cell gets the true
    geometric distance to the closest obstacle cell, so an inflation radius
    means the same clearance along diagonals as along the grid axes.
    """
    inf = float('inf')
    # Stands in for infinity inside the transform so the parabola arithmetic
    # stays finite; larger than any real squared grid distance.
    unreachable = float(width * width + height * height + 1)
    blocked = [[False] * width for _ in range(height)]
    for x, y in obstacles:
        if 0 <= x < width and 0 <= y < height:
            blocked[y][x] = True
    squared = [[0.0] * width for _ in range(height)]
    for x in range(width):
        column = _dt1d([0.0 if blocked[y][x] else unreachable for y in range(height)])
        for y in range(height):
            squared[y][x] = column[y]
    for y in range(height):
        squared[y] = _dt1d(squared[y])
    return [[resolution * math.sqrt(v) if v < unreachable else inf for v in row] for row in squared]

def _dt1d(f):
    """Exact 1-D squared Euclidean distance transform (Felzenszwalb-Huttenlocher)."""
    n = len(f)
    if n == 0:
        return []
    inf = float('inf')
    d = [0.0] * n
    v = [0] * n          # sites of the parabolas forming the lower envelope
    z = [0.0] * (n + 1)  # positions where consecutive envelope parabolas intersect
    k = 0
    z[0] = -inf
    z[1] = inf
    for q in range(1, n):
        s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k])
        while s <= z[k]:
            k -= 1
            s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k])
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = inf
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        d[q] = (q - v[k]) ** 2 + f[v[k]]
    return d

def inflate(field, radius):
    return [[max(0.0, 1 - v / radius) if radius > 0 else 0.0 for v in row] for row in field]
