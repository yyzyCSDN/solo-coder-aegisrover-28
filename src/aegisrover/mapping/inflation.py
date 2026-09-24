import math

def _edt_1d(f):
    # Exact squared Euclidean distance transform of a 1-D sampled function,
    # after Felzenszwalb & Huttenlocher: d[q] = min_p (q - p)^2 + f[p].
    n = len(f)
    d = [0.0] * n
    if n == 0:
        return d
    v = [0] * n           # grid positions of the parabolas on the lower envelope
    z = [0.0] * (n + 1)   # interval boundaries where each parabola is lowest
    k = 0
    z[0] = -math.inf
    z[1] = math.inf
    for q in range(1, n):
        s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k])
        while s <= z[k]:
            k -= 1
            s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k])
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = math.inf
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        d[q] = (q - v[k]) ** 2 + f[v[k]]
    return d

def distance_field(width, height, obstacles, resolution):
    inf = math.inf
    # Finite stand-in for infinity so the envelope intersections stay finite;
    # larger than any real squared grid distance, so it never wins the minimum.
    big = float(4 * (width * width + height * height) + 1)
    f = [[big] * width for _ in range(height)]
    has_source = False
    for x, y in obstacles:
        if 0 <= x < width and 0 <= y < height:
            f[y][x] = 0.0
            has_source = True
    if not has_source:
        return [[inf] * width for _ in range(height)]
    for x in range(width):
        column = _edt_1d([f[y][x] for y in range(height)])
        for y in range(height):
            f[y][x] = column[y]
    field = []
    for y in range(height):
        row = _edt_1d(f[y])
        field.append([math.sqrt(v) * resolution for v in row])
    return field

def inflate(field, radius):
    return [[max(0.0, 1 - v / radius) if radius > 0 else 0.0 for v in row] for row in field]
