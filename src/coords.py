import math


class DD_Coord:
    def __init__(self, dec: float):
        self.dec = dec

    def __str__(self):
        return f"{self.dec}"


class DMS_Coord:
    def __init__(self, deg, min, sec):
        self.deg = deg
        self.min = min
        self.sec = sec

    def __str__(self):
        return f"{self.deg}° {self.min}' {self.sec}\""

    def __eq__(self, other):
        return self.deg == other.deg and self.min == other.min and self.sec == other.sec

    def to_decimal(self):
        return float(self.deg) + float(self.min) / 60 + float(self.sec) / 3600


class Map_Point:
    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def __str__(self):
        return f"Lat: {self.lat}, Lon: {self.lon}"

    def __eq__(self, other):
        return self.lat == other.lat and self.lon == other.lon

    def __iter__(self):
        yield self.lat
        yield self.lon


# Calculate distance between two points on the Earth in km
# https://community.fabric.microsoft.com/t5/Desktop/How-to-calculate-lat-long-distance/td-p/1488227
def calculate_distance(point_1: Map_Point, point_2: Map_Point):
    R = 6371  # Radius of the Earth in km
    dlon = math.radians(point_2.lon) - math.radians(point_1.lon)
    distance = (
        math.acos(
            math.sin(math.radians(point_1.lat)) * math.sin(math.radians(point_2.lat))
            + math.cos(math.radians(point_1.lat))
            * math.cos(math.radians(point_2.lat))
            * math.cos(dlon)
        )
        * R
    )

    return distance
