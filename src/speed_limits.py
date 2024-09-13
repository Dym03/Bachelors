import osmnx as ox
import networkx as nx
import os
import pandas as pd
import overpy
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from geopy import distance

from coords import Map_Point, DMS_Coord

import image_processing as ip

MAX_RADIUS = 5  # Range in meters
QUERIE_DISTANCE = 150  # Range in meters


def get_nearest_road(point_of_interest: Map_Point):
    point = (point_of_interest.lat, point_of_interest.lon)
    G = ox.graph_from_point(point, dist=200, network_type="drive", simplify=True)
    # for u, v, data in G.edges(keys=False, data=True):
    #     print(data.get("maxspeed")
    u, v, k = ox.distance.nearest_edges(
        G, X=point_of_interest.lat, Y=point_of_interest.lon
    )
    print(f"U = {u}, V = {v}, K = {k}")
    edge_data = G.get_edge_data(u, v, k)
    # print(edge_data)
    # if "name" in edge_data:
    #     print(edge_data["name"])
    # if "maxspeed" in edge_data:
    #     print(edge_data["maxspeed"])
    speed_limit = edge_data.get("maxspeed", "No speed limit information available")
    road_name = edge_data.get("name", "No known name")
    print(road_name, speed_limit)
    # return (road_name, speed_limit)


def calculate_distance(point1, point2):
    return distance.distance(point1, point2).meters


def query_speed_limits(point: Map_Point):
    api = overpy.Overpass()

    query = f"""
    [out:json];
    (
        way(around:{MAX_RADIUS}, {point.lat}, {point.lon})["highway"];
    );
    out body;
    >;
    out skel qt;
    """
    result = api.query(query)
    name_speed_limit = ("Unknown", 0)
    closest_road = None
    min_distance = float("inf")
    for way in result.ways:
        for node in way.nodes:
            node_location = (node.lat, node.lon)
            distance = calculate_distance(point, node_location)
            if distance < min_distance:
                min_distance = distance
                closest_road = way

    if closest_road is not None:
        name_speed_limit = (
            closest_road.tags.get("name", "No known name"),
            closest_road.tags.get("maxspeed", "No known maxspeed"),
        )
        print(name_speed_limit)


if __name__ == "__main__":
    data_folder_path = "data/video"
    video_name = "2018_1106_063528_019F.MP4"
    video_path = f"{data_folder_path}/{video_name}"
    template_digits = ip.load_templates()
    template_marks = ip.load_mark_templates()

    curr_lat = None
    curr_long = None
    last_queried = None
    # path = "2018_1106_063528_019F.csv"
    # if not os.path.exists(path):
    #     print(f"File {path} does not exist")
    #     exit()

    # df = pd.read_csv(path)

    # for index, row in df.iterrows():
    #     point = Map_Point(row["Lat"], row["Lon"])
    #     get_nearest_road(point)

    # Test with video
    video = cv.VideoCapture(video_path)
    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Just the bottom strip of the image to get the coords is needed, i make this to make the process faster
        gray = gray[ip.START_OF_Y_COORDINATE : gray.shape[0], 0 : gray.shape[1]]
        lat, long = ip.get_coords_images_from_image(gray, template_marks)
        if lat is not None and long is not None:
            act_lat = ip.split_numbers_with_marks(lat, template_marks, template_digits)
            act_long = ip.split_numbers_with_marks(
                long, template_marks, template_digits
            )
            if act_lat is not None and act_long is not None:
                # if is_close_to_crossroad(
                #     Map_Point(act_lat.to_decimal(), act_long.to_decimal()),
                #     crossroads_coords,
                #     DISTANCE_THRESHOLD,
                # ):
                #     print("Close to crossroad")
                #     video_writer.write(frame)
                if (
                    curr_lat is None
                    or curr_long is None
                    or act_lat != curr_lat
                    or act_long != curr_long
                ):
                    curr_lat = act_lat
                    curr_long = act_long
                    curr_coords = Map_Point(
                        curr_lat.to_decimal(), curr_long.to_decimal()
                    )

                    # print(f"Lat: {curr_lat}, Long: {curr_long}, {curr_coords}")
                    print("Nearest road using osmnx")
                    get_nearest_road(curr_coords)
                    print("\n\n")
                    if last_queried is None:
                        last_queried = curr_coords
                    elif (
                        calculate_distance(last_queried, curr_coords) > QUERIE_DISTANCE
                    ):
                        last_queried = curr_coords
                        print("Nearest ways using Overpass api")
                        query_speed_limits(curr_coords)
        cv.imshow("frame", frame)

        if cv.waitKey(1) == ord("q") or cv.waitKey(1) == ord("d"):
            break

    video.release()
    cv.destroyAllWindows()
