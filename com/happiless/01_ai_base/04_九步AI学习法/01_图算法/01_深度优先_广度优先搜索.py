#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# 解析数据
from collections import defaultdict

coordination_source = """
{name:'兰州', geoCoord:[103.73, 36.03]},
{name:'嘉峪关', geoCoord:[98.17, 39.47]},
{name:'西宁', geoCoord:[101.74, 36.56]},
{name:'成都', geoCoord:[104.06, 30.67]},
{name:'石家庄', geoCoord:[114.48, 38.03]},
{name:'拉萨', geoCoord:[102.73, 25.04]},
{name:'贵阳', geoCoord:[106.71, 26.57]},
{name:'武汉', geoCoord:[114.31, 30.52]},
{name:'郑州', geoCoord:[113.65, 34.76]},
{name:'济南', geoCoord:[117, 36.65]},
{name:'南京', geoCoord:[118.78, 32.04]},
{name:'合肥', geoCoord:[117.27, 31.86]},
{name:'杭州', geoCoord:[120.19, 30.26]},
{name:'南昌', geoCoord:[115.89, 28.68]},
{name:'福州', geoCoord:[119.3, 26.08]},
{name:'广州', geoCoord:[113.23, 23.16]},
{name:'长沙', geoCoord:[113, 28.21]},
//{name:'海口', geoCoord:[110.35, 20.02]},
{name:'沈阳', geoCoord:[123.38, 41.8]},
{name:'长春', geoCoord:[125.35, 43.88]},
{name:'哈尔滨', geoCoord:[126.63, 45.75]},
{name:'太原', geoCoord:[112.53, 37.87]},
{name:'西安', geoCoord:[108.95, 34.27]},
//{name:'台湾', geoCoord:[121.30, 25.03]},
{name:'北京', geoCoord:[116.46, 39.92]},
{name:'上海', geoCoord:[121.48, 31.22]},
{name:'重庆', geoCoord:[106.54, 29.59]},
{name:'天津', geoCoord:[117.2, 39.13]},
{name:'呼和浩特', geoCoord:[111.65, 40.82]},
{name:'南宁', geoCoord:[108.33, 22.84]},
//{name:'西藏', geoCoord:[91.11, 29.97]},
{name:'银川', geoCoord:[106.27, 38.47]},
{name:'乌鲁木齐', geoCoord:[87.68, 43.77]},
{name:'香港', geoCoord:[114.17, 22.28]},
{name:'澳门', geoCoord:[113.54, 22.19]}
"""

import re


def get_city_info(city_coordination: str):
    city_location = {}
    for line in city_coordination.split("\n"):
        if line.startswith("//"): continue
        if line.strip() == "": continue
        city = re.findall("name:'(\w+)'", line)[0]
        x_y = re.findall("Coord:\[(\d+.\d+, \d+.\d+)\]", line)[0]
        x_y = tuple(map(float, x_y.split(",")))
        city_location[city] = x_y

    return city_location


# Compute distance between cities
def geo_distance(origin, destination):
    """

    :param origin:
    :param destination:
    :return: distance_in_km : float
    Examples
    --------
    origin = (48.1372, 11.5756)  # Munich
    destination = (52.5186, 13.4083)  # Berlin
    round(distance(origin, destination), 1)
    504.2
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371
    import math
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)

    # a = (math.sin(d_lat / 2) * math.sin(d_lat / 2) +
    #      math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
    #      * math.sin(d_lon / 2) * math.sin(d_lon / 2))

    a = (math.sin(d_lat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(d_lon / 2) ** 2)

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c
    return d


city_info = get_city_info(coordination_source)


def get_city_distance(city1, city2):
    return geo_distance(city_info[city1], city_info[city2])


import networkx as nx

social_network = {
    '小张': ['小刘', '小王', '小红'],
    '小王': ['六六', '娇娇', '小曲'],
    '娇娇': ['宝宝', '花花', '喵喵'],
    '六六': ['小罗', '奥巴马']
}


def search_graph(graph, start, expand_position):
    need_to_check = [start]
    expanded = set()
    while need_to_check:
        person = need_to_check.pop(expand_position)
        if person in expanded: continue
        new_expanded = graph.get(person, [])
        need_to_check += new_expanded
        expanded.add(person)
    return expanded


def dfs(graph, start): return search_graph(graph, start, -1)


def bfs(graph, start): return search_graph(graph, start, 0)


def build_connection(city_info: dict, threshold=700):
    cities_connection = defaultdict(list)
    cities = list(city_info.keys())
    for c1 in cities:
        for c2 in cities:
            if c1 == c2: continue
            if get_city_distance(c1, c2) < threshold:
                cities_connection[c1].append(c2)
    return cities_connection


cities_connection = build_connection(city_info)

print(dfs(social_network, "小张"))
print(bfs(social_network, "小张"))
print(cities_connection)

social_graph = nx.Graph(social_network)
nx.draw(social_graph, with_labels=True)

cities_connection_graph = nx.Graph(cities_connection)
nx.draw(cities_connection_graph, city_info, with_labels=True, node_size=10)
