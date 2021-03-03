import math

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


def geo_parse(coordination_source):
    city_location = {}
    import re
    pattern = re.compile(r"{name:'(\w+)',\s+geoCoord:\[(\d+.\d+),\s+(\d+.\d+)\]}", )
    for line in coordination_source.split('\n'):
        city_info = re.findall(pattern, line)
        if not city_info: continue
        city, long, lat = city_info[0]
        city_location[city] = (float(long), float(lat))
    return city_location


city_location = geo_parse(coordination_source)


def geo_distance(orgin, destination):

    long1, lat1 = orgin
    long2, lat2 = destination
    radius = 6371

    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(long2 - long1)
    a = math.sin(d_lat / 2) ** 2 + \
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d


# import matplotlib.pyplot as plt
# import networkx as nx
# city_graph = nx.Graph()
# city_graph.add_nodes_from(list(city_location.keys()))
# nx.draw(city_graph, city_location, with_labels=True, node_size=30)


# K-means: Initial k random centers
k = 10
import random

all_x, all_y = [], []

for _, location in city_location.items():
    x, y = location

    all_x.append(x)
    all_y.append(y)


def get_random_center(all_x, all_y):
    r_x = random.uniform(min(all_x), max(all_x))
    r_y = random.uniform(min(all_y), max(all_y))

    return r_x, r_y


print(get_random_center(all_x, all_y))

K = 5
centers = {f'{i+1}': get_random_center(all_x, all_y) for i in range(K)}
print(centers)

from collections import defaultdict

closet_point = defaultdict(list)
for x, y in zip(all_x, all_y):
    closet_c, closet_dis = min([])