#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# @File    : 03_homework_能源中心问题.py
# @todo    : k-means原理
import math
import re
import numpy as np
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx

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


def geo_parse(coordination):
    """
    解析字符串
    :param coordination:
    :return:
    """
    city_location = {}
    pattern = re.compile(r"{name:'(\w+)',\s+geoCoord:\[(\d+.\d+),\s+(\d+.\d+)\]}")
    for line in coordination.split('\n'):
        if not line: continue
        city_info = re.findall(pattern, line)
        city_name, lon, lat = city_info[0]
        city_location[city_name] = [float(lon), float(lat)]
    return city_location


def geo_distance(orgin, destination):
    """
    计算地理距离
    :param orgin:
    :param destination:
    :return:
    """
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


def draw_nx_graph(city_location):
    """
    画出地理位置图
    :param city_location:
    :return:
    """
    from pylab import mpl
    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    import matplotlib.pyplot as plt
    import networkx as nx
    city_graph = nx.Graph()
    city_graph.add_nodes_from(list(city_location.keys()))
    nx.draw(city_graph, city_location, with_labels=True, node_size=30)
    plt.show()


def get_random_center(all_x, all_y):
    return random.uniform(min(all_x), max(all_x)), random.uniform(min(all_y), max(all_y))


def iter_once(centers, closet_points, threshold=5):
    changed = False
    for c in closet_points:
        former_center = centers[c]
        neighbors = closet_points[c]
        neighbors_center = np.mean(neighbors, axis=0)
        if geo_distance(former_center, neighbors_center) > threshold:
            centers[c] = neighbors_center
            changed = True
        else:
            pass  # keep former center
    return centers, changed


def user_define_kmeans(X, k, threshold=5):
    all_x, all_y = X[:, 0], X[:, 1]
    centers = {f'{i + 1}': get_random_center(all_x, all_y) for i in range(k)}

    changed = True

    closet_point = defaultdict(list)
    while changed:

        for x, y in zip(all_x, all_y):
            closet_c, closet_dis = min([(k, geo_distance((x, y), centers[k])) for k in centers], key=lambda x: x[1])
            closet_point[closet_c].append([x, y])

        centers, changed = iter_once(centers, closet_point, threshold)
        print("iteration")

    return centers, closet_point


def draw_clusters(X, centers, closet_point):
    plt.scatter(X[:, 0], X[:, 1])
    plt.scatter(*zip(*centers.values()))
    for c, point in closet_point.items():
        plt.scatter(*zip(*point))
    plt.show()


def draw_cities(cities, color=None):
    city_graph = nx.Graph()
    city_graph.add_nodes_from(list(cities.keys()))
    nx.draw(city_graph, cities, node_color=color, with_labels=True, node_size=30)


def cluster_by_sklearn_kmeans(X, k):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    return kmeans.cluster_centers_


if __name__ == '__main__':
    city_location = geo_parse(coordination_source)

    draw_nx_graph(city_location)

    X = np.array(list(city_location.values()))
    K = 5
    threshold = 3

    centers, closet_point = user_define_kmeans(X, K, threshold)

    draw_clusters(X, centers, closet_point)

    centers_2 = cluster_by_sklearn_kmeans(X, K)

    city_location_with_station = {
        '能源站-{}'.format(int(i)): position for i, position in centers.items()
    }
    city_location_with_station2 = {
        'sk-能源站-{}'.format(int(i) + 1): [x, y] for i, (x, y) in enumerate(zip(centers_2[:, 0], centers_2[:, 1]))
    }
    draw_cities(city_location_with_station, color='green')
    draw_cities(city_location_with_station2, color='yellow')
    draw_cities(city_location, color='red')

    plt.show()
