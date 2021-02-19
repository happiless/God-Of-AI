#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# 1.获取URL数据（北京地铁数据，来自高德）：http://map.amap.com/service/subway?_1469083453978&srhdata=1100_drw_beijing.json
import requests
import json
import networkx as nx
import matplotlib as plot
import matplotlib.pyplot as plt
import numpy as np

r = requests.get('http://map.amap.com/service/subway?_1469083453978&srhdata=1100_drw_beijing.json')


def get_lines_stations_info(text):
    stations_json = json.loads(text)
    lines_info = {}
    stations_info = {}
    for line in stations_json['l']:
        line_name = line['ln']
        station_name_list = []
        for station in line['st']:
            # 站名
            station_name = station['n']
            station_name_list.append(station_name)
            # 坐标(x,y)
            position = station['sl']
            position = tuple(map(float, position.split(',')))
            # 数据加入站点信息dict
            stations_info[station_name] = position
        # 数据加入地铁线路dict
        lines_info[line_name] = station_name_list
    return lines_info, stations_info


def get_neighbor_info(lines_info: dict):
    def add_neighbor_dict(info, station1, station2):
        station_list = info.get(station1)
        if not station_list:
            station_list = []
        station_list.append(station2)
        info[station1] = station_list
        return info

    neighbor_info = {}
    for line_name, station_name_list in lines_info.items():
        for i in range(len(station_name_list) - 1):
            station1 = station_name_list[i]
            station2 = station_name_list[i + 1]
            neighbor_info = add_neighbor_dict(neighbor_info, station1, station2)
            neighbor_info = add_neighbor_dict(neighbor_info, station2, station1)
    return neighbor_info


lines_info, stations_info = get_lines_stations_info(r.text)
neighbor_info = get_neighbor_info(lines_info)

# 画出地铁图
plt.figure(figsize=(20, 20))
stations_graph = nx.Graph()
stations_graph.add_nodes_from(list(stations_info.keys()))
nx.draw(stations_graph, stations_info, with_labels=True, node_size=10)

# 画出站点连接图
plt.figure(figsize=(40, 40))
stations_connection_graph = nx.Graph(neighbor_info)
nx.draw(stations_connection_graph, stations_info, with_labels=True, node_size=10)


# 第一种算法：递归查找所有路径
def get_path_DFS_ALL(lines_info, neighbor_info, from_station, to_station):
    # 递归算法，本质上是深度优先
    # 遍历所有路径
    # 这种情况下，站点间的坐标距离难以转化为可靠的启发函数，所以只用简单的BFS算法
    if not neighbor_info.get(from_station):
        print(f'起始站点“{from_station}”不存在。请正确输入！')
        return None
    if not neighbor_info.get(to_station):
        print(f'目的站点“{to_station}”不存在。请正确输入！')
        return None
    path = []
    this_station = from_station
    path.append(this_station)
    neighbors = neighbor_info.get(this_station)
    node = {
        'pre_station': '',
        'this_station': this_station,
        'neighbors': neighbors,
        'path': path
    }
    return get_next_station_DFS_ALL(node, neighbor_info, to_station)


def get_next_station_DFS_ALL(node, neighbor_info, to_station):
    neighbors = node.get('neighbors')
    pre_station = node.get('this_station')
    path = node.get('path')
    paths = []
    for i in range(len(neighbors)):
        this_station = neighbors[i]
        if this_station in path:
            # 如果此站点已经在路径中，说明环路，此路不通
            return None
        if neighbors[i] == to_station:
            # 找到终点，返回路径
            path.append(to_station)
            paths.append(path)
            return paths
        else:
            _neighbors = neighbor_info.get(this_station).copy()
            _neighbors.remove(pre_station)
            _path = path.copy()
            _path.append(this_station)
            new_node = {
                'pre_station': pre_station,
                'this_station': this_station,
                'neighbors': _neighbors,
                'path': _path
            }
            _paths = get_next_station_DFS_ALL(new_node, neighbor_info, to_station)
            if _paths:
                paths.extend(_paths)
    return paths


#
# paths = get_path_DFS_ALL(lines_info, neighbor_info, '回龙观', '西二旗')
# print(f'共有{len(paths)}种路径')
# for item in paths:
#     print(f'此路径共计{len(item) - 1}站')
#     print('-'.join(item))


# 第二种算法：没有启发函数的简单宽度优先
def get_path_BFS(lines_info, neighbor_info, from_station, to_station):
    # 搜索策略：以站点数量为cost（因为车票价格是按站算的）
    # 这种情况下，站点间的坐标距离难以转化为可靠的启发函数，所以只用简单的BFS算法
    # 由于每深一层就是cost加1，所以每层的cost都相同，算和不算没区别，所以省略
    if not neighbor_info.get(from_station):
        print(f'起始站点“{from_station}”不存在。请正确输入！')
        return None
    if not neighbor_info.get(to_station):
        print(f'目的站点“{to_station}”不存在。请正确输入！')
        return None

    # 搜索节点是个dict，key=站名，value是包含路过的站点list
    nodes = {}
    nodes[from_station] = [from_station]
    while True:
        new_nodes = {}
        for k, v in nodes.items():
            neighbor = neighbor_info.get(k).copy()
            if len(v) >= 2:
                # 不往上一站走
                pre_station = v[-2]
                neighbor.remove(pre_station)
            for station in neighbor:
                # 遍历邻居
                if station in nodes:
                    continue
                path = v.copy()
                path.append(station)
                new_nodes[station] = path
                if station == to_station:
                    # 找到路径, 结束
                    return path
        nodes = new_nodes
    else:
        print('未能找到路径')
    return None


# paths = get_path_BFS(lines_info, neighbor_info, '北安河', '石门')
# print("路径总计%d站。" % (len(paths) - 1))
# print("-".join(paths))

import pandas as pd


# 第三种算法：以路径路程为cost的启发式搜索
def get_path_Astar(lines_info, neighbor_info, stations_info, from_station, to_station):
    # 搜索策略：以路径的站点间直线距离累加为cost，以当前站点到目标的直线距离为启发函数
    # 检查输入站点名称
    if not neighbor_info.get(from_station):
        print(f'起始站点“{from_station}”不存在。请正确输入！')
        return None
    if not neighbor_info.get(to_station):
        print(f'目的站点“{to_station}”不存在。请正确输入！')
        return None

    # 计算所有节点到目标节点的直线距离，备用
    distances = {}
    x, y = stations_info.get(to_station)
    for k, v in stations_info.items():
        x0, y0 = stations_info.get(k)
        l = ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5
        distances[k] = l

    # 已搜索过的节点，dict
    # key=站点名称，value是已知的起点到此站点的最小cost
    searched = {}
    searched[from_station] = 0

    # 数据结构为pandas的dataframe
    # index为站点名称
    # g为已走路径，h为启发函数值（当前到目标的直线距离）
    nodes = pd.DataFrame([[[from_station], 0, 0, distances.get(from_station)]],
                         index=[from_station], columns=['path', 'cost', 'g', 'h'])
    count = 0
    while True:
        if count > 1000:
            break
        nodes.sort_values('cost', inplace=True)
        for index, node in nodes.iterrows():
            count += 1
            # 向邻居中离目的地最短的那个站点搜索
            neighbors = neighbor_info.get(index).copy()
            if len(node['path']) >= 2:
                # 不向这个路径的反向去搜索
                neighbors.remove(node['path'][-2])
            for i in range(len(neighbors)):
                count += 1
                neighbor = neighbors[i]
                g = node['g'] + get_distance(stations_info, index, neighbor)
                h = distances[neighbor]
                cost = g + h
                path = node['path'].copy()
                path.append(neighbor)
                if neighbor == to_station:
                    # 找到目标, 结束
                    print(f'共检索{count}次。')
                    return path
                if neighbor in searched:
                    if g >= searched[neighbor]:
                        # 说明现在搜索的路径不是最优，忽略
                        continue
                    else:
                        searched[neighbor] = g
                        nodes.drop(neighbor, axis=0, inplace=True)
                        row = pd.DataFrame([[path, cost, g, h]], index=[neighbor], columns=['path', 'cost', 'g', 'h'])
                        nodes = nodes.append(row)
                else:
                    searched[neighbor] = g
                    row = pd.DataFrame([[path, cost, g, h]], index=[neighbor], columns=['path', 'cost', 'g', 'h'])
                    nodes = nodes.append(row)

            # 个站点的所有邻居都搜索完了，删除这个节点
            nodes.drop(index, axis=0, inplace=True)
        # 外层for循环只跑第一行数据，然后重新sort后再计算
        continue
    print('未能找到路径')
    return None


def get_distance(stations_info, station1, station2):
    x1, y1 = stations_info.get(station1)
    x2, y2 = stations_info.get(station2)
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


paths = get_path_Astar(lines_info, neighbor_info, stations_info, '回龙观', '西二旗')
if paths:
    print(f"路径总计{len(paths) - 1}站。")
    print("-".join(paths))