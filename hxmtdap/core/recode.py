import os
import pickle
import re
from abc import ABC, abstractmethod
from functools import reduce
from operator import mul

import hvplot.networkx as hvnx
import networkx as nx

from .logger import capture_exception_fromM, setup_logger
from ..tools.utils import get_expID


class BaseRecoder(ABC):
    def __init__(self, logger=None, exppath=None, gfilename="Graph.pkl"):
        self.logger = logger if logger is not None else setup_logger(__name__)
        self.exppath = exppath if exppath is not None else os.getcwd()

        self.expID = get_expID(self.exppath)
        self.gfilename = gfilename

        # 尝试载入图
        try:
            G = self.load_graph(ignore_missing=True)
        except:
            self.logger.warning(f"{gfilename} is invalid, created a new one.")
            G = None

        # 创建一个图
        if not G:
            self.FileGraph = nx.DiGraph()
            self.save_graph()
            self.logger.debug(f"File Graph not found, created a new one.")
        else:
            self.logger.debug(f'File Graph loaded from "{self.gfilename}"')
        self.logger.debug(f"Recoder Initialized, graph loaded from {self.gfilename}.")

    def __getattr__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            if name in self.FileGraph.nodes:  # 这里返回的节点file属性
                file_path = self.FileGraph.nodes[name]["file"]
                return os.path.join(self.exppath, file_path)
            else:
                raise AttributeError(
                    f"{type(self).__name__} object has no attribute '{name}'"
                )

    def get_graph(self):
        """
        外部访问图文件节点的接口
        """
        return self.FileGraph.nodes

    def get_attr(self, node, attr):
        """
        获取节点属性，主要用于谱背景生成时获取谱绑定的gti (parameters模块)
        """
        return self.FileGraph.nodes()[node][attr]

    def set_status(self):
        """
        仅由Status类调用的方法，只有创建STATUS节点后才能创建这个引用；
        Recoder.status用于传递给Parameters类获取当前活动文件的节点
        """
        self.status = self.FileGraph.nodes["STATUS"]

    @capture_exception_fromM
    def find_files(self, directory, pattern, is_absolute=False):
        """
        根据正则表达式查找文件，返回匹配到的文件列表
        """
        compiled_pattern = re.compile(pattern)
        merge_path = lambda a, b: (
            os.path.join(a, b)
            if is_absolute
            else os.path.relpath(os.path.join(a, b), self.exppath)
        )
        return [
            merge_path(directory, f)
            for f in os.listdir(directory)
            if compiled_pattern.match(f)
        ]

    @capture_exception_fromM
    def find_node_by_filepath(self, filepath, ignore_missing=False):
        """
        通过文件绝对路径查找对应的文件节点；
        如果找不到，则抛出FileNotFoundError
        """
        for node, data in self.FileGraph.nodes(data=True):
            f = data.get("file") or data.get("outfile")
            if f and f in filepath:
                return node
        if ignore_missing:
            return None
        else:
            raise FileNotFoundError(f"Do not find node by filepath: {filepath}")

    @capture_exception_fromM
    def add_files(self, desc, **kwargs):
        """
        新增一个文件节点，如果文件已存在，则更新属性；
        如果文件不存在，但节点名重复，则自动添加后缀（1, 2, 3, ...）；
        如果文件不存在，且节点名不重复，则直接添加该节点。
        返回文件节点的节点名。

        传入的参数中必须包含file，其值为文件的绝对路径。
        """
        # 所有节点属性都是字符串，以方便show
        attrs = {key: str(value) for key, value in kwargs.items()}

        exits_node = self.find_node_by_filepath(attrs["file"], ignore_missing=True)
        if not exits_node:  # 新文件节点
            if self.FileGraph.has_node(desc):  # 如果节点名重复
                index = 2
                new_desc = f"{desc}_{index}"
                while self.FileGraph.has_node(new_desc):
                    index += 1
                    new_desc = f"{desc}_{index}"
                self.FileGraph.add_node(new_desc, **attrs)
                return new_desc
            else:
                # 添加新节点
                self.FileGraph.add_node(desc, **attrs)
                return desc
        else:  # 节点属性被覆盖
            self.logger.info(f"FileNode {exits_node} exists. Overwriting it.")
            self.update_attr(exits_node, **attrs)
            return exits_node

    @capture_exception_fromM
    def update_attr(self, node, **kwargs):
        """
        为指定节点添加属性
        """
        self.FileGraph.nodes[node].update(**kwargs)

    @capture_exception_fromM
    def add_parents(self, node, parents_path_lst, ignore_missing=False):
        """
        为文件节点之间添加边，表示数据血缘关系；
        这里parents_path_lst是由上游数据文件路径组成的列表，应由commander类方法返回
        """
        for p_path in parents_path_lst:
            if p_path == "NONE":
                continue
            p_node = self.find_node_by_filepath(p_path, ignore_missing)
            if p_node:
                self.FileGraph.add_edge(p_node, node)
                self.logger.debug(f"add edge: {p_node} -> {node}")

    @capture_exception_fromM
    def save_graph(self):
        """
        保存图
        """
        os.makedirs(f"{self.exppath}/Graph", exist_ok=True)
        graph_file_path = f"{self.exppath}/Graph/{self.gfilename}"
        with open(graph_file_path, "wb") as f:
            pickle.dump(self.FileGraph, f)
        self.logger.debug(f"Graph saved to {self.gfilename}.")

    @capture_exception_fromM
    def load_graph(self, ignore_missing=False):
        """
        加载图
        """
        graph_file_path = f"{self.exppath}/Graph/{self.gfilename}"
        if os.path.exists(graph_file_path):
            with open(graph_file_path, "rb") as f:
                self.FileGraph = pickle.load(f)
            self.logger.debug(f"Graph loaded from {self.gfilename}.")
            return self.FileGraph
        else:
            if not ignore_missing:  # 默认不忽视异常
                raise FileNotFoundError(f"Graph {self.gfilename} not found.")
            else:
                self.logger.debug(f"Graph {self.gfilename} not found and missing it.")
                return None

    @capture_exception_fromM
    def show(self):
        """
        绘图
        """
        G = self.FileGraph

        # 创建字典存储每个层级的节点
        layer_nodes = {
            layer: [
                node
                for node, attrs in G.nodes(data=True)
                if attrs.get("layer") == layer
            ]
            for layer in ["1L", "2_calib", "3_screen", "4_ext", "5_prod"]
        }

        # 使用多部分布局
        pos = nx.multipartite_layout(G, subset_key="layer", align="vertical")

        # 节点属性配置
        nodes_attr = {
            "labels": "index",
            "font_color": "black",
            "node_size": 700,
            "linewidths": 0,
        }
        nodes_color = ["#9d45c08f", "#2760b48f", "#33b4278f", "#9c71158f", "#ca2b168f"]

        # 绘制每层的节点并存储结果
        nodes_draw = [
            hvnx.draw_networkx_nodes(
                G.subgraph(nodes), pos, **nodes_attr, node_color=color
            )
            for nodes, color in zip(layer_nodes.values(), nodes_color)
            if nodes
        ]

        # 绘制箭头
        arrows = hvnx.draw_networkx_edges(
            G, pos, arrowhead_length=0.0, edge_color="#04080359"
        )
        nodes_draw.append(arrows)

        # 返回分层图中最多的节点数量
        longest_list_len = lambda d: max((len(v) for v in d.values()), default=0)
        overlay = reduce(mul, nodes_draw)
        overlay = overlay.opts(
            width=300 * len(list(layer_nodes.keys())),
            height=200 + longest_list_len(layer_nodes) * 50,
            xaxis=None,
            yaxis=None,
        )
        return overlay

    @abstractmethod
    def get_1L(self):
        pass


class LERecoder(BaseRecoder):
    def __init__(self, logger=None, exppath=None, gfilename="LEGraph.pkl"):
        super().__init__(logger, exppath, gfilename=gfilename)

        self.FileGraph.add_nodes_from(
            ["LE_Evt", "LE_Orbit", "LE_Att", "LE_TH", "LE_EHK", "LE_InsStat"],
            layer="1L",
        )

    @capture_exception_fromM
    def get_1L(self):
        files_patterns = [
            ("LE", r".*LE-Evt_FFFFFF_V[1-9].*", "LE_Evt"),
            ("ACS", r".*Orbit_FFFFFF_V[1-9].*", "LE_Orbit"),
            ("ACS", r".*Att_FFFFFF_V[1-9].*", "LE_Att"),
            ("LE", r".*TH_FFFFFF_V[1-9].*", "LE_TH"),
            ("LE", r".*InsStat_FFFFFF_V[1-9].*", "LE_InsStat"),
            ("AUX", r".*EHK_FFFFFF_V[1-9].*", "LE_EHK"),
        ]

        for subdir, pattern, desc in files_patterns:
            full_dir = os.path.join(self.exppath, subdir)
            matched_files = sorted(self.find_files(full_dir, pattern, is_absolute=True))
            if matched_files:
                self.FileGraph.nodes[desc]["file"] = matched_files[-1]
            else:
                raise FileNotFoundError(f"{desc} not found!")

        self.logger.info("The LE detector provides complete 1L-level data.")
        return

    def get_pi(self):
        files_patterns = [
            ("output", r".*LE_pi.*", "LE_pi"),
            ("output", r".*LE_recon.*", "LE_recon"),
        ]

        for subdir, pattern, desc in files_patterns:
            full_dir = os.path.join(self.exppath, subdir)
            matched_files = sorted(self.find_files(full_dir, pattern))
            if matched_files:
                # self.FileGraph.add_node(desc, layer='calib')
                self.add_files(desc, file=matched_files[-1], layer="calib")
            else:
                raise FileNotFoundError(f"{desc} not found!")


class MERecoder(BaseRecoder):
    def __init__(self, logger=None, exppath=None, gfilename="MEGraph.pkl"):
        super().__init__(logger, exppath, gfilename=gfilename)

        self.FileGraph.add_nodes_from(
            ["ME_Evt", "ME_Orbit", "ME_Att", "ME_TH", "ME_EHK", "ME_InsStat"],
            layer="1L",
        )

    def get_1L(self):
        files_patterns = [
            ("ME", r".*ME-Evt_FFFFFF_V[1-9].*", "ME_Evt"),
            ("ACS", r".*Orbit_FFFFFF_V[1-9].*", "ME_Orbit"),
            ("ACS", r".*Att_FFFFFF_V[1-9].*", "ME_Att"),
            ("ME", r".*TH_FFFFFF_V[1-9].*", "ME_TH"),
            ("ME", r".*InsStat_FFFFFF_V[1-9].*", "ME_InsStat"),
            ("AUX", r".*EHK_FFFFFF_V[1-9].*", "ME_EHK"),
        ]

        for subdir, pattern, desc in files_patterns:
            full_dir = os.path.join(self.exppath, subdir)
            matched_files = sorted(self.find_files(full_dir, pattern))
            if matched_files:
                self.FileGraph.nodes[desc]["file"] = matched_files[-1]
            else:
                raise FileNotFoundError(f"{desc} not found!")


class HERecoder(BaseRecoder):
    def __init__(self, logger=None, exppath=None, gfilename="HEGraph.pkl"):
        super().__init__(logger, exppath, gfilename=gfilename)

        self.FileGraph.add_nodes_from(
            [
                "HE_Evt",
                "HE_Orbit",
                "HE_Att",
                "HE_HV",
                "HE_PM",
                "HE_DTime",
                "HE_TH",
                "HE_EHK",
            ],
            expID=self.expID,
            layer="1L",
        )

    def get_1L(self):
        files_patterns = [
            ("HE", r".*HE-Evt_FFFFFF_V[1-9].*", "HE_Evt"),
            ("ACS", r".*Orbit_FFFFFF_V[1-9].*", "HE_Orbit"),
            ("ACS", r".*Att_FFFFFF_V[1-9].*", "HE_Att"),
            ("HE", r".*HE-HV_FFFFFF_V[1-9].*", "HE_HV"),
            ("HE", r".*PM_FFFFFF_V[1-9].*", "HE_PM"),
            ("HE", r".*DTime_FFFFFF_V[1-9].*", "HE_DTime"),
            ("HE", r".*TH_FFFFFF_V[1-9].*", "HE_TH"),
            ("AUX", r".*EHK_FFFFFF_V[1-9].*", "HE_EHK"),
        ]

        for subdir, pattern, desc in files_patterns:
            full_dir = os.path.join(self.exppath, subdir)
            matched_files = sorted(self.find_files(full_dir, pattern))
            if matched_files:
                self.FileGraph.nodes[desc]["file"] = matched_files[-1]
            else:
                raise FileNotFoundError(f"{desc} not found!")
