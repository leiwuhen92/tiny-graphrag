from neo4j import GraphDatabase
import os
import json
from typing import Dict, List
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm

from tinygraph.utils import (
    get_text_inside_tag,
    cosine_similarity,
    compute_mdhash_id,
    read_json_file,
    write_json_file,
    create_file_if_not_exists,
)
from tinygraph.prompt import *


@dataclass
class Node:
    name: str
    desc: str
    chunks_id: list
    entity_id: str
    similarity: float


class TinyGraph:
    """
    一个用于处理图数据库和语言模型的类。

    该类通过连接到Neo4j图数据库，并使用语言模型（LLM）和嵌入模型（Embedding）来处理文档和图数据。
    它还管理一个工作目录，用于存储文档、文档块和社区数据。
    """

    def __init__(
            self,
            url: str,
            username: str,
            password: str,
            llm,
            embedding,
            working_dir: str = "workspace"
    ):
        """
        初始化TinyGraph类。

        参数:
        - url: Neo4j数据库的URL
        - username: Neo4j数据库的用户名
        - password: Neo4j数据库的密码
        - llm: 语言模型（LLM）实例
        - embeddings: 嵌入模型（Embedding）实例
        - working_dir: 工作目录，默认为"workspace"
        """
        self.driver = GraphDatabase.driver(
            uri=url,
            auth=(username, password)
        )                                             # 创建Neo4j数据库驱动
        self.llm = llm                                # 设置语言模型
        self.embedding = embedding                    # 设置嵌入模型
        self.working_dir = working_dir                # 设置工作目录
        os.makedirs(self.working_dir, exist_ok=True)  # 创建工作目录（如果不存在）

        # 定义文档、文档块和社区数据的文件路径
        self.doc_path = os.path.join(working_dir, "doc.txt")
        self.chunk_path = os.path.join(working_dir, "chunk.json")
        self.community_path = os.path.join(working_dir, "community.json")

        # 创建文件（如果不存在）
        create_file_if_not_exists(self.doc_path)
        create_file_if_not_exists(self.chunk_path)
        create_file_if_not_exists(self.community_path)

        # 加载已加载的文档
        self.loaded_documents = self.get_loaded_documents()

    def create_triplet_2_db(self, subject: dict, predicate, object: dict) -> None:
        """
        创建一个三元组（Triplet）并将其存储到Neo4j数据库中。

        参数:
        - subject: 主题实体的字典，包含名称、描述、块ID和实体ID
        - predicate: 关系名称
        - object: 对象实体的字典，包含名称、描述、块ID和实体ID

        返回:
        - 查询结果
        """
        # 定义Cypher查询语句，用于创建或合并实体节点和关系
        query = (
            "MERGE (a:Entity {name: $subject_name, description: $subject_desc, chunks_id: $subject_chunks_id, entity_id: $subject_entity_id}) "
            "MERGE (b:Entity {name: $object_name, description: $object_desc, chunks_id: $object_chunks_id, entity_id: $object_entity_id}) "
            "MERGE (a)-[r:Relationship {name: $predicate}]->(b) "
            "RETURN a, b, r"
        )

        # 使用数据库会话执行查询
        with self.driver.session() as session:
            result = session.run(
                query,
                subject_name=subject["name"],
                subject_desc=subject["description"],
                subject_chunks_id=subject["chunks id"],
                subject_entity_id=subject["entity id"],
                object_name=object["name"],
                object_desc=object["description"],
                object_chunks_id=object["chunks id"],
                object_entity_id=object["entity id"],
                predicate=predicate,
            )

        return

    @staticmethod
    def split_text(file_path: str, segment_length=300, overlap_length=50) -> Dict:
        """
        将文本文件分割成多个片段，每个片段的长度为segment_length，相邻片段之间有overlap_length的重叠。

        参数:
        - file_path: 文本文件的路径
        - segment_length: 每个片段的长度，默认为300
        - overlap_length: 相邻片段之间的重叠长度，默认为50

        返回:
        - 包含片段ID和片段内容的字典
        """
        chunks = {}                   # 用于存储片段的字典
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()     # 读取文件内容

        text_segments = []            # 用于存储分割后的文本片段
        start_index = 0               # 初始化起始索引

        # 循环分割文本，直到剩余文本长度不足以形成新的片段
        while start_index + segment_length <= len(content):
            text_segments.append(content[start_index: start_index + segment_length])
            start_index += segment_length - overlap_length  # 更新起始索引，考虑重叠长度

        # 处理剩余的文本，如果剩余文本长度小于segment_length但大于0
        if start_index < len(content):
            text_segments.append(content[start_index:])

        # 为每个片段生成唯一的ID，并将其存储在字典中
        for segment in text_segments:
            chunks.update({compute_mdhash_id(segment, prefix="chunk-"): segment})

        return chunks

    def get_entity(self, text: str, chunk_id: str) -> List[Dict]:
        """
        从给定的文本中提取实体，并为每个实体生成唯一的ID和描述。

        参数:
        - text: 输入的文本
        - chunk_id: 文本块的ID

        返回:
        - 包含提取的实体信息的列表
        """
        try:
            # 使用语言模型预测实体信息
            data = self.llm.invoke(GET_ENTITY.format(text=text))
        except Exception as e:
            print(f"get entity error: {e}")
            return []

        concepts = []  # 用于存储提取的实体信息

        # 从预测结果中提取实体信息
        for concept_html in get_text_inside_tag(data, "concept"):
            concept = {
                "name": get_text_inside_tag(concept_html, "name")[0].strip(),
                "description": get_text_inside_tag(concept_html, "description")[0].strip(),
                "chunks id": [chunk_id]
            }
            concept["entity id"] = compute_mdhash_id(concept["description"], prefix="entity-")
            concepts.append(concept)

        return concepts

    def get_triplets(self, content, entity: list) -> List[Dict]:
        """
        从给定的内容中提取三元组（Triplet）信息，并返回包含这些三元组信息的列表。

        参数:
        - content: 输入的内容
        - entity: 实体列表

        返回:
        - 包含提取的三元组信息的列表
        """
        try:
            # 使用语言模型预测三元组信息
            data = self.llm.invoke(GET_TRIPLETS.format(text=content, entity=entity))
            data = get_text_inside_tag(data, "triplet")
        except Exception as e:
            print(f"get triplets error: {e}")
            return []

        res = []  # 用于存储提取的三元组信息

        # 从预测结果中提取三元组信息
        for triplet_data in data:
            try:
                res.append({
                    "subject": get_text_inside_tag(triplet_data, "subject")[0],
                    "subject_id": get_text_inside_tag(triplet_data, "subject_id")[0],
                    "predicate": get_text_inside_tag(triplet_data, "predicate")[0],
                    "object": get_text_inside_tag(triplet_data, "object")[0],
                    "object_id": get_text_inside_tag(triplet_data, "object_id")[0],
                })
            except Exception as e:
                print(f"Error extracting triplet: {e}", flush=True)
                continue

        return res

    def add_document(self, filepath, use_llm_disambiguation=False) -> None:
        """
        将文档添加到系统中，执行以下步骤：
        1. 检查文档是否已经加载。
        2. 将文档分割成块。
        3. 从块中提取实体和三元组。
        4. 执行实体消岐，有两种方法可选，默认将同名实体认为即为同一实体。
        5. 合并实体和三元组。
        6. 将合并的实体和三元组存储到Neo4j数据库中。

        参数:
        - filepath: 要添加的文档的路径
        - use_llm_disambiguation: 是否使用LLM进行实体消岐
        """
        # ================ Check if the document has been loaded(检查文档是否已经分块) ================
        if filepath in self.get_loaded_documents():
            print(f"Document '{filepath}' has already been loaded, skipping import process.", flush=True)
            return

        # ================ Chunking(将文档分割成块) ================
        chunks = self.split_text(filepath)
        existing_chunks = read_json_file(self.chunk_path)

        # Filter out chunks that are already in storage
        new_chunks = {k: v for k, v in chunks.items() if k not in existing_chunks}

        if not new_chunks:
            print("All chunks are already in the storage.", flush=True)
            return

        # Merge new chunks with existing chunks
        all_chunks = {**existing_chunks, **new_chunks}
        write_json_file(all_chunks, self.chunk_path)  # 写入chunk.json
        print(f"Document '{filepath}' has been chunked.", flush=True)

        # ================ Entity Extraction(从块中提取实体（entities）和三元组（triplets）) ================
        all_entities = []
        all_triplets = []
        for chunk_id, chunk_content in tqdm(new_chunks.items(), desc=f"Processing '{filepath}'"):
            try:
                # 从当前分块中提取实体，每个实体包含名称、描述、关联的分块ID以及唯一的实体ID
                entities = self.get_entity(chunk_content, chunk_id=chunk_id)
                all_entities.extend(entities)

                # 从当前分块中提取三元组，每个三元组由主语（subject）、谓语（predicate）和宾语（object）组成，表示实体之间的关系
                triplets = self.get_triplets(chunk_content, entities)
                all_triplets.extend(triplets)
            except Exception as e:
                raise Exception(f"An error occurred while processing chunk '{chunk_id}'. SKIPPING...")
        print(f"{len(all_entities)} entities and {len(all_triplets)} triplets have been extracted.", flush=True)

        # ================ Entity Disambiguation(实体消歧) ================
        entity_names = list(set(entity["name"] for entity in all_entities))

        entity_id_mapping = {}
        if use_llm_disambiguation:
            for name in entity_names:
                same_name_entities = [entity for entity in all_entities if entity["name"] == name]
                if len(same_name_entities) > 1:
                    transform_text = self.llm.invoke(ENTITY_DISAMBIGUATION.format(same_name_entities))
                    entity_id_mapping.update(get_text_inside_tag(transform_text, "transform"))
        else:
            for entity in all_entities:
                entity_name = entity["name"]
                if entity_name not in entity_id_mapping:
                    entity_id_mapping[entity_name] = entity["entity id"]

        for entity in all_entities:
            entity["entity id"] = entity_id_mapping.get(entity["name"], entity["entity id"])

        # 去除无效的三元组（如果三元组的主语或者宾语的实体ID无法在entity_id_mapping中找到，则将其标记为无效）后更新三元组（更新其主语和宾语的实体ID）
        triplets_to_remove = [
            triplet
            for triplet in all_triplets
            if entity_id_mapping.get(triplet["subject"], triplet["subject_id"]) is None
            or entity_id_mapping.get(triplet["object"], triplet["object_id"]) is None
        ]
        updated_triplets = [
            {
                **triplet,
                "subject_id": entity_id_mapping.get(triplet["subject"], triplet["subject_id"]),
                "object_id": entity_id_mapping.get(triplet["object"], triplet["object_id"]),
            }
            for triplet in all_triplets
            if triplet not in triplets_to_remove
        ]
        all_triplets = updated_triplets

        # ================ Merge Entities(合并实体和三元组) ================
        entity_map = {}
        for entity in all_entities:
            entity_id = entity["entity id"]
            if entity_id not in entity_map:
                entity_map[entity_id] = {
                    "name": entity["name"],
                    "description": entity["description"],
                    "chunks id": [],
                    "entity id": entity_id,
                }
            else:
                entity_map[entity_id]["description"] += " " + entity["description"]
            entity_map[entity_id]["chunks id"].extend(entity["chunks id"])

        # ================ Store Data in Neo4j(将合并的实体和三元组存储到Neo4j的图数据库中) ================
        for triplet in all_triplets:
            subject_id = triplet["subject_id"]
            object_id = triplet["object_id"]

            subject = entity_map.get(subject_id)
            object = entity_map.get(object_id)
            if subject and object:
                self.create_triplet_2_db(subject, triplet["predicate"], object)

        # ================ communities(生成社区与社区报告) ================
        self.gen_community()
        self.generate_community_report()

        # ================ embedding(生成嵌入式向量) ================
        self.add_embedding_for_graph()
        # 将处理过的文档路径记录到缓存文件中，避免重复处理
        self.add_loaded_documents(filepath)
        print(f"doc '{filepath}' has been loaded.", flush=True)

    def detect_communities(self) -> None:
        # 每次运行前强制drop
        with self.driver.session() as session:
            session.run("CALL gds.graph.drop('graph_help', false) YIELD graphName")

        # 从Neo4j的存储图中，把Entity节点和关系复制到GDS的内存图里，命名为graph_help。供leiden等算法使用。graph_help这个图名必须唯一，若已经存在不会自动覆盖，会继续使用旧图，故需要先删除旧图
        query = """
        CALL gds.graph.project(
            'graph_help',
            ['Entity'],
            {
                Relationship: {
                    orientation: 'UNDIRECTED'
                }
            }
        )
        """
        with self.driver.session() as session:
            result = session.run(query)

        query = """
        CALL gds.leiden.write('graph_help', {
            writeProperty: 'communityIds',
            includeIntermediateCommunities: True,
            maxLevels: 10,
            tolerance: 0.0001,
            gamma: 1.0,
            theta: 0.01
        })
        YIELD communityCount, modularity, modularities
        """
        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                print(f"社区数量: {record['communityCount']}, 模块度: {record['modularity']}", flush=True)
            session.run("CALL gds.graph.drop('graph_help', false)")   # 第二个参数表示：如果要drop的GDS内存图不存在，是否抛出错误

    def get_entity_by_name(self, name):
        query = """
        MATCH (n:Entity {name: $name})
        RETURN n
        """
        with self.driver.session() as session:
            result = session.run(query, name=name)
            entities = [record["n"].get("name") for record in result]
        return entities[0]

    def get_node_edges(self, node: Node):
        """获得某个实体的所有关系边"""
        query = """
        MATCH (n)-[r]-(m)
        WHERE n.entity_id = $id
        RETURN n.name AS n,r.name AS r,m.name AS m
        """
        with self.driver.session() as session:
            result = session.run(query, id=node.entity_id)
            edges = [(record["n"], record["r"], record["m"]) for record in result]

        print(f"{node} has {len(edges)} edges", flush=True)
        return edges

    def get_node_chunks(self, node):
        """从chunk.json获取对应实体的chunk内容"""
        existing_chunks = read_json_file(self.chunk_path)
        chunks = [existing_chunks[i] for i in node.chunks_id]
        return chunks

    def add_embedding_for_graph(self):
        query = """
        MATCH (n)
        RETURN n
        """
        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                node = record["n"]
                description = node["description"]
                id = node["entity_id"]
                embedding = self.embedding.embed_documents(description)
                # 更新节点，添加新的 embedding 属性
                update_query = """
                MATCH (n {entity_id: $id})
                SET n.embedding = $embedding
                """
                session.run(update_query, id=id, embedding=embedding)

    def get_topk_similar_entities(self, input_emb, k=1) -> List[Node]:
        """"根据余弦相似度，获取前k个最相似的实体

        :param input_emb: 输出查询的embedding

        :return: 相似度降序排列的实体列表
        """
        res = []
        query = """
        MATCH (n)
        RETURN n
        """
        with self.driver.session() as session:
            result = list(session.run(query))

        for record in result:
            node = record["n"]
            if node["embedding"] is not None:
                similarity = cosine_similarity(input_emb, node["embedding"])
                node = Node(
                    name=node["name"],
                    desc=node["description"],
                    chunks_id=node["chunks_id"],
                    entity_id=node["entity_id"],
                    similarity=similarity,
                )
                res.append(node)
        return sorted(res, key=lambda x: x.similarity, reverse=True)[:k]

    def get_communities(self, nodes: List[Node]):
        communities_schema = self.read_community_schema()
        res = []
        nodes_ids = [i.entity_id for i in nodes]
        for community_id, community_info in communities_schema.items():
            if set(nodes_ids) & set(community_info["nodes"]):
                res.append({
                    "community_id": community_id,
                    "community_info": community_info["report"],
                })
        return res

    def get_relations(self, nodes: List, input_emb):
        """调用get_node_edges方法，获取该实体的所有关系边，认为这些边就是similar relation

        :param nodes: 实体
        :param input_emb: 输入查询的embedding
        """
        res = []
        for i in nodes:
            res.append(self.get_node_edges(i))
        return res

    def get_chunks(self, nodes, input_emb):
        """调用get_node_chunks方法，获取该实体关联的文档块，认为这些文档块就是similar chunks

        :param nodes: 实体
        :param input_emb: 输入查询的embedding
        """
        chunks = []
        for i in nodes:
            chunks.append(self.get_node_chunks(i))
        return chunks

    def gen_community_schema(self) -> dict[str, dict]:
        results = defaultdict(
            lambda: dict(
                level=None,
                title=None,
                edges=set(),
                nodes=set(),
                chunk_ids=set(),
                sub_communities=[],
            )
        )

        with self.driver.session() as session:
            # Fetch community data
            result = session.run(
                f"""
                MATCH (n:Entity)
                WITH n, n.communityIds AS communityIds, [(n)-[]-(m:Entity) | m.entity_id] AS connected_nodes
                RETURN n.entity_id AS node_id, 
                       communityIds AS cluster_key,
                       connected_nodes
                """
            )

            max_num_ids = 0
            for record in result:
                for index, c_id in enumerate(record["cluster_key"]):
                    node_id = str(record["node_id"])
                    level = index
                    cluster_key = str(c_id)
                    connected_nodes = record["connected_nodes"]

                    results[cluster_key]["level"] = level
                    results[cluster_key]["title"] = f"Cluster {cluster_key}"
                    results[cluster_key]["nodes"].add(node_id)
                    results[cluster_key]["edges"].update(
                        [
                            tuple(sorted([node_id, str(connected)]))
                            for connected in connected_nodes
                            if connected != node_id
                        ]
                    )
            for k, v in results.items():
                v["edges"] = [list(e) for e in v["edges"]]
                v["nodes"] = list(v["nodes"])
                v["chunk_ids"] = list(v["chunk_ids"])
            for cluster in results.values():
                cluster["sub_communities"] = [
                    sub_key
                    for sub_key, sub_cluster in results.items()
                    if sub_cluster["level"] > cluster["level"]
                    and set(sub_cluster["nodes"]).issubset(set(cluster["nodes"]))
                ]

        return dict(results)

    def gen_community(self):
        """生成社区：通过图算法（本项目为 Leiden 算法）检测图中的社区结构"""
        # 使用 Neo4j 的图算法（如 gds.leiden.write）检测社区
        self.detect_communities()

        # 生成社区架构（community schema），包括社区的层级、节点、边等信息
        community_schema = self.gen_community_schema()

        # 将社区架构存储到 community.json 文件中
        with open(self.community_path, "w", encoding="utf-8") as file:
            json.dump(community_schema, file, indent=4)

    def read_community_schema(self) -> dict:
        try:
            with open(self.community_path, "r", encoding="utf-8") as file:
                community_schema = json.load(file)
        except:
            raise FileNotFoundError("Community schema not found. Please make sure to generate it first.")
        return community_schema

    def get_loaded_documents(self):
        try:
            with open(self.doc_path, "r", encoding="utf-8") as file:
                lines = file.readlines()
                return set(line.strip() for line in lines)
        except:
            raise FileNotFoundError("Cache file not found.")

    def add_loaded_documents(self, file_path):
        if file_path in self.loaded_documents:
            print(f"Document '{file_path}' has already been loaded, skipping addition to cache.", flush=True)
            return

        # 追加写入doc.txt
        with open(self.doc_path, "a", encoding="utf-8") as file:
            file.write(file_path + "\n")
        self.loaded_documents.add(file_path)

    def get_node_by_id(self, node_id):
        query = """
        MATCH (n:Entity {entity_id: $node_id})
        RETURN n
        """
        with self.driver.session() as session:
            result = session.run(query, node_id=node_id)
            nodes = [record["n"] for record in result]
        return nodes[0]

    def get_edges_by_id(self, src, tar):
        query = """
        MATCH (n:Entity {entity_id: $src})-[r]-(m:Entity {entity_id: $tar})
        RETURN {src: n.name, r: r.name, tar: m.name} AS R
        """
        with self.driver.session() as session:
            result = session.run(query, {"src": src, "tar": tar})
            edges = [record["R"] for record in result]
        return edges[0]

    def gen_single_community_report(self, community: dict):
        nodes = community["nodes"]
        edges = community["edges"]
        nodes_describe = []
        edges_describe = []

        # 遍历每个社区，生成包含实体和关系的报告
        for i in nodes:
            node = self.get_node_by_id(i)
            nodes_describe.append({"name": node["name"], "desc": node["description"]})
        for i in edges:
            edge = self.get_edges_by_id(i[0], i[1])
            edges_describe.append(
                {"source": edge["src"], "target": edge["tar"], "desc": edge["r"]}
            )
        nodes_csv = "entity,description\n"
        for node in nodes_describe:
            nodes_csv += f"{node['name']},{node['desc']}\n"
        edges_csv = "source,target,description\n"
        for edge in edges_describe:
            edges_csv += f"{edge['source']},{edge['target']},{edge['desc']}\n"

        # 报告通过大语言模型（LLM）生成，描述社区的结构和内容
        data = f"""
        Text:
        -----Entities-----
        ```csv
        {nodes_csv}
        ```
        -----Relationships-----
        ```csv
        {edges_csv}
        ```"""
        prompt = GEN_COMMUNITY_REPORT.format(input_text=data)
        report = self.llm.invoke(prompt)
        return report

    def generate_community_report(self):
        """生成社区报告：借助大语言模型为每个社区生成详细的报告，描述社区中的实体和关系"""
        communities_schema = self.read_community_schema()

        for community_key, community in tqdm(communities_schema.items(), desc="generating community report"):
            community["report"] = self.gen_single_community_report(community)

        with open(self.community_path, "w", encoding="utf-8") as file:
            json.dump(communities_schema, file, indent=4)
        print("All community report has been generated.", flush=True)

    def build_local_query_context(self, query):
        """根据用户问题得到的包括社区、实体、关系、文档块这四部分内容的上下文
        :param query: 用户问题

        :return: 返回的是一个多行字符串，包括:
            Reports：社区报告；
            Entities：与查询最相似的实体；
            Relationships：这些实体之间的关系；
            Sources：这些实体关联的文档块
        """
        # 输入查询的嵌入向量
        query_emb = self.embedding.embed_documents(query)

        # 根据余弦相似找到前k个最相似的实体
        topk_similar_entities_context = self.get_topk_similar_entities(query_emb)

        # 根据上一步得到的最相似实体，获得前k个最相似的社区
        topk_similar_communities_context = self.get_communities(topk_similar_entities_context)

        # 根据最相似的实体，获取前k个最相似的关系
        topk_similar_relations_context = self.get_relations(topk_similar_entities_context, query)

        # 根据最相似的实体，获取前k个最相似的文档块
        topk_similar_chunks_context = self.get_chunks(topk_similar_entities_context, query)

        return f"""
        -----Reports-----
        ```csv
        {topk_similar_communities_context}
        ```
        -----Entities-----
        ```csv
        {topk_similar_entities_context}
        ```
        -----Relationships-----
        ```csv
        {topk_similar_relations_context}
        ```
        -----Sources-----
        ```csv
        {topk_similar_chunks_context}
        ```
        """

    def map_community_points(self, community_info_report, query):
        """结合社区报告和大语言模型的能力，为每个候选社区生成与查询内容相关程度的评分

        :param community_info_report: 社区报告
        :query: 用户问题
        """
        points_html = self.llm.invoke(GLOBAL_MAP_POINTS.format(context_data=community_info_report, query=query))

        points = get_text_inside_tag(points_html, "point")

        res = []
        for point in points:
            try:
                score = get_text_inside_tag(point, "score")[0]
                desc = get_text_inside_tag(point, "description")[0]
                res.append((desc, score))
            except:
                continue
        return res

    def build_global_query_context(self, query, level=1):
        """根据用户问题得到的包含社区描述和分数的上下文

        :param query: 用户问题
        :param level: 层级
        """
        communities_schema = self.read_community_schema()

        candidate_community = {}  # 候选社区
        # 筛选符合层级要求的社区，存到候选社区 candidate_community中
        for communityid, community_info in communities_schema.items():
            if community_info["level"] < level:
                candidate_community.update({communityid: community_info})

        points = []  # 社区评分列表
        #  计算候选社区的评分
        for communityid, community_info in candidate_community.items():
            points.extend(self.map_community_points(community_info["report"], query))
        points = sorted(points, key=lambda x: x[-1], reverse=True)
        return points  # 得到包含描述和分数的列表，描述是社区的描述，分数是查询的相关性得分

    def local_query(self, query):
        context = self.build_local_query_context(query)  # 根据用户问题得到的包括社区、实体、关系、文档块这四部分内容的上下文
        prompt = LOCAL_QUERY.format(query=query, context=context)
        response = self.llm.invoke(prompt)
        return response

    def global_query(self, query, level=1):
        context = self.build_global_query_context(query, level)  # 根据用户问题得到的包含社区描述和分数的上下文
        prompt = GLOBAL_QUERY.format(query=query, context=context)
        response = self.llm.invoke(prompt)
        return response
