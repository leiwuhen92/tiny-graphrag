import os
from dotenv import load_dotenv

from tinygraph.model import GPUStackLLM, GpuStackEmbeddings
from tinygraph.graph import TinyGraph

# 加载 .env文件
load_dotenv()

llm = GPUStackLLM(
    base_url=os.getenv('LLM_BASE_URL'),
    api_key=os.getenv('LLM_API_KEY'),
    model_name=os.getenv('LLM_MODEL_NAME')
)
embeddings = GpuStackEmbeddings(
    api_url=os.getenv('EMBEDDING_BASE_URL'),
    api_key=os.getenv('EMBEDDING_API_KEY'),
    model_name=os.getenv('EMBEDDINGS_MODEL_NAME')
)

graph = TinyGraph(
    url=os.getenv('NEO4J_URL'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD'),
    llm=llm,
    embedding=embeddings
)


def test(document, query):
    # 添加文档数据
    graph.add_document(document)

    # 验证数据库连接
    count = 0
    with graph.driver.session() as session:
        result = session.run("MATCH (n) RETURN count(n) as count")
        count = result.single()["count"]
    print(f"数据库连接正常，节点数量: {count}")

    local_res = graph.local_query(query)
    print(f"👉 local_query's result: {local_res}")

    global_res = graph.global_query(query)
    print(f"👉 global_query's result: {global_res}")


if __name__ == '__main__':
    # print("测试英文文档".center(66, '*'))
    # test("example/data.md", "what is ML")

    print("测试中文文档".center(66, '*'))
    test("example/亲子运动知识库.txt", "亲子运动有哪些？")

'''
Processing 'example/亲子运动知识库.txt':   0%|          | 0/6 [00:00<?, ?it/s]******************************测试中文文档******************************
Document 'example/亲子运动知识库.txt' has been chunked.
Processing 'example/亲子运动知识库.txt': 100%|██████████| 6/6 [02:39<00:00, 26.62s/it]
29 entities and 23 triplets have been extracted.
社区数量: 6, 模块度: 0.7145557655954631
Received notification from DBMS server: <GqlStatusObject gql_status='01N03', status_description='warn: procedure field deprecated. The field `schema` of procedure gds.graph.drop() is deprecated.', position=<SummaryInputPosition line=1, column=1, offset=0>, raw_classification='DEPRECATION', classification=<NotificationClassification.DEPRECATION: 'DEPRECATION'>, raw_severity='WARNING', severity=<NotificationSeverity.WARNING: 'WARNING'>, diagnostic_record={'_classification': 'DEPRECATION', '_severity': 'WARNING', '_position': {'offset': 0, 'line': 1, 'column': 1}, 'OPERATION': '', 'OPERATION_CODE': '0', 'CURRENT_SCHEMA': '/'}> for query: "CALL gds.graph.drop('graph_help', false)"
generating community report: 100%|██████████| 6/6 [02:45<00:00, 27.63s/it]
All community report has been generated.
doc 'example/亲子运动知识库.txt' has been loaded.
数据库连接正常，节点数量: 26
Node(name='家庭接力跑 (Family Relay Race)', desc='A parent-child sports activity where family members are divided into teams to compete in a relay race; presented as the main exercise project in the article, aimed at children aged 6-12.', chunks_id=['chunk-4355a405eb29d55f7fb0069bceb35d1c'], entity_id='entity-ddd53e40211cc90887929155e2114886', similarity=np.float64(0.8244384491681956)) has 3 edges
👉 local_query's result: 根据提供的资料，亲子运动项目是专为父母与子女共同参与而设计的体育活动。目前资料中明确提到的具体项目包括：

1. **家庭接力跑**
   * **适合年龄**：6~12岁
   * **运动时间**：15~20分钟
   * **活动介绍**：家庭成员分成两队，进行接力跑比赛。
   * **运动目标**：旨在提高孩子的**奔跑速度**和**团队协作能力**。
   * **注意事项**：确保跑道平整，避免跌倒。

2. **家庭篮球赛**
   * 资料中提到“亲子运动项目”这一类别包含了“家庭接力跑”和“家庭篮球赛”，但关于后者的具体规则、适合年龄等详细信息在当前提供的资料中未详细展开。

**总结建议**：
您可以尝试组织“家庭接力跑”这类简单易行、对场地要求不高的活动，它能有效锻炼孩子的身体素质并促进家庭合作。如果想探索更多样的亲子运动，可以进一步查找关于“家庭篮球赛”或其他亲子运动项目的具体玩法和安全指南。
👉 global_query's result: 根据提供的资料，亲子运动项目是专门为父母与子女共同参与而设计的体育活动，旨在促进健康并增强家庭纽带。以下是资料中提到的具体亲子运动项目：

1. **家庭接力跑**：家庭成员分组进行跑步接力比赛，适合6-12岁儿童。
2. **家庭篮球赛**：旨在提高儿童篮球技能和团队合作意识。
3. **跳绳接力**：父母与孩子轮流跳绳完成设定目标，适合6-12岁儿童，增加运动趣味性。
4. **滚雪球大赛**：冬季游戏，参与者在雪地上滚雪球，适合4-10岁儿童。
5. **拔河小勇士**：以拔河比赛为核心的亲子活动，适合5-12岁儿童。
6. **亲子瑜伽**：旨在增强亲子亲密关系，改善儿童平衡能力和柔韧性。

这些活动通常设计有具体目标，如提升耐力、动手能力、创造力或亲子默契等。您可以根据孩子的年龄和兴趣选择合适的项目进行尝试。
'''




