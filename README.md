# 中文信息检索系统

本项目是《信息与知识获取》作业 2 的课程实验实现，主要完成中文新闻数据采集、文本预处理、倒排索引构建、向量空间模型检索、BM25 对比排序、人工相关性评价、图形界面展示和跨模态多媒体检索。

系统当前以中文科技新闻为主要数据源，文档通过网络爬虫采集后存储在本地 JSON 文件中。文本检索主模型为 TF-IDF 向量空间模型，同时实现 BM25 作为优化对比；多媒体部分使用 Jina CLIP v2 支持文本到图片、文本到视频的语义检索。

## 1. 项目结构

```text
INFO_RETRIEVE/
├── app.py                    # Streamlit 图形界面
├── main.py                   # 命令行主入口
├── config.py                 # 路径、爬虫、数据规模等配置
├── async_crawler.py          # 异步网络爬虫
├── data_cleaner.py           # 文档清洗与去重
├── preprocessor.py           # 中文分词与停用词过滤
├── inverted_index.py         # 倒排索引构建与加载
├── vsm.py                    # TF-IDF 向量空间模型
├── bm25.py                   # BM25 检索模型
├── relevance_feedback.py     # 相关反馈与评分记录
├── evaluator.py              # 人工评价流程
├── multimodal_retrieval.py   # 跨模态图片/视频检索
├── visualization.py          # 实验图表生成
├── requirements.txt          # Python 依赖
├── tests/                    # 单元测试
├── data/
│   ├── documents.json        # 爬取并清洗后的文档
│   ├── inverted_index.json   # 倒排索引文件
│   ├── feedback.json         # 人工评分与反馈数据
│   ├── eval_queries.json     # 人工评价查询集合
│   ├── images/               # 图片检索数据
│   ├── videos/               # 视频检索数据
│   └── charts/               # 可视化结果图
└── 实验报告.md
```

## 2. 已实现功能

- 中文数据爬取：使用 `aiohttp`、`BeautifulSoup` 等开源工具采集中文科技新闻。
- 本地数据存储：文档、索引、评价数据、多媒体索引均保存在 `data/` 目录。
- 文本预处理：使用 `jieba` 进行中文分词，并进行停用词过滤和基础清洗。
- 倒排索引：建立词项到文档的倒排记录，并保存词频、文档长度、IDF 等统计信息。
- 向量空间模型：实现 TF-IDF 文档向量和查询向量，通过余弦相似度排序。
- BM25 对比模型：实现 BM25 排序，用于和 VSM 检索结果进行对比分析。
- 自然语言查询：支持直接输入中文查询语句，输出相关度、标题、摘要、URL、日期等信息。
- 人工评价：支持对检索结果进行 1-5 分评分，并保存反馈数据。
- 相关反馈优化：使用历史评分和 Rocchio 思路对后续检索排序进行优化。
- 图形界面：使用 Streamlit 实现系统状态、文本检索、多媒体检索、人工评价和图表展示。
- 多媒体检索：基于 Jina CLIP v2 实现文本到图片、文本到视频的跨模态检索。
- 可视化分析：生成索引统计、算法对比、P-R 曲线、爬虫性能等图表。

## 3. 环境准备

建议使用 Python 虚拟环境运行项目。

```powershell
cd E:\BUPTStudy\课程\信息与知识获取\final\lab2\INFO_RETRIEVE
python -m venv .venv
.venv\Scripts\Activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果已经创建过虚拟环境，只需要进入项目目录并激活：

```powershell
cd E:\BUPTStudy\课程\信息与知识获取\final\lab2\INFO_RETRIEVE
.venv\Scripts\Activate
```

## 4. 命令行运行

启动命令行系统：

```powershell
python main.py
```

主菜单功能如下：

```text
[1] 异步爬取文档
[2] 构建索引
[3] 搜索文档 - TF-IDF/VSM
[4] 搜索文档 - BM25
[5] 算法对比
[6] 跨模态多媒体检索
[7] 交互式查询
[8] 人工评价检索结果
[9] 生成可视化图表
[10] 显示系统状态
[0] 退出
```

推荐首次运行顺序：

```text
1 -> 2 -> 3/4/5 -> 8 -> 9 -> 6 -> 10
```

当前项目配置中，爬虫目标规模为 700 到 800 篇文档。若本地已经存在 `data/documents.json` 和 `data/inverted_index.json`，可以直接从检索、评价和图形界面部分开始测试。

## 5. 图形界面运行

启动 Streamlit 图形界面：

```powershell
streamlit run app.py
```

浏览器打开后可以测试以下页面：

- 系统状态：查看文档数、索引文档数、词项数、评分数、图片/视频索引数和数据源分布。
- 文本检索：输入中文查询，选择 VSM 或 BM25，查看按相关度排序的检索结果。
- 多媒体检索：输入图片或视频语义查询，查看跨模态检索结果。
- 人工评价：查看人工评分统计和评价查询集合。
- 可视化图表：查看实验生成的统计图和算法对比图。

如果命令行刚刚重新生成了数据或索引，图形界面中可以点击侧边栏的“刷新数据缓存”按钮，让页面重新读取最新文件。

## 6. 多媒体检索说明

多媒体模块使用 `jinaai/jina-clip-v2`。首次运行时需要加载模型，CPU 环境下耗时会比较明显，出现 CUDA、Flash Attention 或 xFormers 不可用的警告属于正常情况，不影响 CPU 推理。

命令行中选择 `[6] 跨模态多媒体检索` 后，可以选择：

```text
[1] 仅图片
[2] 仅视频
[3] 全部
```

索引完成后会生成或更新：

```text
data/image_index.pkl
data/image_metadata.json
data/video_index.pkl
data/video_metadata.json
```

可以用下面命令检查图片和视频索引数量：

```powershell
python -c "import json; print('images', len(json.load(open('data/image_metadata.json', encoding='utf-8')))); print('videos', len(json.load(open('data/video_metadata.json', encoding='utf-8'))))"
```

## 7. 常用检查命令

检查文档和索引规模：

```powershell
python -c "import json; docs=json.load(open('data/documents.json',encoding='utf-8')); idx=json.load(open('data/inverted_index.json',encoding='utf-8')); print('docs',len(docs)); print('unique ids',len(set(d['id'] for d in docs))); print('index docs',idx.get('doc_count')); print('doc_lengths',len(idx.get('doc_lengths',{})))"
```

检查人工评价数据：

```powershell
python -c "import json; fb=json.load(open('data/feedback.json',encoding='utf-8')); eq=json.load(open('data/eval_queries.json',encoding='utf-8')); print(fb.get('stats')); print([len(q.get('relevant_docs',[])) for q in eq])"
```

运行单元测试：

```powershell
pytest tests -v
```

若当前系统临时目录权限异常，个别依赖 `tmp_path` 的测试可能受环境影响。此时可以先运行核心测试：

```powershell
pytest tests -v -k "not save_and_load"
```

## 8. 完整验收测试建议

一次完整验收可以按下面顺序执行：

1. 激活虚拟环境并确认依赖安装完成。
2. 检查 `data/documents.json` 中文档数是否在 700 到 800 篇之间。
3. 检查 `data/inverted_index.json` 的 `doc_count` 是否与文档数一致。
4. 在命令行中分别测试 `[3]` VSM 检索和 `[4]` BM25 检索。
5. 使用 `[5]` 对同一查询进行算法对比。
6. 使用 `[8]` 查看或补充人工评价数据。
7. 使用 `[9]` 生成可视化图表。
8. 使用 `[6]` 建立图片和视频索引，并测试跨模态查询。
9. 使用 `streamlit run app.py` 打开图形界面。
10. 在图形界面中依次测试系统状态、文本检索、多媒体检索、人工评价和可视化图表。

建议保留以下截图用于实验报告或课堂展示：

- 命令行系统状态截图。
- VSM 或 BM25 文本检索结果截图。
- Streamlit 系统状态页面截图。
- Streamlit 文本检索页面截图。
- 多媒体检索结果截图。
- 人工评价或可视化图表截图。

## 9. 依赖与模型

文本检索部分主要依赖：

- `requests`
- `beautifulsoup4`
- `aiohttp`
- `jieba`
- `numpy`
- `matplotlib`
- `streamlit`
- `pandas`

多媒体检索部分主要依赖：

- `torch`
- `transformers`
- `sentence-transformers`
- `Pillow`
- `opencv-python`
- `timm`
- `torchvision`

多媒体模型为 Jina CLIP v2。模型通常会缓存在本机 Hugging Face 缓存目录中，后续运行不需要重复下载。

## 10. 课程报告材料

本项目包含：

- 源代码文件。
- 实验报告：`实验报告.md`。
- 本地数据文件：`data/documents.json`、`data/inverted_index.json`、`data/feedback.json` 等。
- 可视化图表：`data/charts/`。
- 多媒体测试数据：`data/images/`、`data/videos/`。

如果使用 Git 提交，需要注意 `.gitignore` 是否忽略了 `data/` 目录中的实验数据；如果使用压缩包提交，则直接保留完整项目目录即可。
