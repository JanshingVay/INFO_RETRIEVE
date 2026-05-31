# 信息检索系统 - Information Retrieval System

一个功能完整的中文信息检索系统，包含高并发爬虫、多种检索算法、跨模态检索和可视化评价。

## 🌟 主要特性

- 🚀 **高并发异步爬虫** - 使用asyncio+aiohttp，速度可达50-100页/秒
- 📐 **双检索算法** - 支持TF-IDF向量空间模型和BM25概率模型
- 🖼️ **跨模态图像检索** - 支持Jina CLIP v2，90+语言文本→图像检索
- 💬 **相关反馈优化** - Rocchio查询扩展和文档评分boost
- 📈 **可视化评价** - Precision对比图、P-R曲线、算法雷达图
- 📋 **交互式界面** - 完整的命令行菜单系统

---

## 📁 项目结构

```
INFO_RETRIEVE/
├── config.py                    # 配置中心
├── preprocessor.py              # 中文文本预处理
├── data_cleaner.py              # 数据清洗
├── async_crawler.py             # 高性能异步爬虫
├── inverted_index.py            # 倒排索引构建
├── vsm.py                       # TF-IDF向量空间模型
├── bm25.py                      # BM25概率模型
├── multimodal_retrieval.py      # 跨模态图像检索
├── relevance_feedback.py        # 相关反馈系统
├── evaluator.py                 # 人工评价系统
├── visualization.py             # 可视化评价
├── main.py                      # 主程序入口
├── requirements.txt             # 依赖包
├── tests/                       # 单元测试
└── data/                        # 数据目录
    ├── images/                  # 待检索图像
    ├── sample_images/           # 示例图像
    ├── documents.json           # 爬取的文档
    ├── inverted_index.json      # 倒排索引
    ├── feedback.json            # 反馈数据
    └── charts/                  # 可视化图表
```

---

## 📖 模块详细说明

### 🔧 核心配置与数据处理

#### config.py
**作用**: 系统配置中心
- 定义所有路径（文档、索引、停用词等）
- 爬虫参数配置（超时、并发数、目标数量）
- 数据源配置（IT之家、36氪、新华网）
- 数据清理配置（相似度阈值、最小长度等）
- 评测查询列表

#### preprocessor.py
**作用**: 中文文本预处理
- 使用jieba进行中文分词
- 移除停用词（内置+自定义）
- 清洗HTML标签和特殊字符
- 过滤纯数字和单字符（非中文）
- 批量处理文档列表

#### data_cleaner.py
**作用**: 数据清洗和去重
- 移除HTML模板代码（版权、广告、导航）
- URL去重
- 标题相似度去重（SequenceMatcher）
- 过滤无效文档（过短、example.com）
- 空白字符标准化

---

### 🌐 爬虫模块

#### async_crawler.py
**作用**: 高性能异步网页爬虫
- 基于asyncio+aiohttp的并发架构
- 支持多数据源配置（IT之家、36氪、新华网科技）
- 连接池复用和信号量控制
- 自适应延迟和指数退避重试
- 随机User-Agent轮换
- 实时进度显示
- **性能**: 50-100页/秒（vs传统同步1-2页/秒）

---

### 📊 索引与检索算法

#### inverted_index.py
**作用**: 倒排索引构建与管理
- 倒排索引数据结构（词→文档列表）
- TF-IDF统计计算
- 文档长度统计
- 索引的保存/加载
- 统计信息获取（文档数、词项数、倒排记录数）

#### vsm.py
**作用**: TF-IDF向量空间模型
- 稀疏向量表示
- Cosine相似度计算
- 支持相关反馈（Rocchio查询扩展）
- 文档评分boost
- 摘要生成（查询词窗口）

#### bm25.py
**作用**: BM25概率检索模型
- BM25公式实现（k1=1.5, b=0.75）
- 词频饱和度处理
- 文档长度归一化
- 概率IDF计算
- 支持相关反馈优化
- 算法对比功能

---

### 🎨 多模态与可视化

#### multimodal_retrieval.py
**作用**: 跨模态文本→图像检索
- 支持OpenAI CLIP和Jina CLIP v2
- **Jina CLIP v2**: 90+语言支持（中文、英文、日文等）
- 图像特征提取
- 文本特征提取
- 余弦相似度检索
- 纯read-only模式，不会修改源图像
- 简单特征编码器（CLIP不可用时的fallback）

#### visualization.py
**作用**: 可视化评价图表生成
- Precision@10对比柱状图（TF-IDF vs BM25）
- Precision-Recall曲线
- 多算法性能雷达图
- 索引统计图表
- 爬虫性能对比图
- 支持中文显示（SimHei字体）
- 300DPI高清输出

---

### 📊 评价与反馈

#### relevance_feedback.py
**作用**: 相关反馈系统
- 用户评分记录（1-5分）
- Rocchio查询扩展（从好评文档提取高IDF词）
- 文档评分boost（根据历史评分调整排序）
- 反馈数据持久化（JSON）
- 统计信息获取

#### evaluator.py
**作用**: 人工评价系统
- 交互式评价界面
- Precision@10和Recall计算
- 反馈效果对比（before vs after）
- 预设查询评价
- 自定义查询评价
- 反馈数据统计展示

---

### 🎮 主程序

#### main.py
**作用**: 系统主入口
- 完整的命令行菜单界面
- 集成所有功能模块
- 自动加载已有数据
- 实时状态显示

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行系统

```bash
python main.py
```

### 3. 使用流程

```
请选择操作:
  [1] 异步爬取文档 - 从IT之家/36氪/新华网爬取科技文章
  [2] 构建索引 - 构建倒排索引，支持TF-IDF和BM25
  [3] 搜索文档 - TF-IDF/VSM检索
  [4] 搜索文档 - BM25检索
  [5] 算法对比 - 对比两种算法的检索结果
  [6] 跨模态图像检索 - 使用文本查询检索相关图像
  [7] 交互式查询 - 支持算法切换和反馈优化
  [8] 人工评价 - 评价检索结果并优化
  [9] 生成可视化图表 - 生成评测图表
  [10] 显示系统状态 - 查看当前数据状态
  [0] 退出
```

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                         用户界面 (main.py)                    │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ↓                ↓                ↓
┌───────────────┐  ┌───────────┐  ┌─────────────────┐
│   爬虫模块    │  │  检索模块 │  │  多模态模块    │
│async_crawler.py│  │vsm.py+bm25│  │multimodal_    │
└───────┬───────┘  └─────┬─────┘  │retrieval.py    │
        ↓                ↓        └────────┬────────┘
┌───────────────┐  ┌───────────┐           ↓
│数据清理模块   │  │ 索引模块  │  ┌─────────────────┐
│data_cleaner.py│  │inverted_  │  │  相关反馈模块  │
└───────┬───────┘  │index.py   │  │relevance_     │
        ↓          └─────┬─────┘  │feedback.py    │
┌───────────────┐        ↓        └────────┬────────┘
│文本预处理模块  │  ┌───────────┐           ↓
│preprocessor.py│  │ 评价模块  │  ┌─────────────────┐
└───────┬───────┘  │evaluator.py│  │ 可视化模块    │
        ↓          └─────┬─────┘  │visualization.py│
┌───────────────┐        ↓        └─────────────────┘
│  文档存储     │  ┌───────────┐
│documents.json│  │ feedback.json
└───────────────┘  └───────────┘
```

---

## 🔍 检索算法对比

| 特性 | TF-IDF (VSM) | BM25 |
|------|--------------|------|
| 模型类型 | 向量空间模型 | 概率模型 |
| 词频处理 | 线性TF | 饱和TF (k1) |
| 长度归一化 | Cosine | 文档长度因子 (b) |
| IDF计算 | log(N/df) | log((N-df+0.5)/(df+0.5)) |
| 实现复杂度 | 简单 | 中等 |
| 检索效果 | 良好 | 通常更好 |

---

## 🖼️ 多语言支持

### 使用Jina CLIP v2（推荐）

```python
from multimodal_retrieval import CLIPImageRetriever

retriever = CLIPImageRetriever(use_clip=True, use_jina_clip=True)
retriever.index_images()

# 中文查询
results = retriever.search("人工智能芯片")

# 英文查询
results = retriever.search("AI chip technology")

# 日文查询
results = retriever.search("AIチップ技術")
```

---

## 📈 性能指标

| 组件 | 性能 |
|------|------|
| 异步爬虫 | 50-100页/秒 |
| 索引构建 | ~1秒/100文档 |
| 检索速度 | <10ms/查询 |
| 并发连接 | 20个 |
| 支持文档数 | 100-150 |

---

## 🧪 测试

运行单元测试：

```bash
pytest tests/ -v
```

---

## 📋 配置说明

在 `config.py` 中可以修改：

- 爬虫目标网站和选择器
- 并发连接数
- 文档数量目标
- 停用词列表
- 数据清理参数

---

## 📊 数据目录说明

- `data/images/` - 存放待检索的图像（不会被修改）
- `data/sample_images/` - 自动生成的示例图像
- `data/documents.json` - 爬取的文档数据
- `data/inverted_index.json` - 构建的倒排索引
- `data/feedback.json` - 用户反馈数据
- `data/charts/` - 生成的可视化图表

---

## 🤝 贡献

欢迎提交Issue和Pull Request！

---

## 📄 许可证

MIT License

---

## 🙏 致谢

- jieba - 中文分词
- CLIP - 跨模态模型
- Jina AI - Jina CLIP v2

---

## 📞 联系方式

如有问题，请提交Issue！
