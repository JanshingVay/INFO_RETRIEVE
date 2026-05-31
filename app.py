import json
import os
from pathlib import Path
from collections import Counter

import pandas as pd
import streamlit as st

from bm25 import BM25Model
from config import DATA_DIR, DOCUMENTS_FILE, EVAL_QUERIES_FILE, INDEX_FILE
from inverted_index import InvertedIndex
from preprocessor import TextPreprocessor
from vsm import VectorSpaceModel


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = Path(DATA_DIR)
CHART_DIR = DATA_PATH / "charts"
FEEDBACK_FILE = DATA_PATH / "feedback.json"
VIDEO_METADATA_FILE = DATA_PATH / "video_metadata.json"
VIDEO_INDEX_FILE = DATA_PATH / "video_index.pkl"
IMAGE_METADATA_FILE = DATA_PATH / "image_metadata.json"
IMAGE_INDEX_FILE = DATA_PATH / "image_index.pkl"


st.set_page_config(
    page_title="中文信息检索系统",
    page_icon="",
    layout="wide",
)


st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .result-box {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 14px 16px;
        margin: 10px 0;
        background: #ffffff;
    }
    .muted { color: #6b7280; font-size: 0.9rem; }
    .score { font-weight: 700; color: #0f766e; }
    </style>
    """,
    unsafe_allow_html=True,
)


def read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def file_stamp(path: Path):
    if not path.exists():
        return None
    stat = path.stat()
    return (stat.st_mtime_ns, stat.st_size)


@st.cache_data(show_spinner=False)
def load_documents(docs_stamp):
    docs = read_json(Path(DOCUMENTS_FILE), [])
    return docs if isinstance(docs, list) else []


@st.cache_data(show_spinner=False)
def load_index_raw(index_stamp):
    data = read_json(Path(INDEX_FILE), {})
    return data if isinstance(data, dict) else {}


@st.cache_resource(show_spinner=False)
def load_search_stack(docs_stamp, index_stamp):
    docs = load_documents(docs_stamp)
    index = InvertedIndex()
    if not index.load():
        return None, None, None, None
    index.documents = docs
    preprocessor = TextPreprocessor()
    vsm = VectorSpaceModel(index)
    bm25 = BM25Model(index)
    return index, preprocessor, vsm, bm25


@st.cache_data(show_spinner=False)
def load_feedback(feedback_stamp):
    return read_json(FEEDBACK_FILE, {})


@st.cache_data(show_spinner=False)
def load_eval_queries(eval_stamp):
    data = read_json(Path(EVAL_QUERIES_FILE), [])
    return data if isinstance(data, list) else []


@st.cache_data(show_spinner=False)
def load_video_metadata(video_stamp):
    data = read_json(VIDEO_METADATA_FILE, {})
    return data if isinstance(data, dict) else {}


@st.cache_data(show_spinner=False)
def load_image_metadata(image_stamp):
    data = read_json(IMAGE_METADATA_FILE, {})
    return data if isinstance(data, dict) else {}


@st.cache_resource(show_spinner=True)
def load_multimodal_retriever():
    from multimodal_retrieval import CLIPImageRetriever

    return CLIPImageRetriever()


def format_size(num_bytes: int) -> str:
    if num_bytes is None:
        return "-"
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def render_result(item, rank):
    title = item.get("title", "无标题")
    score = item.get("score", 0)
    date = item.get("date", "")
    url = item.get("url", "")
    snippet = item.get("snippet", "")
    st.markdown(
        f"""
        <div class="result-box">
            <div class="muted">#{rank} · <span class="score">相关度 {score:.6f}</span> · {date or "日期缺失"}</div>
            <h4 style="margin: 0.35rem 0 0.25rem 0;">{title}</h4>
            <div style="line-height: 1.65;">{snippet}</div>
            <div class="muted" style="margin-top: 0.45rem;">{url}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_overview():
    docs = load_documents(file_stamp(Path(DOCUMENTS_FILE)))
    index_raw = load_index_raw(file_stamp(Path(INDEX_FILE)))
    feedback = load_feedback(file_stamp(FEEDBACK_FILE))
    eval_queries = load_eval_queries(file_stamp(Path(EVAL_QUERIES_FILE)))
    video_meta = load_video_metadata(file_stamp(VIDEO_METADATA_FILE))
    image_meta = load_image_metadata(file_stamp(IMAGE_METADATA_FILE))

    unique_ids = len({doc.get("id") for doc in docs})
    docs_with_chinese = sum(
        1 for doc in docs if any("\u4e00" <= ch <= "\u9fff" for ch in (doc.get("title", "") + doc.get("text", "")))
    )
    feedback_stats = feedback.get("stats", {}) if isinstance(feedback, dict) else {}

    st.header("系统状态")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("文档数", len(docs))
    c2.metric("唯一文档 ID", unique_ids)
    c3.metric("中文文档", docs_with_chinese)
    c4.metric("索引词项数", len(index_raw.get("index", {})))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("索引文档数", index_raw.get("doc_count", 0))
    c6.metric("人工评分数", feedback_stats.get("total_ratings", 0))
    c7.metric("评价查询数", len(eval_queries))
    c8.metric("视频索引数", len(video_meta))

    st.subheader("关键文件")
    file_rows = [
        ("文档库", Path(DOCUMENTS_FILE)),
        ("倒排索引", Path(INDEX_FILE)),
        ("人工评价", FEEDBACK_FILE),
        ("评价查询", Path(EVAL_QUERIES_FILE)),
        ("视频索引", VIDEO_INDEX_FILE),
        ("图片索引", IMAGE_INDEX_FILE),
    ]
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "项目": name,
                    "是否存在": path.exists(),
                    "大小": format_size(path.stat().st_size) if path.exists() else "-",
                    "路径": str(path),
                }
                for name, path in file_rows
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("文档样例")
    if docs:
        sample = docs[0]
        st.write(
            {
                "title": sample.get("title", ""),
                "date": sample.get("date", ""),
                "url": sample.get("url", ""),
                "tokens_count": len(sample.get("tokens", [])),
            }
        )

    if docs:
        st.subheader("数据源分布")
        source_counts = Counter(doc.get("source") or "未标注来源" for doc in docs)
        source_rows = [{"来源": source, "文档数": count} for source, count in source_counts.most_common()]
        st.dataframe(pd.DataFrame(source_rows), use_container_width=True, hide_index=True)


def page_search():
    st.header("文本检索")
    index, preprocessor, vsm, bm25 = load_search_stack(
        file_stamp(Path(DOCUMENTS_FILE)),
        file_stamp(Path(INDEX_FILE)),
    )
    if not all([index, preprocessor, vsm, bm25]):
        st.error("未能加载倒排索引。请先在命令行执行 main.py 的 [2] 构建索引。")
        return

    left, right = st.columns([3, 1])
    with left:
        query = st.text_input("输入自然语言查询", value="人工智能 大模型")
    with right:
        algorithm = st.selectbox("检索算法", ["VSM / TF-IDF", "BM25"])
        top_k = st.slider("返回数量", 5, 20, 10)

    if st.button("检索", type="primary", use_container_width=True):
        tokens = preprocessor.segment(query)
        st.caption("分词结果：" + (" / ".join(tokens) if tokens else "无有效词项"))
        model = vsm if algorithm.startswith("VSM") else bm25
        results = model.search(tokens, top_k=top_k)

        if not results:
            st.warning("没有找到相关结果。")
            return

        st.subheader(f"检索结果（{algorithm}）")
        for i, item in enumerate(results, 1):
            render_result(item, i)


def page_multimedia():
    st.header("多媒体检索")
    video_meta = load_video_metadata(file_stamp(VIDEO_METADATA_FILE))
    image_meta = load_image_metadata(file_stamp(IMAGE_METADATA_FILE))

    c1, c2 = st.columns(2)
    c1.metric("已索引视频", len(video_meta))
    c2.metric("已索引图片", len(image_meta))

    query = st.text_input("输入多媒体语义查询", value="猫")
    top_k = st.slider("返回数量", 1, 5, 3, key="media_top_k")

    if st.button("检索视频", type="primary", use_container_width=True):
        if not video_meta:
            st.warning("没有可用的视频元数据。请先在命令行菜单 [6] 中索引视频。")
            return
        try:
            retriever = load_multimodal_retriever()
            results = retriever.search_videos(query, top_k=top_k)
        except Exception as exc:
            st.error(f"多媒体模型加载或检索失败：{exc}")
            return

        if not results:
            st.warning("没有找到视频结果。")
            return

        for item in results:
            path = item.get("path", "")
            st.markdown(
                f"**{item.get('video_id', 'unknown')}** · 相似度 `{item.get('score', 0):.6f}` · {item.get('size_mb', 0)} MB"
            )
            if path and Path(path).exists():
                st.video(path)
            else:
                st.caption(path or "视频路径缺失")

    with st.expander("已索引视频列表", expanded=False):
        rows = []
        for name, meta in video_meta.items():
            rows.append(
                {
                    "文件": name,
                    "大小": format_size(meta.get("size_bytes", 0)),
                    "路径": meta.get("path", ""),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def page_evaluation():
    st.header("人工评价")
    feedback = load_feedback(file_stamp(FEEDBACK_FILE))
    eval_queries = load_eval_queries(file_stamp(Path(EVAL_QUERIES_FILE)))
    stats = feedback.get("stats", {}) if isinstance(feedback, dict) else {}

    c1, c2, c3 = st.columns(3)
    c1.metric("总评分数", stats.get("total_ratings", 0))
    c2.metric("已评价文档数", stats.get("rated_docs", 0))
    c3.metric("最后更新时间", (stats.get("last_updated") or "-")[:19])

    rows = []
    for item in eval_queries:
        rows.append(
            {
                "查询": item.get("query", ""),
                "说明": item.get("description", ""),
                "相关文档数": len(item.get("relevant_docs", [])),
                "相关文档 ID": ", ".join(map(str, item.get("relevant_docs", []))),
            }
        )
    if rows:
        st.subheader("评价查询覆盖情况")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("尚未找到评价查询文件。")

    query_feedback = feedback.get("query_feedback", {}) if isinstance(feedback, dict) else {}
    if query_feedback:
        st.subheader("查询反馈明细")
        detail_rows = []
        for query, data in query_feedback.items():
            detail_rows.append(
                {
                    "查询": query,
                    "相关": len(data.get("relevant_docs", [])),
                    "不相关": len(data.get("irrelevant_docs", [])),
                    "评分次数": data.get("count", 0),
                }
            )
        st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)


def page_charts():
    st.header("可视化图表")
    chart_files = sorted(CHART_DIR.glob("*.png"))
    if not chart_files:
        st.warning("未找到图表。请先在命令行菜单 [9] 生成可视化图表。")
        return

    for chart in chart_files:
        st.subheader(chart.stem.replace("_", " ").title())
        st.image(str(chart), use_container_width=True)


def main():
    st.sidebar.title("中文信息检索系统")
    page = st.sidebar.radio(
        "功能导航",
        ["系统状态", "文本检索", "多媒体检索", "人工评价", "可视化图表"],
    )
    st.sidebar.markdown("---")
    if st.sidebar.button("刷新数据缓存", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    st.sidebar.caption("VSM / BM25 / 人工评价 / 文本到视频检索")

    if page == "系统状态":
        page_overview()
    elif page == "文本检索":
        page_search()
    elif page == "多媒体检索":
        page_multimedia()
    elif page == "人工评价":
        page_evaluation()
    elif page == "可视化图表":
        page_charts()


if __name__ == "__main__":
    main()
