import requests
from bs4 import BeautifulSoup
import time
import random
import re
import json
import os
from datetime import datetime
from config import (
    CRAWL_MIN_DOCS, CRAWL_TIMEOUT, CRAWL_DELAY,
    CRAWL_MAX_DOCS, DOCUMENTS_FILE
)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}


def get_random_headers():
    return {**HEADERS, "User-Agent": random.choice(USER_AGENTS)}


def fetch_url(url, timeout=CRAWL_TIMEOUT, retries=3):
    for attempt in range(retries):
        try:
            resp = requests.get(
                url, headers=get_random_headers(),
                timeout=timeout, allow_redirects=True
            )
            resp.raise_for_status()
            if "text/html" not in resp.headers.get("Content-Type", ""):
                return None
            resp.encoding = resp.apparent_encoding or "utf-8"
            return resp
        except Exception:
            if attempt < retries - 1:
                time.sleep(1 * (attempt + 1))
            else:
                return None


def extract_text_from_html(html):
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                     "noscript", "iframe", "form", "button", "input"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r'\s+', ' ', text)
    return text


class WebCrawler:
    """Multi-source web crawler for Chinese news/articles."""

    def __init__(self):
        self.documents = []
        self.seen_urls = set()
        if os.path.exists(DOCUMENTS_FILE):
            try:
                with open(DOCUMENTS_FILE, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                self.documents = existing
                self.seen_urls = {d["url"] for d in existing}
                print(f"[Crawler] Loaded {len(self.documents)} existing documents.")
            except Exception:
                pass

    def _save(self):
        os.makedirs(os.path.dirname(DOCUMENTS_FILE), exist_ok=True)
        with open(DOCUMENTS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)

    def _add_document(self, url, title, text, date_str=""):
        if url in self.seen_urls:
            return False
        if len(text.strip()) < 100:
            return False
        doc_id = len(self.documents)
        doc = {
            "id": doc_id,
            "url": url,
            "title": title.strip() if title else "无标题",
            "text": text.strip(),
            "date": date_str,
            "crawled_at": datetime.now().isoformat(),
        }
        self.documents.append(doc)
        self.seen_urls.add(url)
        return True

    # ---- Source 1: People.cn (人民网科技频道) ----
    def crawl_people_tech(self, max_pages=10):
        print("[Crawler] Crawling people.cn tech channel...")
        base = "https://scitech.people.com.cn"
        list_urls = [f"{base}/GB/index{i}.html" for i in range(1, max_pages + 1)]

        article_urls = []
        for list_url in list_urls:
            resp = fetch_url(list_url)
            if not resp:
                continue
            soup = BeautifulSoup(resp.text, "lxml")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "/GB/" not in href and "/n1/" not in href:
                    continue
                if not href.startswith("http"):
                    if href.startswith("/"):
                        href = "http://scitech.people.com.cn" + href
                    else:
                        href = base + "/" + href
                if href not in self.seen_urls:
                    article_urls.append(href)
            time.sleep(CRAWL_DELAY)

        article_urls = list(set(article_urls))
        print(f"[Crawler] Found {len(article_urls)} candidate article URLs.")
        random.shuffle(article_urls)

        for url in article_urls:
            if len(self.documents) >= CRAWL_MAX_DOCS:
                break
            resp = fetch_url(url)
            if not resp:
                continue
            soup = BeautifulSoup(resp.text, "lxml")

            title_tag = soup.find("h1") or soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else ""

            date_str = ""
            date_patterns = [
                r'(\d{4}年\d{1,2}月\d{1,2}日)',
                r'(\d{4}-\d{2}-\d{2})',
                r'source\s*:\s*(\d{4}-\d{2}-\d{2})',
            ]
            page_text = resp.text
            for pat in date_patterns:
                m = re.search(pat, page_text, re.IGNORECASE)
                if m:
                    date_str = m.group(1)
                    break

            content_div = soup.find("div", class_=re.compile(r"(article|content|text|detail|body)", re.I))
            if content_div:
                text = extract_text_from_html(str(content_div))
            else:
                text = extract_text_from_html(resp.text)

            if self._add_document(url, title, text, date_str):
                print(f"  [{len(self.documents)}] {title[:40]}...")

            time.sleep(random.uniform(0.3, 1.0))

    # ---- Source 2: Xinhua News (新华网科技) ----
    def crawl_xinhua_tech(self, max_pages=10):
        print("[Crawler] Crawling xinhuanet.com tech channel...")
        base = "http://www.news.cn/tech"
        list_urls = [f"{base}/index.html" for _ in range(1)]

        for pg in range(1, max_pages + 1):
            list_url = f"{base}/{pg}.html" if pg > 1 else f"{base}/index.html"
            resp = fetch_url(list_url)
            if not resp:
                continue
            soup = BeautifulSoup(resp.text, "lxml")
            article_urls = []
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "/202" not in href:
                    continue
                if not href.startswith("http"):
                    href = "http://www.news.cn" + href if href.startswith("/") else "http://www.news.cn/" + href
                if href not in self.seen_urls:
                    article_urls.append(href)

            print(f"[Crawler] Page {pg}: {len(article_urls)} candidate URLs.")
            for url in article_urls:
                if len(self.documents) >= CRAWL_MAX_DOCS:
                    break
                resp = fetch_url(url)
                if not resp:
                    continue
                soup = BeautifulSoup(resp.text, "lxml")

                title_tag = soup.find("h1") or soup.find("title")
                title = title_tag.get_text(strip=True) if title_tag else ""

                date_str = ""
                m = re.search(r'(\d{4}-\d{2}-\d{2})', resp.text)
                if m:
                    date_str = m.group(1)

                content_div = soup.find("div", id="detail-content") or \
                              soup.find("div", class_=re.compile(r"(article|content)", re.I))
                if content_div:
                    text = extract_text_from_html(str(content_div))
                else:
                    text = extract_text_from_html(resp.text)

                if self._add_document(url, title, text, date_str):
                    print(f"  [{len(self.documents)}] {title[:40]}...")

                time.sleep(random.uniform(0.3, 1.0))
            time.sleep(CRAWL_DELAY)

    # ---- Source 3: China Daily Tech (中国日报科技) ----
    def crawl_chinadaily_tech(self, max_pages=10):
        print("[Crawler] Crawling chinadaily.com.cn tech channel...")
        base = "https://www.chinadaily.com.cn"
        categories = ["tech", "science"]
        article_urls = []

        for cat in categories:
            list_url = f"{base}/{cat}/"
            resp = fetch_url(list_url)
            if not resp:
                continue
            soup = BeautifulSoup(resp.text, "lxml")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "/a/" not in href or "/photo/" in href:
                    continue
                if not href.startswith("http"):
                    href = base + href if href.startswith("/") else base + "/" + href
                if href not in self.seen_urls:
                    article_urls.append(href)
            time.sleep(CRAWL_DELAY)

        article_urls = list(set(article_urls))
        print(f"[Crawler] Found {len(article_urls)} candidate article URLs.")
        random.shuffle(article_urls)

        for url in article_urls:
            if len(self.documents) >= CRAWL_MAX_DOCS:
                break
            resp = fetch_url(url)
            if not resp:
                continue
            soup = BeautifulSoup(resp.text, "lxml")

            title_tag = soup.find("h1") or soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else ""

            date_str = ""
            m = re.search(r'(\d{4}-\d{2}-\d{2}\s*\d{2}:\d{2})', resp.text)
            if not m:
                m = re.search(r'(\d{4}-\d{2}-\d{2})', resp.text)
            if m:
                date_str = m.group(1)

            content_div = soup.find("div", id="Content") or \
                          soup.find("div", class_=re.compile(r"(article|content|text)", re.I))
            if content_div:
                text = extract_text_from_html(str(content_div))
            else:
                text = extract_text_from_html(resp.text)

            if self._add_document(url, title, text, date_str):
                print(f"  [{len(self.documents)}] {title[:40]}...")

            time.sleep(random.uniform(0.3, 1.0))

    # ---- Source 4: Sample documents as fallback ----
    def add_sample_documents(self):
        """Add curated sample documents if web crawling insufficient."""
        print("[Crawler] Generating sample documents as fallback...")
        samples = [
            ("人工智能技术发展趋势与未来展望",
             "人工智能技术近年来取得了突飞猛进的发展，从深度学习到大语言模型，AI正在深刻改变各行各业。"
             "ChatGPT等大模型的出现标志着人工智能进入了新的发展阶段，自然语言处理、计算机视觉、"
             "语音识别等技术不断突破。未来，AI将在医疗诊断、自动驾驶、智能制造等领域发挥更大作用。"
             "同时，AI伦理和安全问题也引起了广泛关注，如何确保AI技术的健康发展成为重要课题。",
             "2024-01-15", "https://example.com/ai-trends"),
            ("5G通信技术推动数字化转型",
             "5G作为新一代移动通信技术，以其高速率、低时延、大连接的特性，正在加速各行各业的数字化转型。"
             "工业互联网、智慧城市、远程医疗等应用场景不断丰富。中国5G基站数量已超过300万个，"
             "5G终端用户突破7亿。5G与AI、物联网的融合将催生更多创新应用，推动数字经济发展。",
             "2024-02-20", "https://example.com/5g-digital"),
            ("新能源汽车产业蓬勃发展",
             "新能源汽车市场持续火热，2023年中国新能源汽车产销量均突破900万辆。比亚迪、特斯拉、"
             "蔚来、小鹏等品牌竞争激烈。电池技术不断进步，固态电池、钠离子电池等新型电池技术"
             "有望突破能量密度瓶颈。充电基础设施也在加速建设，换电模式逐渐普及。"
             "新能源汽车正从电动化向智能化、网联化方向发展。",
             "2024-03-10", "https://example.com/ev-industry"),
            ("芯片半导体产业自主创新之路",
             "芯片是现代信息技术的基石，中国半导体产业正加速自主创新。从设计到制造，"
             "国产芯片在多个领域取得突破。华为麒麟芯片、长江存储3D NAND闪存等产品"
             "彰显了中国芯片产业的进步。面对国际竞争和封锁，中国加大研发投入，"
             "培养芯片人才，建设完整的半导体产业链。光刻机、EDA工具等关键环节也在逐步突破。",
             "2024-01-28", "https://example.com/chip-innovation"),
            ("大数据时代的数据安全与隐私保护",
             "随着大数据技术的广泛应用，数据安全和隐私保护成为热点问题。个人信息保护法"
             "的实施为数据保护提供了法律框架。企业需要建立完善的数据治理体系，确保数据"
             "采集、存储、处理、传输各环节的安全。隐私计算、联邦学习、同态加密等新技术"
             "为数据安全使用提供了新思路。数据要素市场化配置也在有序推进。",
             "2024-02-05", "https://example.com/data-security"),
            ("云计算技术赋能企业数字化转型",
             "云计算已成为企业数字化转型的核心基础设施。阿里云、腾讯云、华为云等国内云服务商"
             "不断完善产品体系。混合云、多云架构成为主流选择。Serverless、容器技术降低了"
             "应用部署门槛。边缘计算与云计算的协同，使实时数据处理成为可能。"
             "云原生技术推动了应用架构的现代化，微服务、DevOps实践助力业务敏捷创新。",
             "2024-03-15", "https://example.com/cloud-computing"),
            ("中国航天探索取得新突破",
             "中国航天事业不断取得新成就，嫦娥探月工程、天问火星探测、空间站建设等"
             "重大项目顺利推进。新一代载人火箭研制取得进展，重复使用火箭技术开展验证。"
             "商业航天蓬勃发展，民营火箭企业相继成功发射。卫星互联网建设加速，"
             "遥感、导航、通信卫星体系不断完善。深空探测计划稳步推进。",
             "2024-04-01", "https://example.com/space-exploration"),
            ("区块链技术赋能数字经济",
             "区块链技术凭借去中心化、不可篡改等特性，在金融、供应链、版权保护等领域"
             "得到广泛应用。央行数字货币（数字人民币）试点范围不断扩大。DeFi、NFT等"
             "Web3应用快速发展。区块链与物联网、AI的技术融合，为数据可信流通提供了"
             "坚实基础。联盟链在政务、司法等领域的应用逐渐成熟。",
             "2024-02-15", "https://example.com/blockchain-tech"),
            ("量子计算研究进展与展望",
             "量子计算作为前沿技术，正在从实验室走向实用化。中国在超导量子计算、光量子"
             "计算等路线上均有布局。'九章'光量子计算机和'祖冲之'超导量子计算机展示了"
             "量子计算优越性。量子纠错、量子算法等关键技术不断突破。量子计算与AI的"
             "结合有望解决经典计算机难以处理的复杂问题，在药物研发、材料科学等领域发挥重要作用。",
             "2024-03-25", "https://example.com/quantum-computing"),
            ("元宇宙概念与实践探索",
             "元宇宙作为下一代互联网的愿景，融合了VR/AR、数字孪生、区块链等技术。"
             "Meta、字节跳动、腾讯等科技巨头纷纷布局。工业元宇宙在数字孪生工厂、虚拟仿真"
             "等领域已有了实际应用。教育元宇宙为在线教育带来沉浸式体验。"
             "然而，元宇宙的发展也面临技术瓶颈、内容生态、监管政策等挑战。",
             "2024-01-10", "https://example.com/metaverse-explore"),
            ("生物医药创新推动健康中国建设",
             "生物医药产业正迎来快速发展期，基因编辑、细胞治疗、mRNA疫苗等新技术不断涌现。"
             "国产创新药出海加速，PD-1抗体、CAR-T细胞治疗等产品获得国际认可。"
             "AI制药提高了新药研发效率。精准医疗基于基因组学和大数据分析，"
             "为个体化诊疗提供了可能。医疗器械国产替代也在稳步推进。",
             "2024-02-28", "https://example.com/biomedicine"),
            ("智能制造引领工业4.0变革",
             "智能制造是工业4.0的核心，融合了物联网、AI、机器人等先进技术。"
             "数字孪生技术在工厂规划、生产过程优化中发挥重要作用。工业机器人密度"
             "持续提升，协作机器人实现人机协同作业。工业互联网平台连接设备、汇聚数据，"
             "赋能制造企业提质增效。灯塔工厂引领全球智能制造最佳实践。",
             "2024-03-05", "https://example.com/smart-manufacturing"),
            ("网络安全防护体系建设",
             "随着数字化转型深入，网络安全威胁日益严峻。勒索软件、APT攻击、数据泄露等"
             "安全事件频发。零信任安全架构成为企业安全建设的新范式。AI安全技术用于"
             "威胁检测和响应。关键信息基础设施安全保护条例实施，提升了重点领域的安全防护水平。"
             "网络安全人才培养和队伍建设也在加速推进。",
             "2024-01-20", "https://example.com/cyber-security"),
            ("绿色能源转型与碳中和路径",
             "实现碳达峰碳中和目标，能源转型是关键。光伏、风电等可再生能源装机快速增长，"
             "发电成本持续下降。新型储能技术发展迅猛，锂离子电池、钠离子电池、液流电池"
             "等路线百花齐放。氢能作为清洁能源受到广泛关注，绿氢制备技术不断进步。"
             "碳交易市场建设完善，为企业减排提供市场化机制。",
             "2024-04-10", "https://example.com/green-energy"),
            ("自动驾驶技术发展现状与挑战",
             "自动驾驶技术持续演进，L2级辅助驾驶已大规模商用，L4级Robotaxi在多个城市"
             "开展示范运营。感知、决策、控制三大技术模块不断优化。激光雷达成本快速下降，"
             "4D毫米波雷达提升了感知能力。车路协同为自动驾驶提供了新的技术路径。"
             "法规标准和责任认定问题仍在持续探索中。",
             "2024-02-18", "https://example.com/autonomous-driving"),
        ]

        templates = [
            {"category": "人工智能", "subtopics": ["深度学习", "神经网络", "大语言模型", "强化学习", "迁移学习",
                         "GAN生成对抗网络", "注意力机制", "Transformer架构", "预训练模型", "提示工程"],
             "prefix": "人工智能领域的", "suffix": "技术取得了突破性进展"},
            {"category": "互联网", "subtopics": ["Web3.0", "微服务", "云原生", "边缘计算", "IoT物联网",
                         "API网关", "负载均衡", "CDN加速", "DNS解析", "HTTP协议优化"],
             "prefix": "互联网技术的", "suffix": "正在改变网络架构设计"},
            {"category": "数据库", "subtopics": ["分布式数据库", "时序数据库", "图数据库", "向量数据库",
                         "NewSQL", "NoSQL", "数据湖", "流计算", "OLAP分析", "HTAP混合负载"],
             "prefix": "数据库领域的", "suffix": "提升了数据处理能力"},
            {"category": "编程语言", "subtopics": ["Rust语言", "Go语言", "TypeScript", "Python生态",
                         "Kotlin多平台", "WebAssembly", "Julia科学计算", "Swift并发", "Dart语言", "Zig语言"],
             "prefix": "", "suffix": "在开发者社区中受到广泛关注"},
            {"category": "操作系统", "subtopics": ["鸿蒙OS", "Linux内核", "实时操作系统", "微内核架构",
                         "容器OS", "分布式OS", "RTOS物联网", "车载OS", "嵌入式Linux", "eBPF技术"],
             "prefix": "", "suffix": "为智能设备提供了坚实的基础"},
            {"category": "机器人", "subtopics": ["人形机器人", "四足机器人", "协作机器人", "手术机器人",
                         "扫地机器人", "无人机配送", "水下机器人", "蛇形机器人", "软体机器人", "外骨骼"],
             "prefix": "机器人技术中的", "suffix": "突破了传统机械的局限"},
            {"category": "金融科技", "subtopics": ["移动支付", "智能投顾", "量化交易", "保险科技",
                         "供应链金融", "开放银行", "监管沙盒", "数字货币", "跨境支付", "征信系统"],
             "prefix": "金融科技领域的", "suffix": "革新了传统金融服务模式"},
            {"category": "教育科技", "subtopics": ["在线教育", "自适应学习", "知识图谱", "智慧课堂",
                         "虚拟实验室", "AI批改", "个性化推荐", "MOOC课程", "微学习", "游戏化学习"],
             "prefix": "教育科技领域的", "suffix": "为个性化学习提供了新可能"},
            {"category": "农业科技", "subtopics": ["精准农业", "智慧灌溉", "无人机植保", "农业大数据",
                         "垂直农场", "基因育种", "土壤传感器", "智能温室", "畜牧物联网", "农产品溯源"],
             "prefix": "农业科技中的", "suffix": "助力现代农业提质增效"},
            {"category": "智慧城市", "subtopics": ["城市大脑", "智慧交通", "智慧能源", "智慧安防",
                         "数字孪生城市", "智慧政务", "智慧医疗", "智慧社区", "智慧环保", "智能灯杆"],
             "prefix": "智慧城市建设中的", "suffix": "大幅提升了城市治理水平"},
        ]

        months = [f"2024-{m:02d}" for m in range(1, 13)]
        days = [f"{d:02d}" for d in range(1, 29)]

        for i, (title, text, date, url) in enumerate(samples):
            if len(self.documents) >= CRAWL_MAX_DOCS:
                break
            self._add_document(url, title, text, date)

        for i, tmpl in enumerate(templates):
            for j, subtopic in enumerate(tmpl["subtopics"]):
                if len(self.documents) >= CRAWL_MAX_DOCS:
                    break
                category = tmpl["category"]
                title = f"{tmpl['prefix']}{subtopic}{tmpl['suffix']}"
                date_str = f"{random.choice(months)}-{random.choice(days)}"
                url = f"https://example.com/{category}/{subtopic}-{i*10+j}"

                text_parts = [
                    f"{subtopic}是{category}领域的重要研究方向。",
                    f"近年来，{subtopic}技术不断发展成熟，在多个行业得到了广泛应用。",
                    f"研究表明，{subtopic}可以显著提升系统性能和用户体验。",
                    f"专家指出，{subtopic}的未来发展需要产学研各方的通力协作。",
                    f"随着技术的进步，{subtopic}的应用场景将进一步拓展。",
                    f"业内领先企业纷纷布局{subtopic}，投入大量研发资源。",
                    f"相关数据显示，{subtopic}市场规模持续增长，前景广阔。",
                    f"与此同时，{subtopic}也面临技术挑战和标准制定等问题需要解决。",
                    f"国内研究团队在{subtopic}方面取得了一系列重要成果。",
                    f"未来{subtopic}将与AI、大数据等技术深度融合，创造更大价值。",
                ]
                text = " ".join(text_parts)
                self._add_document(url, title, text, date_str)

        self._save()
        print(f"[Crawler] Sample documents generated. Total: {len(self.documents)}")

    def crawl_all(self):
        print(f"[Crawler] Starting crawl. Target: {CRAWL_MIN_DOCS}-{CRAWL_MAX_DOCS} documents.")
        self.crawl_people_tech(max_pages=5)
        self._save()

        if len(self.documents) < CRAWL_MIN_DOCS:
            self.crawl_xinhua_tech(max_pages=5)
            self._save()

        if len(self.documents) < CRAWL_MIN_DOCS:
            self.crawl_chinadaily_tech(max_pages=5)
            self._save()

        if len(self.documents) < CRAWL_MIN_DOCS:
            self.add_sample_documents()
            self._save()

        print(f"[Crawler] Crawling complete. Total documents: {len(self.documents)}")
        self._save()
        return self.documents


if __name__ == "__main__":
    crawler = WebCrawler()
    docs = crawler.crawl_all()
    print(f"Done. {len(docs)} documents saved.")
