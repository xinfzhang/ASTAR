# ASTAR: 医学报告自动模板归纳系统

ASTAR 是一个用于从医学报告中自动归纳结构化模板的三阶段流水线。给定一组自由文本医学报告，ASTAR 能够归纳出捕捉关键临床概念的结构化模板，然后使用该模板对新报告进行结构化处理。

## 特性

- **自动模板归纳**：从数据中学习报告结构，无需手动设计模板
- **三阶段流水线**：槽位归纳、模板组装和精炼
- **全面评估指标**：模板质量、信息保真度和诊断保真度
- **模块化设计**：易于扩展和定制，适用于不同医学领域

## 安装

```bash
# 克隆仓库
git clone https://github.com/your-org/astar.git
cd astar

# 以开发模式安装
pip install -e .

# 安装评估依赖
pip install -e ".[eval]"

# 安装所有依赖
pip install -e ".[all]"
```

## 快速开始

### 1. 配置

复制示例环境文件并添加API密钥：

```bash
cp .env.example .env
# 编辑 .env 并设置 ASTAR_API_KEY=your_api_key_here
```

编辑 `config.yaml` 配置：
- LLM 端点和模型
- 数据路径
- 流水线参数

### 2. 模板归纳

```bash
# 运行完整的模板归纳流水线
astar-induction --config config.yaml

# 或运行特定步骤
astar-induction --steps 0 1 2  # 仅运行阶段 I（槽位归纳）
astar-induction --start 3      # 从阶段 II 开始
```

### 3. 报告结构化

```bash
# 使用归纳的模板结构化报告
python -m astar.structuring.structure \
    --data data/test.csv \
    --template output/exp_001/template_final.json \
    --output output/structured_output.json
```

### 4. 评估

```bash
# 运行所有评估指标
astar-eval \
    --template output/exp_001/template_final.json \
    --structured output/structured_output.json \
    --gt data/test.csv

# 跳过较慢的 BERTScore 评估
astar-eval ... --skip_bert
```

## 流水线概述

### 阶段 I：槽位归纳（步骤 0-2）

1. **步骤 0 - Span聚类**：从报告中抽取临床概念片段并聚类
2. **步骤 1 - 槽位归纳**：细分聚类以归纳规范化槽位 (k, τ, Ω)
3. **步骤 2 - 槽位合并**：按键名全局合并槽位

### 阶段 II：模板组装（步骤 3-5）

4. **步骤 3 - 骨架构建**：从样本报告归纳高层结构
5. **步骤 4 - 槽位分配**：将槽位挂载到骨架上（两轮过程）
6. **步骤 5 - 树组装**：合并骨架和槽位为完整模板

### 阶段 III：模板精炼（步骤 6）

7. **步骤 6 - 精炼**：节点内重组和全局协调

## 输出文件

运行流水线后，输出目录包含：

```
output/exp_001/
├── spans_with_cluster.jsonl     # 步骤 0：抽取的 spans
├── induced_slots_step1.jsonl    # 步骤 1：归纳的槽位
├── slot_catalog.jsonl           # 步骤 2：合并的槽位
├── template_skeleton.json       # 步骤 3：结构骨架
├── template_skeleton_updated.json  # 步骤 4：更新的骨架
├── slot_assignments.jsonl       # 步骤 4：槽位分配
├── template_with_slots.json     # 步骤 5：带槽位的模板
└── template_final.json          # 步骤 6：最终精炼模板
```

## 评估指标

- **模板质量**：Cov_case（病例级覆盖率）、Cov_key（键级覆盖率）
- **信息保真度**：ROUGE-1/2/L 分数
- **文本相似度**：chrF、chrF++、BERTScore
- **诊断保真度**：PDA（主诊断准确度）、KFP（关键发现保留率）、CA（临床可操作性）

## 项目结构

```
astar/
├── astar/
│   ├── core/           # 共享基础设施
│   │   ├── config.py   # 统一配置
│   │   ├── llm.py      # LLM API 客户端
│   │   └── utils.py    # 文件 I/O 工具
│   ├── induction/      # 模板归纳流水线
│   │   ├── step0_span_clustering.py
│   │   ├── step1_slot_induction.py
│   │   ├── step2_slot_merging.py
│   │   ├── step3_skeleton.py
│   │   ├── step4_slot_assignment.py
│   │   ├── step5_tree_assembly.py
│   │   ├── step6_refinement.py
│   │   └── pipeline.py
│   ├── structuring/    # 报告结构化
│   │   ├── structure.py
│   │   └── template_utils.py
│   └── evaluation/     # 评估套件
│       ├── run_eval.py
│       ├── reconstruct.py
│       ├── template_quality.py
│       ├── info_fidelity.py
│       ├── text_similarity.py
│       └── diag_fidelity.py
├── examples/
│   ├── example_report.csv
│   └── example_template.json
├── config.yaml
├── pyproject.toml
└── README.md
```

## 配置参考

查看 `config.yaml` 了解所有可用选项。关键设置：

```yaml
llm:
  base_url: "https://api.openai.com/v1"  # OpenAI 兼容端点
  model: "gpt-4o-mini"
  embedding_model: "text-embedding-3-small"

data:
  train_csv: "./data/train.csv"
  description_column: "description"  # 影像描述列名
  diagnosis_column: "diagnosis"      # 影像诊断列名
```

## Python API

```python
from astar.core.config import get_config, load_config
from astar.induction import run_pipeline
from astar.structuring.structure import structure_reports
from astar.evaluation import run_all_evaluations

# 加载自定义配置
load_config("path/to/config.yaml")

# 运行模板归纳
output_dir = run_pipeline()

# 结构化报告
results = structure_reports(
    data_path="data/test.csv",
    template_path=f"{output_dir}/template_final.json",
)

# 评估
metrics = run_all_evaluations(
    template_path=f"{output_dir}/template_final.json",
    structured_path="output/structured_output.json",
    gt_path="data/test.csv",
)
```

## 系统要求

- Python >= 3.9
- 完整依赖列表见 `pyproject.toml`

## 许可证

MIT License

## 引用

如果您在研究中使用了 ASTAR，请引用：

```bibtex
@article{astar2024,
  title={ASTAR: Automatic Template Induction for Medical Report Structuring},
  author={...},
  journal={...},
  year={2026}
}
```
