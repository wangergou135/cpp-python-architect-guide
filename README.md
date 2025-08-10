# C++/Python 架构师技术指南

一个全面的C++和Python架构师培养指南，包含核心技术、架构设计、最佳实践和学习路径。

## 目录结构

```
.
├── docs/                    # 详细文档
│   ├── cpp/                # C++相关文档
│   ├── python/             # Python相关文档
│   ├── architecture/       # 架构设计文档
│   └── guidelines/         # 规范和指南
├── examples/               # 示例代码
│   ├── cpp/                # C++示例
│   ├── python/             # Python示例
│   └── projects/           # 完整项目示例
└── tests/                  # 测试用例
```

## 快速开始

1. 克隆仓库:
```bash
git clone https://github.com/wangergou135/cpp-python-architect-guide.git
cd cpp-python-architect-guide
```

2. 查看文档:
- 核心技能路径: [docs/guidelines/learning_path.md](docs/guidelines/learning_path.md)
- C++进阶指南: [docs/cpp/README.md](docs/cpp/README.md)
- Python进阶指南: [docs/python/README.md](docs/python/README.md)
- 架构设计指南: [docs/architecture/README.md](docs/architecture/README.md)

3. 运行示例:
```bash
# C++示例
cd examples/cpp
mkdir build && cd build
cmake ..
make

# Python示例
cd examples/python
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python example.py
```

## 学习路径

1. [核心技能](docs/guidelines/core_skills.md)
   - C++核心技术
   - Python高级特性
   - 工程实践能力

2. [架构设计](docs/architecture/README.md)
   - 系统架构
   - 代码架构
   - 技术选型

3. [工程实践](docs/guidelines/engineering_practices.md)
   - 开发流程
   - 质量保证
   - 性能优化

4. [团队管理](docs/guidelines/team_management.md)
   - 技术决策
   - 团队建设
   - 项目管理

## 贡献指南

欢迎贡献代码和文档！请查看[贡献指南](CONTRIBUTING.md)。

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件