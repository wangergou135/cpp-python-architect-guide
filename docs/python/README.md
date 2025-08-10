# Python 开发指南 / Python Development Guide

*最后更新 / Last Updated: 2025-08-10 10:33:02 UTC*

## 简介 / Introduction

本指南提供了项目中Python开发的标准、最佳实践和技术细节。它涵盖了从基础配置到高级应用的各个方面。

This guide provides standards, best practices, and technical details for Python development in the project. It covers various aspects from basic configuration to advanced applications.

## 目录 / Contents

1. [环境配置 / Environment Setup](setup.md)
   - Python版本要求 / Python Version Requirements
   - 虚拟环境管理 / Virtual Environment Management
   - 依赖包管理 / Dependency Management
   - 开发工具配置 / Development Tools Setup

2. [编码规范 / Coding Standards](standards.md)
   - PEP 8规范 / PEP 8 Guidelines
   - 项目特定规范 / Project-Specific Standards
   - 命名约定 / Naming Conventions
   - 文档规范 / Documentation Standards

3. [C++集成 / C++ Integration](cpp_integration.md)
   - Python C API使用 / Python C API Usage
   - pybind11使用指南 / pybind11 Usage Guide
   - 性能优化技巧 / Performance Optimization Tips
   - 内存管理 / Memory Management

4. [测试指南 / Testing Guidelines](testing.md)
   - 单元测试 / Unit Testing
   - 集成测试 / Integration Testing
   - 性能测试 / Performance Testing
   - 测试覆盖率 / Test Coverage

5. [最佳实践 / Best Practices](best_practices.md)
   - 代码组织 / Code Organization
   - 错误处理 / Error Handling
   - 日志记录 / Logging
   - 性能优化 / Performance Optimization

## 开发环境设置 / Development Environment Setup

### Python版本 / Python Version
- 主要版本：Python 3.9+
- 推荐使用最新的稳定版本
- 确保与C++模块兼容

### 虚拟环境 / Virtual Environment
```bash
# 创建虚拟环境 / Create virtual environment
python -m venv venv

# 激活虚拟环境 / Activate virtual environment
# Windows
venv\Scripts\activate
# Unix/MacOS
source venv/bin/activate

# 安装依赖 / Install dependencies
pip install -r requirements.txt
```

### 依赖管理 / Dependency Management
- 使用requirements.txt管理依赖
- 定期更新依赖版本
- 检查依赖的安全性

## 编码标准 / Coding Standards

### 基本规范 / Basic Standards
- 遵循PEP 8规范
- 使用4个空格缩进
- 最大行长度为120字符
- 使用清晰的变量和函数命名

### 示例代码 / Code Examples
```python
def calculate_average(numbers: List[float]) -> float:
    """
    Calculate the average of a list of numbers.
    
    Args:
        numbers: List of numbers to average
        
    Returns:
        float: The average value
        
    Raises:
        ValueError: If the list is empty
    """
    if not numbers:
        raise ValueError("Cannot calculate average of empty list")
    return sum(numbers) / len(numbers)
```

## C++集成指南 / C++ Integration Guide

### pybind11使用 / pybind11 Usage
```python
import pybind11
import cpp_module

# 调用C++函数示例 / Example of calling C++ function
result = cpp_module.process_data(input_data)
```

### 性能优化 / Performance Optimization
- 使用NumPy进行数值计算
- 避免频繁的Python/C++切换
- 合理使用内存视图（memoryview）

## 测试规范 / Testing Standards

### 单元测试 / Unit Testing
```python
import unittest

class TestCalculations(unittest.TestCase):
    def test_average(self):
        self.assertEqual(calculate_average([1, 2, 3]), 2)
        with self.assertRaises(ValueError):
            calculate_average([])
```

### 覆盖率要求 / Coverage Requirements
- 单元测试覆盖率 > 80%
- 关键模块覆盖率 > 90%
- 定期进行覆盖率检查

## 错误处理 / Error Handling

### 异常处理规范 / Exception Handling Standards
```python
try:
    result = process_data(input_data)
except ValueError as e:
    logger.error(f"Invalid input: {e}")
    raise
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise RuntimeError("Processing failed") from e
```

## 日志规范 / Logging Standards

### 日志配置 / Logging Configuration
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

## 持续改进 / Continuous Improvement

本文档会持续更新以反映最新的开发实践和标准。欢迎团队成员提供反馈和建议。

This documentation will be continuously updated to reflect the latest development practices and standards. Team members are welcome to provide feedback and suggestions.

## 参考资料 / References

- [Python Official Documentation](https://docs.python.org/)
- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [pybind11 Documentation](https://pybind11.readthedocs.io/)