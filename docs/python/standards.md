# Python 编码规范 / Python Coding Standards

*最后更新 / Last Updated: 2025-08-10 10:35:26 UTC*

## PEP 8规范 / PEP 8 Guidelines

### 代码格式 / Code Formatting
- 使用4个空格缩进 / Use 4 spaces for indentation
- 每行最大长度120字符 / Maximum line length 120 characters
- 使用空格分隔运算符 / Use spaces around operators

### 命名约定 / Naming Conventions
- 类名使用驼峰命名法 / Class names use CamelCase
- 函数和变量使用小写下划线 / Functions and variables use lowercase_with_underscores
- 常量使用大写下划线 / Constants use UPPERCASE_WITH_UNDERSCORES

### 文档规范 / Documentation Standards
```python
def calculate_total(items: List[Dict[str, Any]]) -> float:
    """
    计算商品总价 / Calculate total price of items.

    Args:
        items: 商品列表，每个商品包含'price'和'quantity' /
               List of items, each containing 'price' and 'quantity'

    Returns:
        float: 总价格 / Total price

    Raises:
        ValueError: 当价格或数量为负数时 / When price or quantity is negative
    """
    pass
```

## 项目特定规范 / Project-Specific Standards

### 导入规范 / Import Standards
```python
# 标准库导入 / Standard library imports
import os
import sys

# 第三方库导入 / Third-party imports
import numpy as np
import pandas as pd

# 本地模块导入 / Local imports
from .utils import helper
from .core import main
```

### 注释规范 / Comment Standards
- 所有公共API必须有文档字符串 / All public APIs must have docstrings
- 复杂逻辑需要添加行内注释 / Complex logic needs inline comments
- 注释使用中英双语 / Comments should be bilingual

### 类型提示 / Type Hints
```python
from typing import List, Dict, Optional

def process_data(data: List[Dict[str, any]], 
                threshold: Optional[float] = None) -> Dict[str, float]:
    pass
```

## 代码审查清单 / Code Review Checklist

### 基础检查 / Basic Checks
- [ ] 代码符合PEP 8规范 / Code follows PEP 8
- [ ] 所有测试通过 / All tests pass
- [ ] 文档完整且准确 / Documentation is complete and accurate
- [ ] 无未使用的导入 / No unused imports
- [ ] 代码已格式化 / Code is formatted

### 进阶检查 / Advanced Checks
- [ ] 错误处理完善 / Error handling is comprehensive
- [ ] 性能考虑充分 / Performance considerations
- [ ] 安全性检查 / Security checks
- [ ] 代码可维护性 / Code maintainability

## 工具配置 / Tool Configuration

### Black配置 / Black Configuration
```toml
# pyproject.toml
[tool.black]
line-length = 120
target-version = ['py39']
include = '\.pyi?$'
```

### Flake8配置 / Flake8 Configuration
```ini
# setup.cfg
[flake8]
max-line-length = 120
extend-ignore = E203, W503
max-complexity = 10
```

### MyPy配置 / MyPy Configuration
```ini
# setup.cfg
[mypy]
python_version = 3.9
warn_return_any = True
warn_unused_configs = True
```

## 最佳实践示例 / Best Practice Examples

### 错误处理 / Error Handling
```python
def divide_numbers(a: float, b: float) -> float:
    """
    安全的除法操作 / Safe division operation
    """
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Arguments must be numbers")
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
```

### 单元测试 / Unit Testing
```python
import pytest

def test_divide_numbers():
    assert divide_numbers(6, 2) == 3
    with pytest.raises(ValueError):
        divide_numbers(1, 0)
    with pytest.raises(TypeError):
        divide_numbers("1", 2)
```

## 持续集成 / Continuous Integration

### 预提交钩子 / Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
- repo: https://github.com/psf/black
  rev: 22.3.0
  hooks:
    - id: black
- repo: https://github.com/pycqa/flake8
  rev: 4.0.1
  hooks:
    - id: flake8
```

## 参考资料 / References

- [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Type Hints Documentation](https://docs.python.org/3/library/typing.html)