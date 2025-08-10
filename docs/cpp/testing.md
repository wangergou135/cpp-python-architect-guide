# C++测试指南 / C++ Testing Guidelines

*最后更新 / Last Updated: 2025-08-10 10:45:25 UTC*

## 单元测试 / Unit Testing

### Google Test框架 / Google Test Framework
```cpp
#include <gtest/gtest.h>

TEST(MyTestCase, TestName) {
    EXPECT_EQ(function(), expectedValue);
}
```

[Rest of testing guidelines content...]