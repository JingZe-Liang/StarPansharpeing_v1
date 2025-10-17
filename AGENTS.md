# OpenCode 基本要求
- 始终用中文与用户沟通
- **不要**使用try-catch, **不要**使用try-catch, **不要**使用try-catch, **不要**使用try-catch, **不要**使用try-catch, **不要**使用try-catch
- 写代码的时候使用ruff格式进行python代码的格式化，确保代码风格一致
- 遇到用户解释性的问题的时候，尽可能地将细节解释完整，描述清楚
- Python函数的变量需要有Python >3.12 type hint，不要用List, Tuple这些过时的hint，使用list, tuple这样的
- 写代码之前你需要先行查看项目规范
- 当明确要求block注释的时候才注释，行内注释自由无限制。
- 当要求block注释的时候，主要函数和类的注释都必须使用numpy docstring的风格，例如
```text
    Usage of this function.

    Returns some value.

    Parameters
    ----------
    a : array_like of real numbers
    b : array_like of real numbers
    ...

    Raise
    ----------
    ...

    Note (if you have something to note)
    ----------
```
对于非重要的函数，例如子函数等，只需要简单进行注释即可。
确保代码的注释清晰，并且**必须**使用英文进行注释
- 写代码的时候需要考虑复用，尽可能把不同的功能分开成小的子函数
- 写完代码之后记得使用mypy进行语法检查，确保代码无错误
- **不要**使用try-catch

# OpenCode 八荣八耻
- 以瞎猜接口为耻，以认真查询为荣。
- 以模糊执行为耻，以寻求确认为荣。
- 以臆想业务为耻，以人类确认为荣。
- 以创造接口为耻，以复用现有为荣。
- 以跳过验证为耻，以主动测试为荣。
- 以破坏架构为耻，以遵循规范为荣。
- 以假装理解为耻，以诚实无知为荣。
- 以盲目修改为耻，以谨慎重构为荣。