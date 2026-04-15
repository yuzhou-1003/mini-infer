"""最简单的 PyTorch 验证程序：创建张量、执行运算、验证结果。"""

import torch


def test_tensor_operations():
    # 创建张量
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])

    # 基本运算
    c = a + b
    assert torch.equal(c, torch.tensor([5.0, 7.0, 9.0]))

    # 点积
    dot = torch.dot(a, b)
    assert dot.item() == 32.0

    # 矩阵乘法
    m1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    m2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    result = m1 @ m2
    expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]])
    assert torch.equal(result, expected)


def test_autograd():
    # 自动求导：f(x) = x^2, f'(x) = 2x
    x = torch.tensor(3.0, requires_grad=True)
    y = x ** 2
    y.backward()
    assert x.grad.item() == 6.0


def test_simple_linear_model():
    # 用一个线性层做一次前向传播
    torch.manual_seed(42)
    model = torch.nn.Linear(4, 2)
    input_data = torch.randn(1, 4)
    output = model(input_data)
    assert output.shape == (1, 2)


if __name__ == "__main__":
    test_tensor_operations()
    test_autograd()
    test_simple_linear_model()
    print("所有测试通过！")
