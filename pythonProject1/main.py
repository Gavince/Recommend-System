import random


class Solution:

    def param_init(self, a, b, c, x, y, y_hat):
        """参数更新"""
        loss_w = (y - y_hat)
        a = a + 0.01 * loss_w * (x ** 3)
        b = b + 0.01 * loss_w * (x ** 2)
        c = c + 0.01 * loss_w * 1

        return a, b, c

    def modeFit(self, datas):
        a = b = c = random.random()
        for x, y in datas:
            y_hat = a * (x**3) + b * (x**2) + c
            a, b, c = self.param_init(a, b, c, x, y, y_hat)

        return a, b, c


if __name__ == "__main__":
    n = int(input())
    datas = []
    for i in range(n):
        datas.append(list(map(float, input().split())))
    obj = Solution()
    res = obj.modeFit(datas)
    for val in res:
        print(val, end=" ")
