class Solution:

    def pooling(self, matrix):
        ang = [int(i) for i in range(len(matrix)) if i % 2 == 0]
        res = 0
        for i in ang:
            for j in ang:
                tmp = []
                for val in matrix[i:i + 2][j: j + 2]:
                    tmp += val
                res = max(res, sorted(val)[-2])
        return res


if __name__ == "__main__":
    n = int(input())
    res = []
    for i in range(n):
        tmp = list(map(int, input().split()))
        res.append(tmp)
    obj = Solution()
    print(obj.pooling(res))