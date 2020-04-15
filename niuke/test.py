import time
import math


def mutl(num):
        if num == 1:
            return 1
        num = num*mutl(num-1)
        return num


def add(num):
        if num == 1:
            return 1
        num = num + add(num-1)
        return num


def Fibonacci(n):
    if n == 1 or n == 2:
        return 1
    n = Fibonacci(n-1) + Fibonacci(n-2)
    return n


class Solution:
    # array 二维列表
    def Find(self, target, array):
        # write code here
        if len(array[0]) == 0:
            return False
        low = 0
        high = len(array)
        while low <= high-1:
            if target in array[low]:
                return True
            else:
                low = low + 1
        return False

    def replaceSpace(self, s):
        # write code here
        a = ""
        for num, i in enumerate(s):
            # print(s)
            if i == " ":
                i = "%20"
            a = a + i
        return a

    def minNumberInRotateArray(self, rotateArray):
        len_num = len(rotateArray)
        if len_num == 0:
            return 0
        low = 0
        high = len_num - 1
        num = 0
        # num = min(rotateArray)
        min_num = rotateArray[0]
        for i in range(len_num-1):
            if rotateArray[i] > rotateArray[i + 1]:
                return rotateArray[i + 1]
            # mid = (low + high)//2
            # if rotateArray[mid] > rotateArray[low]:
            #     low = mid + 1
            # if rotateArray[mid] > rotateArray[high]:
            #     high = mid - 1
            # else:
            #     num = rotateArray[mid]
        return rotateArray[0]

    def Fibonacci(self, n):
        a = 1
        b = 2
        # d = []
        if n == 0:
            return 0
        for i in range(n):
            if i == 0:
                c = 1
                # d.append(c)
            if i == 1:
                c = 2
            else:
                c = a + b
                # d.append(c)
                a = b
                b = c
        return c

    def factorial(self, end, start):
        num = 1
        if end == start or start == 0:
            return 1
        for i in range(start, end+1):
            num = num * i
        return num

    def jumpFloor(self, n):
        a = 1
        b = 2
        # d = []
        if n == 0:
            return 0
        for i in range(n):
            if i == 0:
                c = 1
                # d.append(c)
            if i == 1:
                c = 2
            if i >= 2:
                c = a + b
                # d.append(c)
                a = b
                b = c
        return c

    def Power(self, base, exponent):
        # write code here
        num = 1
        if base == 0:
            return 0.0
        if base == 1 or exponent == 0:
            return 1
        if exponent == 1:
            return base
        exp = 1
        abs_num = abs(exponent)
        for i in range(abs_num):
            exp = exp * base
        if exponent < 0:
            exp = 1 / exp
        return exp


start = time.perf_counter()

t = Solution()
print(t.second(2, 1, [[1, 1, 2]]))
end = time.perf_counter()
print(end - start)
