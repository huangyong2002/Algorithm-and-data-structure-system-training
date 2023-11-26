# 栈的操作
## 简单栈的性质

```py
// leetcode 32

```
## 栈的模拟
```py

```

## 单调栈，找到右边比我小的数的位置
```py
class Solution:
  def findRightSmall(self, A):
    if not A or len(A) == 0:
      return []

    # 结果数组
    ans =[0] * len(A)

    # 注意，栈中的元素记录的是下标
    t = []

    for i in range(0, len(A)):
      x = A[i]
      # 每个元素都向左遍历栈中的元素完成消除动作
      while len(t) > 0 and A[t[-1]] > x:
        # 消除的时候，记录一下被谁消除了
        ans[t[-1]] = i
        # 消除时候，值更大的需要从栈中消失
        t.pop()

      # 剩下的入栈
      t.append(i)

    # 栈中剩下的元素，由于没有人能消除他们，因此，只能将结果设置为-1。
    while len(t) > 0:
      ans[t[-1]] = -1
      t.pop()

    return ans
```
## 单调栈的性质
```py
// LeetCode 84
class Solution(object):
    def largestRectangleArea(self, A):
        N = 0 if not A else len(A)

        # 递增栈
        t = []

        ans = 0

        for i in range(0, N+1):
            x = A[i] if i < N else -1
            while len(t) > 0 and A[t[-1]] > x:
                # 这里以A[idx]为高度
                idx = t[-1]
                height = A[idx]
                t.pop()
                # 根据性质2，右边比A[idx]大的就是[idx + 1... i)
                rightPos = i
                leftPos = t[-1] if len(t) > 0 else -1

                width = rightPos - leftPos - 1
                area = height * width

                ans = max(ans, area)
            t.append(i)
        
        return ans
```

# 队列
## 循环队列
```py
class MyCircularQueue(object):
    def __init__(self, k):
        """
        :type k: int
        """
        # 第一个元素所在位置
        self.front = 0
        # rear是enQueue可在存放的位置
        # 注意开闭原则
        # [front, rear)
        self.rear = 0
        # 循环队列的存储空间, 注意这里使用的是k+1
        self.a = [0] * (k + 1)
        # 记录最大空量
        self.capacity = k + 1

    def enQueue(self, value):
        """
        :type value: int
        :rtype: bool
        """
        # 如果已经满了，
        if self.isFull():
            return False
        # 如果没有放满，那么a[rear]用来存放新进来的元素
        self.a[self.rear] = value
        # rear向前进
        self.rear = (self.rear + 1) % self.capacity
        return True

    def deQueue(self):
        """
        :rtype: bool
        """
        # 如果是一个空队列，当然不能出队
        if self.isEmpty():
            return False
        # 注意取模
        self.front = (self.front + 1) % self.capacity
        # 取出元素成功
        return True

    def Front(self):
        """
        :rtype: int
        """
        # 如果为空，不能取出队首元素
        if self.isEmpty():
            return -1
        # 取出队首元素
        return self.a[self.front]

    def Rear(self):
        """
        :rtype: int
        """
        # 如果为空，不能取出队尾元素
        if self.isEmpty():
            return -1
        # 注意：这里不能使用rear - 1
        # 需要取模
        tail = (self.rear - 1 + self.capacity) % self.capacity
        return self.a[tail]

    def isEmpty(self):
        """
        :rtype: bool
        """
        return self.front == self.rear

    def isFull(self):
        """
        :rtype: bool
        """
        nextRear = (self.rear + 1) % self.capacity
        return nextRear == self.front
```
## 单调队列
```py
class Solution(object):
    def __init__(self):
        # 单调队列使用双端队列来实现
        self.Q = deque()
    
    def push(self, val):
        """
        # 入队的时候，last方向入队，但是入队的时候
        # 需要保证整个队列的数值是单调的
        # (在这个题里面我们需要是递减的)
        # 并且需要注意，这里是Q[-1] < val
        """
        while self.Q and self.Q[-1] < val:
            self.Q.pop()
        # 将元素入队
        self.Q.append(val)

    def pop(self, val):
        # 出队的时候，要相等的时候才会出队
        if self.Q and self.Q[0] == val:
            self.Q.popleft()
```
## 堆
由于堆往往用来实现优先级队列，因此，这里我也整理好了堆的实现的代码：
```py
class Heap(object):
    a = []
    n = 0

    def _sink(self, i):
        t = self.a[i]
        while i + i + 1 < self.n:
            j = i + i + 1
            if j < self.n - 1 and self.a[j] < self.a[j+1]:
                j += 1
            if self.a[j] > t:
                self.a[i] = self.a[j]
                i = j
            else:
                break
        self.a[i] = t

    def _swim(self, i):
        t = self.a[i]
        while i > 0:
            par = (i-1) / 2
            if self.a[par] < t:
                self.a[i] = self.a[par]
                i = par
            else:
                break
        self.a[i] = t

    def push(self, x):
        self.a.append(x)
        self._swim(self.n)
        self.n += 1

    def pop(self):
        ret = self.a[0]
        self.a[0] = self.a[self.n-1]
        self.a.pop()
        self.n -= 1
        self._sink(0)
        return ret

    def size(self):
        return self.n
```
# 链表
## 链表的基本操作
```py
#单链表结点的定义
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

# 实现单链表
# 1. 假设链表中的所有节点都是 0-index的。

class MyLinkedList(object):
    def __init__(self):
        self.dummy = ListNode(0)
        self.tail = self.dummy
        self.length = 0

    def getPrevNode(self, index):
        front = self.dummy.next
        back = self.dummy
        for i in range(index):
            back = front
            front = front.next
        return back

    def get(self, index):
        if index < 0 or index >= self.length:
            return -1
        return self.getPrevNode(index).next.val

    def addAtHead(self, val):
        p = ListNode(val)
        p.next = self.dummy.next
        self.dummy.next = p

        # 注意，这里一定要记得修改tail
        if self.tail == self.dummy:
            self.tail = p

        self.length += 1

    def addAtTail(self, val):
        self.tail.next = ListNode(val)
        self.tail = self.tail.next
        self.length += 1

    # 在链表中的第 index 个节点之前添加值为 val  的节点。
    # 1. 如果 index 等于链表的长度，则该节点将附加到链表的末尾。
    # 2. 如果 index 大于链表长度，则不会插入节点。
    # 3. 如果index小于0，则在头部插入节点。
    def addAtIndex(self, index, val):
        if index > self.length:
            return
        elif index == self.length:
            self.addAtTail(val)
            return
        elif index <= 0:
            self.addAtHead(val)
            return

        pre = self.getPrevNode(index)
        p = ListNode(val)
        p.next = pre.next
        pre.next = p

        # NOTE: here tail has been changed
        self.length += 1

    # 如果索引 index 有效，则删除链表中的第 index 个节点。
    def deleteAtIndex(self, index):
        if index < 0 or index >= self.length:
            return

        pre = self.getPrevNode(index)

        # NOTE:  change tail
        if self.tail == pre.next:
            self.tail = pre

        self.length -= 1

        pre.next = pre.next.next
```

# 树
## 前序遍历
```py

```
## 中序遍历
```py

```
## 后序遍历
采用递归实现的后序遍历代码模板如下（解析在注释里）：
```py
        def postOrder(root, ans):
            if root:
                // 先遍历左子树
                postOrder(root.left, ans)
                // 然后遍历右子树
                postOrder(root.right, ans)
                // 最后遍历中间的根节点
                ans.append(root.val)
```
采用非递归的后序遍历【后序遍历——栈】代码如下（解析在注释里）：
```py
class Solution(object):
    def postorderTraversal(self, t):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        ans = []
        # pre 在后序遍历时的前面一个结点
        pre = None
        s = []
        # 如果栈中还有元素，或者当前结点t非空
        while len(s) > 0 or t:
            # 顺着左子树走，并且将所有的元素压入栈中
            while t:
                s.append(t);
                t = t.left;

            # 当没有任何元素可以压栈的时候
            # 拿栈顶元素，注意这里并不将栈顶元素弹出
            # 因为在迭代时，根结点需要遍历两次，这里需要判断一下
            # 如果是第一次遍历是不能弹栈的。
            t = s[-1]
            # 1. 如果当前结点左子树为空，那么右子树没有遍历的必要
            # 需要将当前结点放到ans中
            # 2. 当t.right == pre时，说明右子树已经被打印过了
            # 那么此时需要将当前结点放到ans中
            if not t.right or t.right == pre:
                # 此时为第二次遍历根结点，所以放到ans中。
                ans.append(t.val)
                # 因为已经遍历了当前结点，所以需要更新pre结点
                s.pop()
                # 已经打印完毕。需要设置为空，否则下一轮循环
                # 还会遍历t的左子树。
                pre = t
                t = None
            else:
                # 第一次走到t结点，不能放到ans中，因为t的右子树还没有遍历。
                # 需要将t结点的右子树遍历
                t = t.right;
        return ans
```

# 并查集
```python
class UF(object):
    def __init__(self, N):
        self.F = [0] * N
        self.Cnt = [0] * N
        for i in range(N):
            self.F[i] = i
            self.Cnt[i] = 1

        self.count = N

    def Find(self, x):
        if x == self.F[x]:
            return x
        self.F[x] = self.Find(self.F[x])
        return self.F[x]

    def Union(self, x, y):
        xpar = self.Find(x)
        ypar = self.Find(y)
        if xpar != ypar:
            self.F[xpar] = ypar
            self.Cnt[ypar] += self.Cnt[xpar]
            self.count -= 1

    def Size(self, i):
        return self.Cnt[self.Find(i)]

    def Count(self):
        return self.count
```

# 排序算法
## 合并排序
```py
class Solution(object):
    def mergeSort(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        if not nums or len(nums) == 0:
            return

        t = [0] * len(nums)

        def msort(a, b, e, t):
            if b >= e or b + 1 >= e:
                return

            m = b + ((e-b)>>1)
            msort(a, b, m, t)
            msort(a, m, e, t)

            i = b
            j = m
            to = b

            while i < m or j < e:
                if j >= e or (i < m and a[i] <= a[j]):
                    t[to] = a[i]
                    to += 1
                    i += 1
                else:
                    t[to] = a[j]
                    to += 1
                    j += 1

            for i in range(b, e):
                a[i] = t[i]
        msort(nums, 0, len(nums), t)
```

## 三路切分
```py
// LeetCode 75
class Solution(object):
    def singleNumber(self, A):
        N = len(A) if A else 0

        def split(A, b, e):
            if b >= e:
                return 0
            if b + 1 >= e:
                return A[b]

            m = b + ((e-b)>>1)
            x = A[m]

            i = b
            l = b
            r = e - 1

            while i <= r:
                if A[i] < x:
                    A[l], A[i] = A[i], A[l]
                    i += 1
                    l += 1
                elif A[i] == x:
                    i += 1
                else:
                    A[r], A[i] = A[i], A[r]
                    r -= 1
            
            if i - l == 1:
                return A[l]
            
            if ((l - b) & 0x01) == 1:
                return split(A, b, l)
            return split(A, i, e)

        return split(A, 0, N)
```

# 二分搜索
lower_bound
```python
def lowerBound(A, target):
    l = 0
    r = len(A) if A else 0
    while l < r:
        m = l + ((r-l)>>1)
        if (A[m] < target):
            l = m + 1
        else:
            r = m
    return l
```

upper_bound
```python
def upperBound(A, target):
    l = 0
    r = len(A) if A else 0
    while l < r:
        m = l + ((r-l)>>1)
        if (A[m] <= target):
            l = m + 1
        else:
            r = m
    return l
```

# 双指针
## 最长区间
```python

```
## 定长区间
```python

```
## 最短区间
```python

```

# 贪心
相互不覆盖的区间的数目模板
```python
def nonOverlapIntervals(self, A):
    if not A or len(A) == 0:
        return 0

    A.sort(key=lambda x: x[1])

    preEnd = float('-inf')
    ans = 0

    for r in A:
        start = r[0]
        end = r[1]

        if preEnd <= start:
            preEnd = end
            ans += 1

    return ans
```
青蛙跳算法的模板
```python
    def canJump(self, A):
        """
        :type nums: List[int]
        :rtype: bool
        """

        N = 0 if not A else len(A)

        i = 0

        while i < N and (i + A[i] < N - 1):
            old = i + A[i]
            j = i + 1
            maxPos = old
            while j <= old:
                if j + A[j] > maxPos:
                    maxPos = j + A[j]
                    i = j
                j += 1
            
            if maxPos == old:
                return False
        
        return True
```

# 回溯
```python

```

# DFS与BFS
## DFS
```python

```
收集所有的满足条件的解
```python

```
## BFS
```python

```

```python

```

# 动态规划
```python
class Solution(object):
    # 先不管这个函数的实现
    def buildNext(self, sub):
        N = 0 if not sub else len(sub)
        next = [0] * (N + 1)

        i = 0
        j = -1
        next[0] = -1

        while i < N:
            if -1 == j or sub[i] == sub[j]:
                i += 1
                j += 1

                if i < N and j < N and sub[i] == sub[j]:
                    next[i] = next[j]
                else:
                    next[i] = j
            else:
                j = next[j]

        return next

    def strStr(self, main, sub):
        alen = 0 if not main else len(main)
        blen = 0 if not sub else len(sub)

        next = self.buildNext(sub)

        i = 0
        j = 0
        while i < alen and j < blen:
            if -1 == j or main[i] == sub[j]:
                i += 1
                j += 1
            else:
                j = next[j]

        return i - blen if j == blen else -1

```