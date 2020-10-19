# Leetcode-classic-problem

## Array & Linked list

1. https://leetcode.com/problems/remove-duplicates-from-sorted-array/
```
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        i = 0
        for j in range(1, len(nums)):
            if nums[i] != nums[j]:
                i += 1
                nums[i] = nums[j]
        return i+1
```
2. https://leetcode.com/problems/merge-two-sorted-lists/
```
class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        dummy = ListNode(0)
        curr = dummy
        while l1 and l2:
            if l1.val <= l2.val:
                curr.next = l1
                l1 = l1.next
            else:
                curr.next = l2
                l2 = l2.next
            curr = curr.next
        curr.next = l1 or l2
        return dummy.next
```
3. https://leetcode.com/problems/merge-sorted-array/
```
class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        while m > 0 and n > 0:
            if nums1[m-1] >= nums2[n-1]:
                nums1[m+n-1] = nums1[m-1]
                m -= 1
            else:
                nums1[m+n-1] = nums2[n-1]
                n -= 1
        if n > 0:
            nums1[:n] = nums2[:n]
```
4. https://leetcode.com/problems/swap-nodes-in-pairs/
```
class Solution(object):
    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head or not head.next: return head
        dummy = ListNode(0)
        dummy.next = head
        cur = dummy
        while cur.next and cur.next.next:
            first = cur.next # 1
            sec = cur.next.next # 2
            cur.next = sec # dummy->2
            first.next = sec.next # 1->3
            sec.next = first # 3 = first
            cur = cur.next.next # cur = 1
        return dummy.next   
```
5. https://leetcode.com/problems/3sum/
```
class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        n = len(nums)
        res = []
        nums.sort()
        # [-4,-1,-1,0,1,2]
        for i in range(n-1):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            left = i+1
            right = n-1
            while left < right:
                three_sum = nums[i] + nums[left] + nums[right]
                if three_sum > 0:
                    right-=1
                elif three_sum < 0:
                    left+=1
                else:
                    res.append([nums[i], nums[left], nums[right]])
                    while left+1 < right and nums[left] == nums[left+1]:
                        left += 1
                    while right-1 > left and nums[right] == nums[right-1]:
                        right -= 1
                    left += 1
                    right -= 1
        return res
```
## Map & Set
6. https://leetcode.com/problems/valid-anagram/
```
class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        table1, table2 = {}, {}
        for i in s:
            if i not in table1:
                table1[i] = 1
            else:
                table1[i] += 1
        for j in t:
            if j not in table2:
                table2[j] = 1
            else:
                table2[j] += 1
        return table1 == table2
```
7. https://leetcode-cn.com/problems/group-anagrams/
```
class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        table = {}
        for str in strs:
            key = tuple(sorted(str))
            if key in table:
                table[key].append(str)
            else:
                table[key] = [str]
        return table.values()
```
## Stack & Set
8. https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string/
```
class Solution(object):
    def removeDuplicates(self, S):
        """
        :type S: str
        :rtype: str
        """
        stack = []
        for s in S:
            stack.append(s)
            if len(stack) > 1 and stack[-1] == stack[-2]:
                stack.pop()
                stack.pop()
        return ''.join(stack)
```
9. https://leetcode.com/problems/remove-outermost-parentheses/
```
class Solution(object):
    def removeOuterParentheses(self, S):
        """
        :type S: str
        :rtype: str
        """
        res = ""
        left, right = 0, 0
        start, end = 0, 0
        for i in range(len(S)):
            if S[i] == "(":
                left += 1
                if left == 1:
                    start = i
            else:
                right += 1
                end = i
            if left == right:
                res += S[start+1:end]
                left, right = 0, 0
        return res
```
10. https://leetcode.com/problems/largest-rectangle-in-histogram/
```
class Solution(object):
    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        heights.append(0)
        stack = [-1]
        ans = 0
        for i in xrange(len(heights)):
            while heights[i] < heights[stack[-1]]:
                h = heights[stack.pop()]
                w = i - stack[-1] - 1
                ans = max(ans, h * w)
            stack.append(i)
        heights.pop()
        return ans
```
11. https://leetcode.com/problems/trapping-rain-water/
```
class Solution(object):
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        stack = []
        i = 0
        res = 0
        while i < len(height):
            while stack and height[i] > height[stack[-1]]:
                top = stack.pop()
                if not stack: break
                h = min(height[i], height[stack[-1]]) - height[top]
                w = i - stack[-1] - 1
                res += h*w
            stack.append(i)
            i += 1
        return res
```
## Bineary search
12. https://leetcode.com/problems/powx-n/
```
class Solution(object):
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        #x^10 = (x^5)^2 = (x^2*x^*2*x)^2
        if n == 0:
            return 1
        if n == 1:
            return x
        if n == -1:
            return 1/x
        if n%2 == 0:
            num = self.myPow(x, n/2)
            return num*num
        else:
            num = self.myPow(x, (n-1)/2)
            return num*num*x
```
13. https://leetcode.com/problems/dungeon-game/
```
class Solution(object):
    def calculateMinimumHP(self, dungeon):
        """
        :type dungeon: List[List[int]]
        :rtype: int
        """
        # dp[i][j] = min(dp[i+1][j], dp[i][j+1]), below and right 
        m, n = len(dungeon), len(dungeon[0])
        dp = [[float('inf') for _ in range(n+1)] for _ in range(m+1)]
        dp[m-1][n], dp[m][n-1] = 1, 1
        for i in range(m-1,-1,-1):
            for j in range(n-1,-1,-1):
                val = min(dp[i+1][j], dp[i][j+1]) - dungeon[i][j]
                if val < 1:
                    dp[i][j] = 1
                else:
                    dp[i][j] = val
        return dp[0][0]
```
## Recursion
14. https://leetcode.com/problems/maximum-depth-of-binary-tree/
```
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
```
15. https://leetcode-cn.com/problems/symmetric-tree/
```
class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if not root:
            return True
        return self.helper(root.left, root.right)
    
    def helper(self, left, right):
        if not left and not right:
            return True
        if not left or not right or left.val != right.val:
            return False
        return self.helper(left.left, right.right) and self.helper(left.right, right.left)
```
16. https://leetcode.com/problems/minimum-depth-of-binary-tree/
```
class Solution(object):
    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root is None:
            return 0
        
        if root.left is None:
            return self.minDepth(root.right) + 1
        if root.right is None:
            return self.minDepth(root.left) + 1
        
        return min(self.minDepth(root.left), self.minDepth(root.right)) + 1
```
17. https://leetcode.com/problems/minimum-distance-between-bst-nodes/
```
class Solution(object):
    def minDiffInBST(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        # left,root,right inorder
        res = []
        self.inOrder(root, res)
        min_distance = float('inf')
        for i in range(1, len(res)):
            min_distance = min(min_distance, res[i] - res[i-1])
        return min_distance
        
    def inOrder(self, root, res):
        if not root:
            return
        self.inOrder(root.left, res)
        res.append(root.val)
        self.inOrder(root.right, res)
```
18. https://leetcode.com/problems/binary-tree-paths/
