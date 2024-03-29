# Leetcode-classic-problem

## 数组和链表 Array & Linked list
加起来和最大的子数组
https://leetcode.com/problems/maximum-subarray/
```
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.
```
```
class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 方法1:动态规划dp[i]表示以nums[i]结尾的子数组能形成的最大和
        # dp[i]=max(dp[i-1]+nums[i],nums[i])，两种情况，在前面数字结尾的子数组基础上加上当前数字，或者另起炉灶
        # n = len(nums)
        # dp = [nums[0]] * n
        # for i in range(1, n):
        #     dp[i] = max(dp[i-1] + nums[i], nums[i])
        # # print(dp) # [-2, 1, -2, 4, 3, 5, 6, 1, 5]
        # return max(dp)
        
        # 方法2:优化动态规划，因为dp[i]只取决于dp[i-1],所以只用一个变量来记录dp[i-1],也就是curr_max
        n = len(nums)
        curr_max = final_max = nums[0]
        for i in range(1, n):
            curr_max = max(curr_max + nums[i], nums[i])
            # 如果当前数字结尾的子数组和比前面的子数组和更大，就更新结果
            final_max = max(final_max, curr_max)
        return final_max
        
        # 方法3:思想转变为：只要curr_max是正数，加上当前的数就可能变得更大。时间不会更快因为有max()操作
        # 但如果当前的数是负数，和不会变得更大。但是只要和不为负数，就可以为后面出一份力，后面一个数可能很大可以弥补，比如[5,4,-1,7,8]
        # 如果当前负数很小，导致和为负数，我们就会抛弃前面，另起炉灶
        # for i in range(1, len(nums)):
        #     if nums[i-1] > 0:
        #         nums[i] += nums[i-1]
        # # print(nums) # [5, 9, 8, 15, 23]
        # return max(nums)
```

排序数组去重
https://leetcode.com/problems/remove-duplicates-from-sorted-array/
```
Input: nums = [0,0,1,1,1,2,2,3,3,4]
Output: 5, nums = [0,1,2,3,4,_,_,_,_,_]
Explanation: Your function should return k = 5, with the first five elements of nums being 0, 1, 2, 3, and 4 respectively.
It does not matter what you leave beyond the returned k (hence they are underscores).
```
```
class Solution(object):
    def removeDuplicates(self, nums):
        # 使用快（j=1）慢（i=0）指针。只要j所在的元素不等于i所在的元素，就把i后面的元素变为j元素，i往右移
        # 可以理解为i和i前面的数都是不重复的，i后面的位置应该放下一个不重复的数字
        i = 0
        for j in range(1, len(nums)):
            if nums[i] != nums[j]:
                i += 1
                nums[i] = nums[j]
        # 返回不重复元素长度
        return i+1
```
合并两个排序链表
https://leetcode.com/problems/merge-two-sorted-lists/
```
Input: list1 = [1,2,4], list2 = [1,3,4]
Output: [1,1,2,3,4,4]
```
```
class Solution(object):
    def mergeTwoLists(self, l1, l2):
        # 假链表头
        # 取两个链表中较小值，然后将指针后移
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
        # l1或l2有剩余
        curr.next = l1 or l2
        return dummy.next
```
合并两个排序数组
https://leetcode.com/problems/merge-sorted-array/
```
Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
Output: [1,2,2,3,5,6]
Explanation: The arrays we are merging are [1,2,3] and [2,5,6].
The result of the merge is [1,2,2,3,5,6] with the underlined elements coming from nums1.
```
```
class Solution(object):
    def merge(self, nums1, m, nums2, n):
        # 从数组后面开始比较
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
柱子能形成的最大容器
https://leetcode.com/problems/container-with-most-water/
```
Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
Explanation: The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. In this case, the max area of water (blue section) the container can contain is 49.
```
```
class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        # 方法1:暴力解法，超时O(n^2)。每个柱子跟后面每根柱子形成的容器体积，取最大体积
        # res = 0
        # n = len(height)
        # for i in range(n-1):
        #     for j in range(i+1, n):
        #         # 体积取决于较矮的柱子
        #         v = min(height[i], height[j]) * (j-i)
        #         res = max(res, v)
        # return res
    
        # 方法2: 最宽的容器有可能是最大体积（第一根和最后一根）
        # 又因为体积取决于较矮的柱子，中间可能有更高的柱子，所以我们排除较矮的柱子，重新组成容器
        i, j = 0, len(height) - 1
        res = 0
        while i < j:
            res = max(res, min(height[i], height[j]) * (j - i))
            if height[i] < height[j]:
                i += 1
            else:
                j -= 1
        return res
```
反转整个链表
https://leetcode.com/problems/reverse-linked-list/
```
Input: head = [1,2,3,4,5]
Output: [5,4,3,2,1]
```
```
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        # 思考方式1：从第二个数开始，依次把数字移到最前面
        # dummy->1->2->3->4->5
        # dummy->2->1->3->4->5
        # dummy->3->2->1->4->5
        # dummy->4->3->2->1->5
        # dummy->5->4->3->2->1
        # 只有一个数字的情况
        if not head or not head.next:
            return head
        # 多于一个数字的情况
        dummy = ListNode(0, head)
        curr = head.next # 2
        pre_curr = head # 1
        while curr:
            pre_curr.next = curr.next # 1->3
            next_curr = curr.next
            curr.next = dummy.next # 2->1
            dummy.next = curr # dummy->2
            curr = next_curr # curr=3      
        return dummy.next
        
        # 思考方式2:后面一个数指向前面一个数
        # None  1->2->3->4->5
        # None<-1  2->3->4->5
        # None<-1<-2  3->4->5
        # None<-1<-2<-3  4->5
        # None<-1<-2-<3<-4  5
        # None<-1<-2<-3<-4<-5
        if not head or not head.next:
            return head
        pre = None
        curr = head # 1 
        while curr:
            cur_next = cur.next # 2
            curr.next = pre # 1->None
            pre = curr
            curr = curr_next # 2
        return pre
```
成对交换链表节点
https://leetcode.com/problems/swap-nodes-in-pairs/
```
Input: head = [1,2,3,4]
Output: [2,1,4,3]
```
```
class Solution(object):
    def swapPairs(self, head):
        if not head or not head.next: return head
        dummy = ListNode(0)
        # 需要记录pair前面的位置
        # 所以前面增加一个假头
        dummy.next = head
        cur = dummy
        # 假头后面和后面的后面存在
        while cur.next and cur.next.next:
            first = cur.next # 1
            sec = cur.next.next # 2
            # 注意顺序
            cur.next = sec # dummy->2
            first.next = sec.next # 1->3
            sec.next = first # 2 -> 1
            # 重新设定cur，作为下一个pair前的位置
            cur = first # cur = 1
        return dummy.next
```
链表去掉倒数第n个节点
https://leetcode.com/problems/remove-nth-node-from-end-of-list/
```
Input: head = [1,2,3,4,5], n = 2
Output: [1,2,3,5]
```
```
class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        # 两个指针，快指针比慢指针先走n个节点。当快指针到null时，慢指针所在的地方就是要去掉的节点
        # 快指针先走n个节点
        slow = fast = head
        for _ in range(n):
            fast = fast.next
        # 同步走快慢指针, 出循环时快指针到null
        dummy = ListNode(0, head)
        pre_slow = dummy
        while fast:
            pre_slow = pre_slow.next
            slow = slow.next
            fast = fast.next
        # 去掉slow所在节点，需要slow.prev.next = slow.next, 所以要记录slow前一个节点，使用dummy节点
        pre_slow.next = slow.next
        return dummy.next
```
三个数之和为0
https://leetcode.com/problems/3sum/
```
Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]
```
```
class Solution(object):
    def threeSum(self, nums):
        n = len(nums)
        res = []
        nums.sort()
        # [-4,-1,-1,0,1,2], 排序，转化为2sum问题
        for i in range(n-1):
            # 去掉重复元素情况
            if i > 0 and nums[i] == nums[i-1]:
                continue
            # 两边向中间收缩，找到2sum
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
                    # 考虑[left, right]之间有重复元素的情况
                    while left+1 < right and nums[left] == nums[left+1]:
                        left += 1
                    while right-1 > left and nums[right] == nums[right-1]:
                        right -= 1
                    # 找到3sum匹配，继续往中间收缩，因为可能出现[-2,-1,1,2]这种情况，和都为0
                    left += 1
                    right -= 1
        return res
```
合并多个范围
https://leetcode.com/problems/merge-intervals/
```
Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlap, merge them into [1,6].
```
```
class Solution(object):
    def merge(self, intervals):
        # 对于多个interval的题，基本都要对起点或终点进行排序
        result = []
        if len(intervals) == 0:
            return result
        # 对起点进行排序，对之后的interval来说，只要起点小于前面interval的终点，就有交集
        intervals.sort(key=lambda x:x[0])
        result.append(intervals[0])
        for i in range(1, len(intervals)):
            # 没有交集，直接加入result
            if result[-1][1] < intervals[i][0]:
                result.append(intervals[i])
            # 有交集，有两种情况，如果前面interval的终点小于当前interval的终点，需要更新前面interval的终点
            else:
                if result[-1][1] < intervals[i][1]:
                    result[-1][1] = intervals[i][1]
        return result
```
跳跃游戏2(中等/困难)
https://leetcode.com/problems/jump-game-ii/
```
Input: nums = [2,3,1,1,4]
Output: 2
Explanation: The minimum number of jumps to reach the last index is 2. Jump 1 step from index 0 to 1, then 3 steps to the last index.
```
```
class Solution(object):
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 方法1:动态规划O(n^2)超时，dp[i]表示nums[0:i+1]的最小步数
        # dp[0]=0,不用跳就到达，dp[1]=1如果dp[0]有大于等于1步
        # dp[2]=1如果dp[0]有大于等于2步或者dp[2]=2如果dp[0]只有1步然后dp[1]有大于等于1步。。。
        # 遍历i位置前面的位置j，找到能从j跳到到i的j值，取最小步数。dp[i]=min(dp[j])+1,0<j<i并且i-j<=nums[j]
        # n = len(nums)
        # dp = [0] * n
        # for i in range(1, n):
        #     local_min = float('inf')
        #     for j in range(i):
        #         # j到i的距离小于等于能从j跳的步数，说明能从j跳到i
        #         if i-j <= nums[j]:
        #             local_min = min(local_min, dp[j])
        #     dp[i] = local_min + 1
        # # print(dp) # [0, 1, 1, 2, 2]
        # return dp[-1]
        
        # 方法2:倒推法，贪心算法，O(n^2)但是不超时，因为有大量的剪枝
        # position从最后一个位置开始，找最后一跳最左边的起跳位置，依次倒推找前面的起跳位置
        # position = len(nums) - 1;
        # steps = 0;
        # # 只要还没跳回起点，就一直倒推
        # while position != 0:
        #     # 判断是否能从前面的位置i跳到位置position, 如果可以就找到了最左边的起跳位置
        #     for i in range(position):
        #         if  position - i <= nums[i]:
        #             position = i
        #             steps += 1
        #             break
        # return steps

        # 方法3:贪心算法，线性扫描O(n)。
        # [2,1,3,1,1,4]，如果len(nums)>1,至少一跳，跳出起点。初始last_max_range=0，表示i=0就是极限位置，必须跳
        # i=0，最大能到的位置cur_max_range是i+nums[i]=2，0==last_max_range，到达极限，要更新last_max_range为cur_max_range=2
        # 意思是下一跳必须从[1,3]开始，第二跳肯定在其中，到最大位置i=2时为极限，必须跳
        # 所以当i==last_max_range时，增加一步，并不是说在极限位置跳，而是在这个区间里面跳
        # i=1，最大能到的位置cur_max_range为2。
        # i=2，最大能到的位置max_range为5，i==last_max_range，达到极限，必须跳一步，更新last_max_range为5
        # 如果最大能到的位置覆盖最后一个位置，就可以提前结束循环
        n = len(nums)
        steps = 0
        last_max_range = 0
        cur_max_range = 0
        # 最后一个位置不用遍历，已经到达
        for i in range(n-1):
            cur_max_range = max(cur_max_range, i+nums[i])
            if cur_max_range >= n - 1:
                steps += 1
                break
            # 到达极限，需要跳一步
            if i == last_max_range:
                steps += 1
                last_max_range = cur_max_range
        return steps
```
两个排序数组中的中位数
https://leetcode.com/problems/median-of-two-sorted-arrays/
```
Input: nums1 = [1,3], nums2 = [2]
Output: 2.00000
Explanation: merged array = [1,2,3] and median is 2.
```
```
class Solution(object):
    # 思想：中位数就是把两个数组合起来一分为二的那个数
    # 这个数位置的左边都比他小，右边都比他大。左边的数由左边的A和左边的B组成，右边也一样
    # 所以要找到一个位置分割A，同时也就找到了分割位置B（A左+B左=half_len）。右边的元素都大于左边的元素
    # 如果分割的位置恰好为中位数的位置的话，会满足条件：A右第一个元素>B左最后一个元素，B右第一个元素>A左最后一个元素
    def findMedianSortedArrays(self, A, B):
        m, n = len(A), len(B)
        # 我们需要在更短的数组里面进行二分查找，我们认为更短的数组为A
        if m > n:
            A, B, m, n = B, A, n, m
        if n == 0:
            raise ValueError
        # 在A里面进行二分查找，half_len是两个数组长度的一半
        # m+n+1是为了处理奇数个的情况
        imin, imax, half_len = 0, m, (m + n + 1) / 2
        while imin <= imax:
            i = (imin + imax) / 2 # A左边的长度
            j = half_len - i      # B被分割后左边的长度（A左边长度+B左边的长度=half_len）
            # 第一种情况，不满足A右第一个元素>B左最后一个元素，分割位置需要右移，同时判断A边界
            if i < m and B[j-1] > A[i]: 
                imin = i + 1
            # 第二种情况，不满足B右第一个元素>A左最后一个元素，分割线需要左移，同时判断A边界
            elif i > 0 and A[i-1] > B[j]:
                imax = i - 1
            # 满足两个条件，找到合适分割位置
            else:
                if i == 0: max_of_left = B[j-1]
                elif j == 0: max_of_left = A[i-1]
                else: max_of_left = max(A[i-1], B[j-1])

                if (m + n) % 2 == 1:
                    return max_of_left

                if i == m: min_of_right = B[j]
                elif j == n: min_of_right = A[i]
                else: min_of_right = min(A[i], B[j])

                return (max_of_left + min_of_right) / 2.0
```
最长递增子序列
https://leetcode.com/problems/longest-increasing-subsequence/
```
Input: nums = [10,9,2,5,3,7,101,18]
Output: 4
Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4.
```
```
class Solution(object):
    def lengthOfLIS(self, nums):
        # 方法1: O(n^2)动态规划。dp[i]表示从头到第i个数（选择第i个数），最长子序列长度
        # dp[i] = max(dp[i前面所有的数j]) + 1, 0 <= j <= i-1
        # 当i比前面某一个元素大时nums[i] > nums[j]，有可能组成最长递增子序列，需要在可能的长度中取最大值
        # 如果i没有比前面任何元素大，最长递增子序列就是本身，长度为1
        # n = len(nums)
        # dp = [1] * n
        # for i in range(1, n):
        #     preDPMax = dp[0] //前面元素结尾能组成最长递增子序列的最大长度
        #     for j in range(i):
        #         if nums[j] < nums[i]: //有可能组成最长递增子序列
        #             preDPMax = max(preDPMax, dp[j]) //判断是否最长，如果更长则更新
        #             dp[i] = preDPMax + 1
        # # [10,9,2,5,3,7,101,18]
        # # dp = [1, 1, 1, 2, 2, 3, 4, 4]
        # return max(dp)
    
        # 方法2: O(nlogn)
        # 构建结果数组res=[nums[0]]. res的长度是结果
        # 如果当前数字小于res里最后一个数（最大数），LIS就可能从这个数开始(这个数最小)或者增长（不是最小），两种情况都需要替换掉比他大的最小数
        # 比如[10,9,4,5,1,2,3,4]中的1，1需要替换4形成[1,5]，因为后面还可能出现2或3或4可以替换掉5，我们需要前面的数越小越好
        # 1替换4形成[1,5]只是增加找到更长的可能性，不是说之后一定能找到更长的。如果没有找到，[1,5]和[4,5]其实长度也是相等的
        # 重要：构建的数组res不一定是得到的最长子序列，但是长度是一样的
        # 如果当前数字比最大数都大，LIS就可以增长，加入res最后
        # 被替换数的位置用二分法找到，因为res是有序数组。
        res = [nums[0]]
        for i in range(1, len(nums)):
            if nums[i] == res[-1]: continue
            elif nums[i] < res[-1]:
                # 二分查找比nums[i]大的最小数
                index = bisect_left(res, nums[i])
                # 替换
                res[index] = nums[i]
            else:
                res.append(nums[i])
        return len(res)
```
## 二分查找 Binery search
旋转排序数组里寻找某个数字
https://leetcode.com/problems/search-in-rotated-sorted-array/
```
Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4
```
```
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        # 数组是排好序但是经过旋转，可以二分查找，但是要判断是[2,4,5,6,7,0,1]还是[6,7,0,1,2,4,5]
        # 因为现在target和nums[mid]的大小关系不能确定target在左边还是右边
        # 只有在确定是哪种旋转后才能确定是在左边还是右边
        # 使用nums[left]，nums[mid]大小关系判断
        n = len(nums)
        left, right = 0, n-1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            # 第一种情况，左边单调递增
            if nums[left] <= nums[mid]:
                # target有可能在左边或右边
                # 在左边满足：
                if nums[left] <= target <= nums[mid]:
                    right = mid - 1
                # 在右边
                else:
                    left = mid + 1
            # 第二种情况，右边单调递增
            else: # nums[left] > nums[mid]:
                # target有可能在右边或左边
                # 在右边满足：
                if nums[mid] <= target <= nums[right]:
                    left = mid + 1
                # 在左边
                else:
                    right = mid - 1
        return -1
```
在排序矩阵中找到数字
https://leetcode.com/problems/search-a-2d-matrix/
```
Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
Output: true
```
```
class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        # 从右上角出发，因为右上角左边的数字都比他小，下面的数字都比他大，可以二分查找
        # 当目标数比右上角小，就往左边找，j-=1；当目标数比右上角大，就往下面一行找，i+=1，再把目标数与下一行的数进行比较。。。
        if len(matrix) == 0:
            return 0
        # i代表行，j代表列，从右上角开始
        i = 0 
        j = len(matrix[0]) - 1
        while j>=0 and i<=len(matrix)-1:
            if matrix[i][j] == target:
                return True
            # 当目标数比右上角小
            elif target < matrix[i][j]:
                j -= 1
            # 当目标数比右上角大
            else:
                i += 1
        return False
```

## 字符串 String
没有重复字母的最长子字符串
https://leetcode.com/problems/longest-substring-without-repeating-characters/
```
Input: s = "tmmzuxt"
Output: 5
"mzuxt"
```
```
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        # 方法1: O（n^3）用两个for循环遍历所有substring，用set判断里面是否有重复元素
        # 方法1改进: O（n^2）遍历每个字符，看从当前字符开始，往右看，用set记录已经形成的字符串，检查形成的最长不重复字符串是多少
        # 方法2空间换时间: O（n）维护一个滑动窗口，记录元素最后出现位置。如果遇到相同元素，就说明新起始点应该在之前最后一次遇到这个元素位置后面一个位置。
        # 滑动窗口map
        hashmap = {}
        max_ = 0
        # 最长不重复字符串起始位置
        start = 0
        # 遍历字符串
        for i in range(len(s)):
            # 如果元素已经在map，说明出现了重复元素，新起始点应该在上一次遇到这个元素位置后面一个位置
            # 还需要判断上一次遇到这个元素的位置要在start或之后，不然不成立（第二次遇到t，start在第二个m）
            # 长度不会变大，所以不用判断长度
            if s[i] in hashmap and start <= hashmap[s[i]]:
                start = hashmap[s[i]] + 1
                hashmap[s[i]] = i
            # 如果元素不在map，记录元素下标，更新最大值
            else:
                hashmap[s[i]] = i
                max_ = max(i-start+1, max_)
        return max_
```
最长回文子字符串
https://leetcode.com/problems/longest-palindromic-substring/
```
Input: s = "babad"
Output: "bab"
Explanation: "aba" is also a valid answer.
```
```
class Solution(object):
    def longestPalindrome(self, s):
        # 方法1：遍历每个字符，然后向两边扩散，检查是否成立回文字符串
        res = ""
        for i in range(len(s)):
            # 从一个字符扩散 "aba"
            tmp = self.helper(s, i, i)
            if len(tmp) > len(res):
                res = tmp
            # 从两个字符扩散 "abba"
            tmp = self.helper(s, i, i+1)
            # 更新为比较长的回文字符串
            if len(tmp) > len(res):
                res = tmp
        return res

    def helper(self, s, l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1; r += 1
        return s[l+1:r]

    # def longestPalindrome(self, s):
        # 方法2：动态规划。dp[i][j] 表示 s[i:j+1] 是否是回文字符串。2D数组，只用到右上三角形部分。空间复杂度高
        # 基础case：当i==j时，表示同一个字符，肯定是回文字符串。dp对角线全为true，j>=i，所以是右上部分
        # 状态转移方程：当s[i]==s[j]，两端字符一样，如果是相邻字符，或i和j之间的字符串 dp[i+1][j-1]也是回文的话，s[i:j+1]也是回文字符串，记录长度
        # 当s[i]!=s[j]，显然s[i:j+1]不是回文字符串，dp默认为false，不做处理
        # n = len(s)
        # dp = [[False for _ in range(n)] for _ in range(n)]
        # result = s[0]
        # # 因为是dp右上部分，动态转移方程用到了dp[i+1][j-1]（左下），所以从对角线从下往上遍历
        # # j为列，i为行
        # for j in range(n):
        #     for i in range(j, -1, -1):
        #         if i == j:
        #             dp[i][j] = True
        #         else:
        #             if s[i] == s[j] and (abs(j-i) == 1 or dp[i+1][j-1]):
        #                 dp[i][j] = True
        #                 if len(s[i:j+1]) > len(result):
        #                     result = s[i:j+1]
        #             # s[i] != s[j]默认为false
        # return result
        #     j=0    j=1    j=2   j=3    j=4
        #i=0[[True, False, True, False, False], 
        #i=1 [False, True, False, True, False], 
        #i=2 [False, False, True, False, False], 
        #i=3 [False, False, False, True, False], 
        #i=4 [False, False, False, False, True]]
        #s[0:3], "bab"是回文字符串；s[1:4], "aba"也是回文字符串，但是长度不大于"bab"，最后返回"bab"
```
锯齿状转化字符
https://leetcode.com/problems/zigzag-conversion/
```
Input: s = "PAYPALISHIRING", numRows = 3
Output: "PAHNAPLSIIGYIR"
P   A   H   N
 A P L S I I G
  Y   I   R
```
```
class Solution(object):
    def convert(self, s, numRows):
        # 遍历字符串，往下走把step设为1，往上走时step设为-1，row+=step就可以先增加后减少
        # 如果只有一行直接return
        if numRows == 1 or numRows > len(s):
            return s
        # row代表现在遍历的char应该在哪行，step代表是往下走（1）还是往上走（-1）
        row, step = 0, 1
        # 使用array存每一行的字符串，最后合并起来
        zigzag = ['' for _ in range(numRows)]
        for char in s:
            # 把字符加入该行
            zigzag[row] += char
            # 在第一行，需要往下走
            if row == 0:
                step = 1
            # 在最后一行，需要往上走
            elif row == numRows - 1:
                step = -1  
            # 在中间行，不用改变方向
            # 改变row值，往下或者往上走
            row += step
        return ''.join(zigzag)
```
最长公共前缀
https://leetcode.com/problems/longest-common-prefix/
```
Input: strs = ["flower","flow","flight"]
Output: "fl"
```
```
class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        # 纵向比较, 一个字符一个字符比较, 看成2D array
        res = ''
        if len(strs) == 0:
            return res
        # 使用第一个字符串与其他比较
        n = len(strs[0])
        for i in range(n):
            for j in range(1, len(strs)):
                # 当当前字符已经超过其他某个字符串的长度，或者遇到字符不相等的情况，就返回结果
                if i == len(strs[j]) or strs[0][i] != strs[j][i]:
                    return strs[0][:i]
        # 遍历结束，第一个字符串就是最长公共前缀
        return strs[0]
```
字符串中最短的子串包含另一字符串中的所有字符
https://leetcode.com/problems/minimum-window-substring/
```
Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t.
```
```
class Solution(object):
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        # 首先构建一个t_map，存t中的字符和字符数量
        # 然后使用两个指针i=0,j=0,指向滑动窗口两端，使用另一个window_map存当前滑动窗口中的字符及数量
        # j向右移动，遇到字符就加入window_map，我们想要window_map与t_map相同，但是window_map里面字符数量可能大于t_map，出现了两个B，比如：
        #  i    j
        # "ABOBECODEBANC"
        # 我们需要变量formed记录window满足了几个字符的数量，两个B>=一个B，也是满足。当前formed=3=t_map长度。form+=1条件是某个字符达到了t_map中的数量
        # formed达到3后，因为可能出现"AAABC"的情况，要最小长度，所以i要向右移，只要formed=3，就检查子串长度有没有更小
        # form-=1条件是i的字符在t_map中，并且window中这个字符数量小于t所需要的数量。直到formed<3，i停止右移，j开始右移，直到formed再次等于3
        t_map = {}
        for c in t:
            if c in t_map:
                t_map[c] += 1
            else:
                t_map[c] = 1
        window_map = {}
        formed = 0
        i = 0
        min_len = float('inf')
        res = (0, 0) # start, end
        # j右移，如果是t里面的字符，就加入window_map
        for j in range(len(s)):
            if s[j] in t_map:
                if s[j] in window_map:
                    window_map[s[j]] += 1
                else:
                    window_map[s[j]] = 1
                # 判断formed是否加一
                if window_map[s[j]] == t_map[s[j]]:
                    formed += 1
                # formed达到3，记录子串长度，i右移，从window_map移除一个，直到formed<3
                while formed == len(t_map) and i <= j:
                    # 更短子串
                    if j-i+1 < min_len:
                        min_len = j - i + 1
                        res = (i, j)
                    if s[i] in t_map:
                        window_map[s[i]] -= 1
                        # 判断formed是否减一
                        if window_map[s[i]] < t_map[s[i]]:
                            formed -= 1
                    i += 1
        return "" if min_len == float("inf") else s[res[0] : res[1] + 1]
```

## Map & Set
判断两个字符串是否为变位词
https://leetcode.com/problems/valid-anagram/
```
class Solution(object):
    def isAnagram(self, s, t):
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
把多个变位词分组
https://leetcode-cn.com/problems/group-anagrams/
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
## Heap 堆
合并k个排序的链表
https://leetcode.com/problems/merge-k-sorted-lists/
```
Input: lists = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]
Explanation: The linked-lists are:
[
  1->4->5,
  1->3->4,
  2->6
]
merging them into one sorted list:
1->1->2->3->4->4->5->6
```
```
class Solution(object):
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        # 建立一个大小为len(lists)的最小堆
        # 最开始放入第一列的元素，弹出最小值，然后继续放入最小值后面的元素
        # 最小堆里面需要存节点的值和节点，才能找到最小值后面的元素
        from heapq import heappush, heappop 
        # 假数据头
        dummy = curr = ListNode()
        heap = [] # heap由一个数组实现
        # 以第一列元素建立最小堆
        for head in lists:
            if head:
                heappush(heap, (head.val, head))
        # 弹出最小值直到最小堆为空
        while heap:
            min_node = heappop(heap)[1]
            # 最小值节点放入新链表
            curr.next = min_node
            curr = curr.next
            # 如果下一个不为空，继续放入下一个元素
            if min_node.next:
                heappush(heap, (min_node.next.val, min_node.next))
        return dummy.next
```
## Stack & Set
字符中串移除相邻字符
https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string/
```
class Solution(object):
    def removeDuplicates(self, S):
        # 使用栈
        stack = []
        for s in S:
            stack.append(s)
            if len(stack) > 1 and stack[-1] == stack[-2]:
                stack.pop()
                stack.pop()
        return ''.join(stack)
```
去掉外层括号
https://leetcode.com/problems/remove-outermost-parentheses/
```
class Solution(object):
    def removeOuterParentheses(self, S):
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
括号串中最长合法括号（困难）
https://leetcode.com/problems/longest-valid-parentheses/
```
Input: s = ")()())"
Output: 4
Explanation: The longest valid parentheses substring is "()()".
```
```
class Solution(object):
    def longestValidParentheses(self, s):
        """
        :type s: str
        :rtype: int
        """
        # 更好的例子："(())(())"，"(()((())"  
        # 方法1：暴力O(n^3)超时。判断每个子串是否是合法的括号，使用栈，同时可以剪枝
        
        # 方法2: 动态规划O(n)。dp[i]表示以i位置结尾能形成的最长括号长度
        # 以"("结尾不可能形成，所以为0； 如果以")"结尾，dp[i] = 2+dp[i-1]+dp[i-dp[i-1]-2]
        # 比如i最后一个位置")",dp[i-1]为2，因为中间有一对括号，对称位置是i-dp[i-1]-1，刚好为"("，所以加上中间一对括号，dp[i]结果暂时为2+2=4
        # 还要看对称位置前面一个位置i-dp[i-1]-2是否能和之后的形成合法括号。
        # 第一个例子里dp[i-dp[i-1]-2]为4，所以dp[i]结果为4+4=8。第一个例子里dp[i-dp[i-1]-2]为0，所以dp[i]结果为4
        # [0, 0, 2, 4, 0, 0, 2, 8] and [0, 0, 2, 0, 0, 0, 2, 4]
        # n = len(s)
        # if n == 0:
        #     return 0
        # dp = [0] * n
        # for i in range(n):
        #      if s[i] == ")" and i-dp[i-1]-1 >= 0 and s[i-dp[i-1]-1] == "(":
        #             dp[i] = 2+dp[i-1]+dp[i-dp[i-1]-2]
        # return max(dp)
        
        # 方法3: 使用栈O(n)，类似方法1，思想：如果全部是合法括号，最后栈会为空，所以栈里面剩下的元素就是不合法的位置
        # 以"(()((())"为例，0，1放入栈，2时弹出1，剩下0，说明0位置暂时不合法，0位置是合法括号的前一个位置，所以弹出时需要计算合法长度
        # 为了统一计算，避免最开始括号就是合法导致栈为空，无法计算长度，栈最开始放入-1，方便之后计算合法括号长度
        # 栈里长住元素代表合法括号开始的前一个位置，栈里短住数字代表最合法括号中的左括号的位置
        # 当遇到左括号就把位置加入栈中，右括号就弹出栈顶元素，如果弹出后栈为空，代表此右括号为新的长住，如果不为空，就计算长度
        # '(', [-1, 0]，左括号
        # '((', [-1, 0, 1]，左括号
        # '(()', [-1, 0]，右括号，弹出栈顶，长度为2-0=2，更新
        # '(()(', [-1, 0, 3]，左括号
        # '(()((', [-1, 0, 3, 4]，左括号
        # '(()(((', [-1, 0, 3, 4, 5]，左括号
        # '(()((()', [-1, 0, 3, 4]，右括号，弹出栈顶，长度为6-4=2，不更新
        # '(()((())', [-1, 0, 3]，右括号，弹出栈顶，长度为7-3=4，更新
        stack = [-1]
        res = 0
        for i in range(len(s)):
            if s[i] == "(":
                stack.append(i)
            else:
                # 弹出左括号，如果栈为空说明再次出现不合法的")"，加入栈
                stack.pop()
                if not stack:
                    stack.append(i)
                else:
                    # 长度为当前位置减去栈顶位置，因为栈顶位置是匹配好括号的前一位
                    res = max(res, i - stack[-1])
        return res
        
        # 方法4:从左往右和从右往左遍历，找到最大长度，O(2n)
        # 先从左往右，记录左右括号的个数，只要右括号大于左括号，就不合法，个数清零；如果左右括号个数相同，说明合法，记录长度，但并不清零，因为后面还可能合法
        # 然后从右往左，相反的，只要左括号大于右括号，就不合法，其他一样
        # res = 0
        # left, right = 0, 0
        # for i in range(len(s)):
        #     if s[i] == '(':
        #         left += 1
        #     else:
        #         right += 1
        #     if left == right:
        #         res = max(res, 2*right)
        #     elif right >= left:
        #         left = right = 0
        # left, right = 0, 0
        # for i in range(len(s)-1, -1, -1):
        #     if s[i] == '(':
        #         left += 1
        #     else:
        #         right += 1
        #     if left == right:
        #         res = max(res, 2*left)
        #     elif left >= right:
        #         left = right = 0
        # return res

```
直方图中最大能形成的长方形
https://leetcode.com/problems/largest-rectangle-in-histogram/
```
Input: heights = [2,1,5,6,2,3]
Output: 10
Explanation: The above is a histogram where width of each bar is 1.
The largest rectangle is shown in the red area, which has an area = 10 units.
```
```
class Solution(object):
    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        # 1.暴力解法(超时)：从每根柱子向两边扩散 O(n^2),找到两边第一个比当前柱子矮的位置
        # 左右两端高度0的柱子
        # heights.append(0)  
        # n = len(heights)
        # left = [-1]*(n-1)
        # right = [n-1]*(n-1)
        # res = 0
        # for i in range(n):
        #     # find left
        #     for l in range(i-1, -1, -1): 
        #         if heights[l] < heights[i]:
        #             left[i] = l
        #             break
        #     # find right
        #     for r in range(i+1, n): 
        #         if heights[i] > heights[r]:
        #             right[i] = r
        #             res = max(res, heights[i]*(right[i]-left[i]-1))
        #             break
        # # print(left, right) # ([-1, -1, 1, 2, 1, 4], [1, 6, 4, 4, 6, 6])
        # return res
        
        # 2.stack: 构建一个不严格递增的栈，存放柱子的index。遇到比栈顶矮的柱子，栈顶高度形成的矩形就确定了
        # 高度为栈顶柱子高度，宽度是当前i柱子到栈顶前一个柱子stack[-2]的距离
        heights.append(0) 
        stack = [-1] # 设为-1相当于在前后各增加一个隐形柱子便于计算，第一个隐形柱子永远在栈底
        ans = 0
        for i in xrange(len(heights)):
            # 第一个柱子永远不小于隐形柱子，所以直接加入stack
            # 当遇到比栈顶矮的柱子，比如第五根2柱子，高度为6的柱子形成的最大矩形就确定了，为6
            # 去掉6之后同时高度为5的柱子最大矩形也能确定了，宽度为i-2 = 2，面积为10
            # 去掉5之后高度为1的柱子不能确定，插入高度为2的柱子，继续向右走
            # 最后，栈剩下[-1, 1(1), 4(2), 5(3) ]。因为有高度为0的隐形柱子，所以能消耗完栈里面留下的柱子，留下隐形柱子
            while heights[i] < heights[stack[-1]]:
                # 注意需要先pop再计算宽度，因为宽度不是当前柱子到栈顶柱子的距离，而是到栈顶前一个柱子stack[-2]的距离
                # 比如当当前柱子为0，3柱子确定面积为3，2柱子确定面积的宽度为1柱子到0柱子的距离
                h = heights[stack.pop()]
                w = i - stack[-1] - 1
                ans = max(ans, h * w)
            stack.append(i)  
        heights.pop()
        return ans
```
洼地能装的雨水量
https://leetcode.com/problems/trapping-rain-water/
```
class Solution(object):
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        # materarea[i] = min(leftmaxheight, rightmaxheight) - height[i], so we can expand to left and right, O(n^2)
        # we can traverse from left to right to record leftmaxheight for every block, and right to left to record rightmaxheight
        # reduce to O(n)
        n = len(height)
        if n == 0:
            return 0
        leftmaxheight = height[0]
        leftmaxheights = [height[0]]
        rightmaxheight = height[-1]
        rightmaxheights = [height[-1]]
        for i in range(1, n): # first height cannot store water
            leftmaxheights.append(leftmaxheight)
            leftmaxheight = max(height[i], leftmaxheight)
        reversed_height = height[::-1]
        for i in range(n-1, -1, -1): # last height cannot store water
            rightmaxheights.append(rightmaxheight)
            rightmaxheight = max(height[i], rightmaxheight) 
        rightmaxheights = rightmaxheights[::-1]
        # print(leftmaxheights, rightmaxheights) # ([0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 1, 1])
        # e.g. height[2] has leftmax 1, rightmax, 3, so min(1, 3) = 1, water area = 1
        # sum all water area
        res = 0
        for i in range(1, n-1):
            bottleneck = min(leftmaxheights[i], rightmaxheights[i])
            if bottleneck > height[i]: # can store water
                res += bottleneck - height[i]
        return res
```
## 动态规划
从左上角到右下角不同的走法
https://leetcode.com/problems/unique-paths-ii/
```
Input: obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
Output: 2
Explanation: There is one obstacle in the middle of the 3x3 grid above.
```
```
class Solution(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        # 特殊情况：1. 障碍在第一行或第一列，右边或下边的格子都为0 2.障碍在起点或终点，返回0
        if obstacleGrid[0][0] == 1 or obstacleGrid[-1][-1] == 1: return 0
        dp = [[0 for _ in range(len(obstacleGrid[0]))] for _ in range(len(obstacleGrid))]
        # 判断第一列有没有障碍
        for i in range(len(obstacleGrid)):
            if obstacleGrid[i][0] == 0:
                dp[i][0] = 1 
            else:
                break
        # 判断第一行有没有障碍
        for j in range(len(obstacleGrid[0])):
            if obstacleGrid[0][j] == 0:
                dp[0][j] = 1
            else:
                break
        # 其他行列
        for i in range(1, len(obstacleGrid)):
            for j in range(1, len(obstacleGrid[0])):
                # 如果遇到障碍，默认为0
                if obstacleGrid[i][j] == 0:
                    dp[i][j] = dp[i-1][j] + dp[i][j-1]
        # print(dp)
        # [1, 1, 1]
        # [1, 0, 1]
        # [1, 1, 2]
        return dp[-1][-1]
```
地牢游戏
https://leetcode.com/problems/dungeon-game/
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
## 递归 Recursion
数字的所有组合方式（中等）
https://leetcode.com/problems/combinations/
```
Input: n = 4, k = 2
Output:
[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]
```
```
class Solution(object):
    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """
        # [1,2,3,4],先选1，然后再依次选剩下的数，然后选2，然后选2后面的数。。。
        nums = [i+1 for i in range(n)]
        self.k = k
        self.res = []
        # 需要传入剩下的数nums，当前已经选好的数
        self.helper(nums, [])
        return self.res
    
    def helper(self, nums, curr):
        if len(curr) == self.k:
            self.res.append(curr)
            return
        # 开始依次选，分支出去
        for i in range(len(nums)):
            self.helper(nums[i+1:], curr+[nums[i]])
        
```
数字的所有排列方式（中等）
https://leetcode.com/problems/permutations/
```
Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```
```
class Solution(object):
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        # 回溯算法O(2^n)指数型时间复杂度
        # 依次从nums里选一个数字，然后依次再从剩下的数字里选一个数字。。。不断递归分支出去，记录剩下的nums和已经选好的数字curr
        self.res = []
        self.helper(nums, [])
        return self.res
        
    def helper(self, nums, curr):
        # 数字选完就跳出递归，加入结果
        if len(nums) == 0:
            self.res.append(curr)
        # 依次从剩下的数字中选择。先选1，剩下[2,3]，然后分支得到[1,2],[1,3]，最后分别剩下3，2，分支得到[[1,2,3],[1,3,2]
        # 从1开始的就选完了，开始选择2，剩下[1,3]......
        for i in range(len(nums)):
            self.helper(nums[:i]+nums[i+1:], curr+[nums[i]])
```
n皇后问题（困难）
https://leetcode.com/problems/n-queens/
```
Input: n = 4
Output: [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
Explanation: There exist two distinct solutions to the 4-queens puzzle as shown above
```
```
class Solution(object):
    def solveNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        # 回溯算法，指数型时间复杂度。模拟放棋子的过程，对每行每一格每一格的试，放了一行放下一行，判断这样放是否合法
        self.res = []
        self.board = [['.' for _ in range(n)] for _ in range(n)]
        # 从第一行开始放
        self.dfs(0)
        return self.res
    
    def dfs(self, row):
        if row == len(self.board):
            # 已经放完最后一行，到达这里表示前面放法都合法，加入结果
            temp = []
            for i in range(len(self.board)):
                temp.append(''.join(self.board[i]))
            self.res.append(temp)
            return
        # 当前正在放第row行的queen，遍历每列（每格）
        for col in range(len(self.board[0])):
            # 判断放法是否合法，如果合法就放下一行
            if self.isValid(row, col):
                self.board[row][col] = 'Q'
                # 分支出去下一行
                self.dfs(row + 1)
                # 复原棋盘，尝试下一个格子
                self.board[row][col] = '.'
    
    def isValid(self, row, col):
        # 遍历当前board中已经放好的queen（棋盘上面部分），看当前要放的queen位置是否合法
        # 1. 判断当前列是否有queen
        for i in range(row):
            if self.board[i][col] == 'Q': return False
        # 2. 判断当前位置往左对角线是否有queen
        for i, j in zip(range(row-1, -1, -1), range(col-1, -1, -1)):
            if self.board[i][j] == 'Q': return False
        # 3. 判断当前位置往右对角线是否有queen
        for i, j in zip(range(row-1, -1, -1), range(col+1, len(self.board[0]))):
            if self.board[i][j] == 'Q': return False
        return True
```

## 二叉树 Binary tree
二叉树最大深度
https://leetcode.com/problems/maximum-depth-of-binary-tree/
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
判断二叉树是否对称
https://leetcode-cn.com/problems/symmetric-tree/
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
二叉树最小深度
https://leetcode.com/problems/minimum-depth-of-binary-tree/
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
二叉树两节点值最小的差值
https://leetcode.com/problems/minimum-distance-between-bst-nodes/
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
输出二叉树的所有路线
https://leetcode.com/problems/binary-tree-paths/
```
class Solution(object):
    def binaryTreePaths(self, root):
        """
        :type root: TreeNode
        :rtype: List[str]
        """
        self.res = []
        self.helper(root, '')
        return self.res
    
    def helper(self, node, path):
        # 如果不是叶子结点，append node.val数值，包含->,往左右递归
        # 如果是叶子结点，append node.val数值，就不包含->, 放入res
        if node.left:  
            self.helper(node.left, path + str(node.val) + '->')
        if node.right:  
            self.helper(node.right, path + str(node.val) + '->')
        if not node.left and not node.right:
            path += str(node.val)
            self.res.append(path)
```
https://leetcode.com/problems/range-sum-of-bst/
```
class Solution(object): 
      def rangeSumBST(self, root, L, R):
        def dfs(node):
            if node:
                if L <= node.val <= R:
                    self.ans += node.val
                if L < node.val:
                    dfs(node.left)
                if node.val < R:
                    dfs(node.right)

        self.ans = 0
        dfs(root)
        return self.ans  
```
二叉树两个节点最近公共祖先
https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/
```
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root:
            return
        if root == p or root == q:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p , q)
        if left and right:
            return root
        return left or right
```
