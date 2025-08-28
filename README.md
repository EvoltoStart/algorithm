子串
560. 和为 K 的子数组 - 力扣（LeetCode）（中等）
给你一个整数数组 nums 和一个整数 k ，请你统计并返回 该数组中和为 k 的子数组的个数 。
子数组是数组中元素的连续非空序列。
示例 1：
输入：nums = [1,1,1], k = 2
输出：2
示例 2：
输入：nums = [1,2,3], k = 3
输出：2
func subarraySum(nums []int, k int) int {
    pre,ans:=0,0
    mp:=make(map[int]int,len(nums))
    for _,n:=range nums{
        mp[pre]++
        ans+=mp[pre-k]
        pre+=n
    }
    return ans
}
使用前缀和，pre-k~pre之间即为一个符合要求的和，将pre存入map，如果pre-k在map存在，value+1（pre-k有多个）
239. 滑动窗口最大值 - 力扣（LeetCode）（困难）
给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。
返回 滑动窗口中的最大值 。
示例 1：
输入：nums = [1,3,-1,-3,5,3,6,7], k = 3
输出：[3,3,5,5,6,7]
解释：
滑动窗口的位置                最大值
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
示例 2：
输入：nums = [1], k = 1
输出：[1]
func maxSlidingWindow(nums []int, k int) []int {
    ans := make([]int, len(nums)-k+1) // 窗口个数
    q := []int{}

    for i, x := range nums {
        // 1. 右边入
        for len(q) > 0 && nums[q[len(q)-1]] <= x {
            q = q[:len(q)-1] // 维护 q 的单调性
        }
        q = append(q, i)

        // 2. 左边出
        left := i - k + 1 // 窗口左端点
        if q[0] < left {  // 队首已经离开窗口了，left窗口左端点，q[0]窗口最大的元素
            q = q[1:] // Go 的切片是 O(1) 的
        }

        // 3. 在窗口左端点处记录答案
        if left >= 0 { //相当于进入第二个窗口，求第一个窗口的最大值
            // 由于队首到队尾单调递减，所以窗口最大值就在队首
            ans[left] = nums[q[0]]
        }
    }

    return ans
}
使用单调队列，存的值为数组下标，要插入的值如果比队尾大，队尾元素移除，否则添加到队尾；left := i - k + 1为窗口左端点，q[0]为窗口最大值的数组下标测试
76. 最小覆盖子串（困难）
给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。
注意：
● 对于 t 中重复字符，我们寻找的子字符串中该字符数量必须不少于 t 中该字符数量。
● 如果 s 中存在这样的子串，我们保证它是唯一的答案。
示例 1：
输入：s = "ADOBECODEBANC", t = "ABC"
输出："BANC"
解释：最小覆盖子串 "BANC" 包含来自字符串 t 的 'A'、'B' 和 'C'。
示例 2：
输入：s = "a", t = "a"
输出："a"
解释：整个字符串 s 是最小覆盖子串。
示例 3:
输入: s = "a", t = "aa"
输出: ""
解释: t 中两个字符 'a' 均应包含在 s 的子串中，
因此没有符合条件的子字符串，返回空字符串。
func minWindow(s, t string) string {
    cnt := [128]int{}
    less := 0
    for _, c := range t {
        if cnt[c] == 0 {
            less++ // 有 less 种字母的出现次数 < t 中的字母出现次数(如果有重复字符)
        }
        cnt[c]++
    }

    ansLeft, ansRight := -1, len(s)
    left := 0
    for right, c := range s { // 移动子串右端点
        cnt[c]-- // 右端点字母移入子串
        if cnt[c] == 0 {
            // 原来窗口内 c 的出现次数比 t 的少，现在一样多
            less--
        }
        for less == 0 { // 涵盖：所有字母的出现次数都是 >=
            if right-left < ansRight-ansLeft { // 找到更短的子串
                ansLeft, ansRight = left, right // 记录此时的左右端点
            }
            x := s[left] // 左端点字母
            if cnt[x] == 0 {
                // x 移出窗口之前，检查出现次数，
                // 如果窗口内 x 的出现次数和 t 一样，
                // 那么 x 移出窗口后，窗口内 x 的出现次数比 t 的少
                less++
            }
            cnt[x]++ // 左端点字母移出子串
            left++
        }
    }
    if ansLeft < 0 {
        return ""
    }
    return s[ansLeft : ansRight+1]
}
less用来表示子串中是否涵盖t中所有字符，初始化 less 为 t 中的不同字母个数，如果 less=0，说明 cntS 中的每个字母及其出现次数都大于等于 cntT 中的字母出现次数（t中字符都出现在子串了）；cnt[c]==0可以表示子串涵盖字符c，cnt[c]<0表示不相关的字符，cnt[c]>0表示t中字符还没有完全包含到子串中
普通数组
53. 最大子数组和（中等）
给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
子数组是数组中的一个连续部分。
示例 1：
输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
输出：6
解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。
示例 2：
输入：nums = [1]
输出：1
示例 3：
输入：nums = [5,4,-1,7,8]
输出：23
func maxSubArray(nums []int) int {
    pre,ans:=0,nums[0]
    for _,i:=range nums{
        pre=max(pre+i,i)
        ans=max(ans,pre)
    }
    return ans
}
前缀和，如果pre+要加的元素i<i,那么子数组肯定不会是最大的，就从i从新开始累加，ans选择最大的pre
56. 合并区间（中等）
以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。请你合并所有重叠的区间，并返回 一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间 。
示例 1：
输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
输出：[[1,6],[8,10],[15,18]]
解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
示例 2：
输入：intervals = [[1,4],[4,5]]
输出：[[1,5]]
解释：区间 [1,4] 和 [4,5] 可被视为重叠区间。
func merge(intervals [][]int) [][]int {
    sort.Slice(intervals,func(i,j int)bool{return intervals[i][0]<intervals[j][0]})

    merge:=[][]int{
    
    }
    for _,interval:=range intervals{
        n:=len(merge)
        if n==0||interval[0]>merge[n-1][1]{
            merge=append(merge,interval)
        }else{
            merge[n-1][1]=max(merge[n-1][1],interval[1])
        }
    }
    return merge
}

首先将区间按第一个元素排序，合并区间时比较interval[0]大于merge[n-1][1](没有重复元素不能合并)，否则选择两个数组最大的当尾巴
189. 轮转数组（中等）
给定一个整数数组 nums，将数组中的元素向右轮转 k 个位置，其中 k 是非负数。
示例 1:
输入: nums = [1,2,3,4,5,6,7], k = 3
输出: [5,6,7,1,2,3,4]
解释:
向右轮转 1 步: [7,1,2,3,4,5,6]
向右轮转 2 步: [6,7,1,2,3,4,5]
向右轮转 3 步: [5,6,7,1,2,3,4]
示例 2:
输入：nums = [-1,-100,3,99], k = 2
输出：[3,99,-1,-100]
解释: 
向右轮转 1 步: [99,-1,-100,3]
向右轮转 2 步: [3,99,-1,-100]
func rotate(nums []int, k int)  {
    n:=len(nums)
    k%=n
    slices.Reverse(nums)
    slices.Reverse(nums[:k])
    slices.Reverse(nums[k:])

}
238. 除自身以外数组的乘积（中等）
给你一个整数数组 nums，返回 数组 answer ，其中 answer[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积 。
题目数据 保证 数组 nums之中任意元素的全部前缀元素和后缀的乘积都在  32 位 整数范围内。
请 不要使用除法，且在 O(n) 时间复杂度内完成此题。
示例 1:
输入: nums = [1,2,3,4]
输出: [24,12,8,6]
示例 2:
输入: nums = [-1,1,0,-3,3]
输出: [0,0,9,0,0]
func productExceptSelf(nums []int) []int {
    length := len(nums)
    answer := make([]int, length)

    // answer[i] 表示索引 i 左侧所有元素的乘积
    // 因为索引为 '0' 的元素左侧没有元素， 所以 answer[0] = 1
    answer[0] = 1
    for i := 1; i < length; i++ {
        answer[i] = nums[i-1] * answer[i-1]
    }

    // R 为右侧所有元素的乘积
    // 刚开始右边没有元素，所以 R = 1
    R := 1
    for i := length - 1; i >= 0; i-- {
        // 对于索引 i，左边的乘积为 answer[i]，右边的乘积为 R
        answer[i] = answer[i] * R
        // R 需要包含右边所有的乘积，所以计算下一个结果时需要将当前值乘到 R 上
        R *= nums[i]
    }
    return answer
}
41. 缺失的第一个正数（困难）
给你一个未排序的整数数组 nums ，请你找出其中没有出现的最小的正整数。
请你实现时间复杂度为 O(n) 并且只使用常数级别额外空间的解决方案。
示例 1：
输入：nums = [1,2,0]
输出：3
解释：范围 [1,2] 中的数字都在数组中。
示例 2：
输入：nums = [3,4,-1,1]
输出：2
解释：1 在数组中，但 2 没有。
示例 3：
输入：nums = [7,8,9,11,12]
输出：1
解释：最小的正数 1 没有出现。
func firstMissingPositive(nums []int) int {
    n := len(nums)
    for i := range n {
        // 如果当前学生的学号在 [1,n] 中，但（真身）没有坐在正确的座位上
        for 1 <= nums[i] && nums[i] <= n && nums[i] != nums[nums[i]-1] {
            // 那么就交换 nums[i] 和 nums[j]，其中 j 是 i 的学号
            j := nums[i] - 1 // 减一是因为数组下标从 0 开始
            nums[i], nums[j] = nums[j], nums[i]
        }
    }

    // 找第一个学号与座位编号不匹配的学生
    for i := range n {
        if nums[i] != i+1 {
            return i + 1
        }
    }

    // 所有学生都坐在正确的座位上
    return n + 1
}
如果缺失的第一个正数在1~n,用数组下标和对应的值比较，如果相等说明存在，不相等则找到；不在1~n，直接+1；为了让下标和值可能相等，将nun[i]和num[num[i]-1]互换
矩阵
73. 矩阵置零（中等）
矩给定一个 m x n 的矩阵，如果一个元素为 0 ，则将其所在行和列的所有元素都设为 0 。请使用 原地 算法。
示例 1：

输入：matrix = [[1,1,1],[1,0,1],[1,1,1]]
输出：[[1,0,1],[0,0,0],[1,0,1]]
示例 2：

输入：matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
输出：[[0,0,0,0],[0,4,5,0],[0,3,1,0]]
func setZeroes(matrix [][]int) {
	row, col := len(matrix), len(matrix[0])
	row_flag, col_flag := false, false
	//判断第一行是否包含0
	for i := 0; i < col; i++ {
		if matrix[0][i] == 0 {
			row_flag = true
			break
		}
	}
	//判断第一列是否包含0
	for i := 0; i < row; i++ {
		if matrix[i][0] == 0 {
			col_flag = true
			break
		}
	}
	//判断matrix[i][j]是否为0，将第一行和第一列对应标志位置0
	for i := 1; i < row; i++ {
		for j := 1; j < col; j++ {
			if matrix[i][j] == 0 {
				matrix[0][j], matrix[i][0] = 0, 0
			}
		}
	}
	//置零
	for i := 1; i < row; i++ {
		for j := 1; j < col; j++ {
			if matrix[0][j] == 0 || matrix[i][0] == 0 {
				matrix[i][j] = 0
			}
		}
	}
	if row_flag {
		for i := 0; i < col; i++ {
			matrix[0][i] = 0
		}
	}
	if col_flag {
		for i := 0; i < row; i++ {
			matrix[i][0] = 0
		}
	}
}
54. 螺旋矩阵
给你一个 m 行 n 列的矩阵 matrix ，请按照 顺时针螺旋顺序 ，返回矩阵中的所有元素。
示例 1：

输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[1,2,3,6,9,8,7,4,5]
示例 2：

输入：matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
输出：[1,2,3,4,8,12,11,10,9,5,6,7]
var dirs = [4][2]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}} // 右 下 左 上

func spiralOrder(matrix [][]int) []int {
    m, n := len(matrix), len(matrix[0])
    ans := make([]int, 0, m*n)
    i, j := 0, -1 // 从 (0, -1) 开始
    for di := 0; len(ans) < cap(ans); di = (di + 1) % 4 {
        for range n { // 走 n 步（注意 n 会减少）
            i += dirs[di][0]
            j += dirs[di][1] // 先走一步
            ans = append(ans, matrix[i][j]) // 再加入答案
        }
        n, m = m-1, n // 减少后面的循环次数
    }
    return ans
}
定义dirs数组表示移动方向，i,j就按照dirs的方向依次加
48. 旋转图像
给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。
你必须在原地 旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要 使用另一个矩阵来旋转图像。
示例 1：

输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[[7,4,1],[8,5,2],[9,6,3]]
示例 2：

输入：matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
输出：[[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]
func rotate(matrix [][]int) {
    n := len(matrix)
    // 第一步：转置
    for i := range n {
        for j := range i { // 遍历对角线下方元素
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        }
    }

    // 第二步：行翻转
    for _, row := range matrix {
        slices.Reverse(row)
    }
}


240. 搜索二维矩阵 II
编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性：
● 每行的元素从左到右升序排列。
● 每列的元素从上到下升序排列。
示例 1：

输入：matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 5
输出：true
示例 2：

输入：matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 20
输出：false
func searchMatrix(matrix [][]int, target int) bool {
    m, n := len(matrix), len(matrix[0])
    i, j := 0, n-1 // 从右上角开始
    for i < m && j >= 0 { // 还有剩余元素
        if matrix[i][j] == target {
            return true // 找到 target
        }
        if matrix[i][j] < target {
            i++ // 这一行剩余元素全部小于 target，排除
        } else {
            j-- // 这一列剩余元素全部大于 target，排除
        }
    }
    return false
}

链表
160. 相交链表（简单）
给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 null 。
图示两个链表在节点 c1 开始相交：

题目数据 保证 整个链式结构中不存在环。
注意，函数返回结果后，链表必须 保持其原始结构 。
自定义评测：
评测系统 的输入如下（你设计的程序 不适用 此输入）：
● intersectVal - 相交的起始节点的值。如果不存在相交节点，这一值为 0
● listA - 第一个链表
● listB - 第二个链表
● skipA - 在 listA 中（从头节点开始）跳到交叉节点的节点数
● skipB - 在 listB 中（从头节点开始）跳到交叉节点的节点数
评测系统将根据这些输入创建链式数据结构，并将两个头节点 headA 和 headB 传递给你的程序。如果程序能够正确返回相交节点，那么你的解决方案将被 视作正确答案 。

示例 1：

输入：intersectVal = 8, listA = [4,1,8,4,5], listB = [5,6,1,8,4,5], skipA = 2, skipB = 3
输出：Intersected at '8'
解释：相交节点的值为 8 （注意，如果两个链表相交则不能为 0）。
从各自的表头开始算起，链表 A 为 [4,1,8,4,5]，链表 B 为 [5,6,1,8,4,5]。
在 A 中，相交节点前有 2 个节点；在 B 中，相交节点前有 3 个节点。
— 请注意相交节点的值不为 1，因为在链表 A 和链表 B 之中值为 1 的节点 (A 中第二个节点和 B 中第三个节点) 是不同的节点。换句话说，它们在内存中指向两个不同的位置，而链表 A 和链表 B 中值为 8 的节点 (A 中第三个节点，B 中第四个节点) 在内存中指向相同的位置。

示例 2：

输入：intersectVal = 2, listA = [1,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
输出：Intersected at '2'
解释：相交节点的值为 2 （注意，如果两个链表相交则不能为 0）。
从各自的表头开始算起，链表 A 为 [1,9,1,2,4]，链表 B 为 [3,2,4]。
在 A 中，相交节点前有 3 个节点；在 B 中，相交节点前有 1 个节点。
示例 3：

输入：intersectVal = 0, listA = [2,6,4], listB = [1,5], skipA = 3, skipB = 2
输出：No intersection
解释：从各自的表头开始算起，链表 A 为 [2,6,4]，链表 B 为 [1,5]。
由于这两个链表不相交，所以 intersectVal 必须为 0，而 skipA 和 skipB 可以是任意值。
这两个链表不相交，因此返回 null 。
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func getIntersectionNode(headA, headB *ListNode) *ListNode {
    p,q:=headA,headB
    for p!=q{
        if p!=nil{
            p=p.Next
        }else{
            p=headB
        }
        if q!=nil{
            q=q.Next
        }else{
            q=headA
        }
        
    }
    return p
}
A和B先遍历，遍历完后换成对方遍历，直到p=q
206. 反转链表（简单）
给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。
示例 1：

输入：head = [1,2,3,4,5]
输出：[5,4,3,2,1]
示例 2：

输入：head = [1,2]
输出：[2,1]
示例 3：
输入：head = []
输出：[]

提示：
● 链表中节点的数目范围是 [0, 5000]
● -5000 <= Node.val <= 5000
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func reverseList(head *ListNode) *ListNode {
    var pre,cur *ListNode=nil,head
    for cur!=nil{
        next:=cur.Next
        cur.Next=pre
        pre=cur
        cur=next
    }
    return pre
}
头插法
234. 回文链表（简单）
给你一个单链表的头节点 head ，请你判断该链表是否为回文链表。如果是，返回 true ；否则，返回 false 。
示例 1：

输入：head = [1,2,2,1]
输出：true
示例 2：

输入：head = [1,2]
输出：false
// 876. 链表的中间结点
func middleNode(head *ListNode) *ListNode {
    slow, fast := head, head
    for fast != nil && fast.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
    }
    return slow
}

// 206. 反转链表
func reverseList(head *ListNode) *ListNode {
    var pre, cur *ListNode = nil, head
    for cur != nil {
        nxt := cur.Next
        cur.Next = pre
        pre = cur
        cur = nxt
    }
    return pre
}

func isPalindrome(head *ListNode) bool {
    mid := middleNode(head)
    head2 := reverseList(mid)
    for head2 != nil {
        if head.Val != head2.Val { // 不是回文链表
            return false
        }
        head = head.Next
        head2 = head2.Next
    }
    return true
}
将链表从中间分开，将后面的一段链表反转与前面进行比较
141. 环形链表（简单）
给你一个链表的头节点 head ，判断链表中是否有环。
如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。注意：pos 不作为参数进行传递 。仅仅是为了标识链表的实际情况。
如果链表中存在环 ，则返回 true 。 否则，返回 false 。
示例 1：

输入：head = [3,2,0,-4], pos = 1
输出：true
解释：链表中有一个环，其尾部连接到第二个节点。
示例 2：

输入：head = [1,2], pos = 0
输出：true
解释：链表中有一个环，其尾部连接到第一个节点。
示例 3：

输入：head = [1], pos = -1
输出：false
解释：链表中没有环。
func hasCycle(head *ListNode) bool {
    slow, fast := head, head // 乌龟和兔子同时从起点出发
    for fast != nil && fast.Next != nil {
        slow = slow.Next // 乌龟走一步
        fast = fast.Next.Next // 兔子走两步
        if fast == slow { // 兔子追上乌龟（套圈），说明有环
            return true
        }
    }
    return false // 访问到了链表末尾，无环
}
142. 环形链表 II（中等）
给定一个链表的头节点  head ，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。
如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。如果 pos 是 -1，则在该链表中没有环。注意：pos 不作为参数进行传递，仅仅是为了标识链表的实际情况。
不允许修改 链表。

示例 1：

输入：head = [3,2,0,-4], pos = 1
输出：返回索引为 1 的链表节点
解释：链表中有一个环，其尾部连接到第二个节点。
示例 2：

输入：head = [1,2], pos = 0
输出：返回索引为 0 的链表节点
解释：链表中有一个环，其尾部连接到第一个节点。
示例 3：

输入：head = [1], pos = -1
输出：返回 null
解释：链表中没有环。
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func detectCycle(head *ListNode) *ListNode {
    slow,fast:=head,head
    for fast!=nil&&fast.Next!=nil{
        slow=slow.Next
        fast=fast.Next.Next
        if slow==fast{
            for slow!=head{
                slow=slow.Next
                head=head.Next
            }
            return slow
        }
    }
    return nil
}
当slow和fast相遇后，slow和head一起前进，如果slow=head，那么说明slow就是入环的第一个节点
21. 合并两个有序链表（简单）
将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 

示例 1：

输入：l1 = [1,2,4], l2 = [1,3,4]
输出：[1,1,2,3,4,4]
示例 2：
输入：l1 = [], l2 = []
输出：[]
示例 3：
输入：l1 = [], l2 = [0]
输出：[0]
func mergeTwoLists(list1, list2 *ListNode) *ListNode {
    dummy := ListNode{} // 用哨兵节点简化代码逻辑
    cur := &dummy // cur 指向新链表的末尾
    for list1 != nil && list2 != nil {
        if list1.Val < list2.Val {
            cur.Next = list1 // 把 list1 加到新链表中
            list1 = list1.Next
        } else { // 注：相等的情况加哪个节点都是可以的
            cur.Next = list2 // 把 list2 加到新链表中
            list2 = list2.Next
        }
        cur = cur.Next
    }
    // 拼接剩余链表
    if list1 != nil {
        cur.Next = list1
    } else {
        cur.Next = list2
    }
    return dummy.Next
}
2. 两数相加(中等)
给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。
请你将两个数相加，并以相同形式返回一个表示和的链表。
你可以假设除了数字 0 之外，这两个数都不会以 0 开头。

示例 1：
输入：l1 = [2,4,3], l2 = [5,6,4]
输出：[7,0,8]
解释：342 + 465 = 807.
示例 2：
输入：l1 = [0], l2 = [0]
输出：[0]
示例 3：
输入：l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
输出：[8,9,9,9,0,0,0,1]
func addTwoNumbers(l1, l2 *ListNode) *ListNode {
    dummy := ListNode{} // 哨兵节点
    cur := &dummy
    carry := 0 // 进位
    for l1 != nil || l2 != nil || carry != 0 { // 有一个不是空节点，或者还有进位，就继续迭代
        if l1 != nil {
            carry += l1.Val // 节点值和进位加在一起
            l1 = l1.Next // 下一个节点
        }
        if l2 != nil {
            carry += l2.Val // 节点值和进位加在一起
            l2 = l2.Next // 下一个节点
        }
        cur.Next = &ListNode{Val: carry % 10} // 每个节点保存一个数位
        carry /= 10 // 新的进位
        cur = cur.Next // 下一个节点
    }
    return dummy.Next // 哨兵节点的下一个节点就是头节点
}
19. 删除链表的倒数第 N 个结点（中等）
给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。

示例 1：

输入：head = [1,2,3,4,5], n = 2
输出：[1,2,3,5]
示例 2：
输入：head = [1], n = 1
输出：[]
示例 3：
输入：head = [1,2], n = 1
输出：[1]
func removeNthFromEnd(head *ListNode, n int) *ListNode {
    // 由于可能会删除链表头部，用哨兵节点简化代码
    dummy := &ListNode{Next: head}
    left, right := dummy, dummy
    for ; n > 0; n-- {
        right = right.Next // 右指针先向右走 n 步
    }
    for right.Next != nil {
        left = left.Next
        right = right.Next // 左右指针一起走
    }
    left.Next = left.Next.Next // 左指针的下一个节点就是倒数第 n 个节点
    return dummy.Next
}
在头结点前加一个节点，”由于可能会删除链表头部，用哨兵节点简化代码“，删除节点时根据前一个节点删除
‘’遍历到最后一个节点，需要写 while node；如果要遍历到倒数第二个节点，需要写 while node.next‘’（right.Next不会移动到最后一个节点）
24. 两两交换链表中的节点
给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。
示例 1：

输入：head = [1,2,3,4]
输出：[2,1,4,3]
示例 2：
输入：head = []
输出：[]
示例 3：
输入：head = [1]
输出：[1]
func swapPairs(head *ListNode) *ListNode {
    dummy := &ListNode{Next: head} // 用哨兵节点简化代码逻辑
    node0 := dummy
    node1 := head
    for node1 != nil && node1.Next != nil { // 至少有两个节点
        node2 := node1.Next
        node3 := node2.Next

        node0.Next = node2 // 0 -> 2
        node2.Next = node1 // 2 -> 1
        node1.Next = node3 // 1 -> 3

        node0 = node1 // 下一轮交换，0 是 1
        node1 = node3 // 下一轮交换，1 是 3
    }
    return dummy.Next // 返回新链表的头节点
}

25. K 个一组翻转链表（困难）
给你链表的头节点 head ，每 k 个节点一组进行翻转，请你返回修改后的链表。
k 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。
你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。
示例 1：

输入：head = [1,2,3,4,5], k = 2
输出：[2,1,4,3,5]
示例 2：

输入：head = [1,2,3,4,5], k = 3
输出：[3,2,1,4,5]
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func reverseKGroup(head *ListNode, k int) *ListNode {
    dummy:=ListNode{Next:head}//哨兵节点
    p0:=&dummy//当前组的前驱节点，也就是当前组第一个节点的上一个节点，前一组的尾节点
    var pre,cur *ListNode=nil,head
    n:=0 //统计节点个数
    for cur:=head;cur!=nil;cur=cur.Next{
        n++
    }

    for ;k<=n;n-=k{
        //反转每一组节点
        for i:=0;i<k;i++{
            nxt:=cur.Next
            cur.Next=pre
            pre=cur
            cur=nxt
        }

        nxt:=p0.Next //临时保存当前组的最后一个节点，用来重置前驱节点
        nxt.Next=cur //将当前组最后一个节点与下一组第一个节点连接
        p0.Next=pre  //将当前组的一个节点与前驱节点连接，也就是上一组的最后一个节点
        p0=nxt       //重置前驱节点
    }
    return dummy.Next
}
不管翻转哪一组，当前组的第一个节点和最后一个节点都要与前驱节点和后继节点连接，pre是当前组的头节点，p0是前驱节点（前一组的尾节点），cur是后一组的第一个节点
138. 随机链表的复制（中等）
给你一个长度为 n 的链表，每个节点包含一个额外增加的随机指针 random ，该指针可以指向链表中的任何节点或空节点。
构造这个链表的 深拷贝。 深拷贝应该正好由 n 个 全新 节点组成，其中每个新节点的值都设为其对应的原节点的值。新节点的 next 指针和 random 指针也都应指向复制链表中的新节点，并使原链表和复制链表中的这些指针能够表示相同的链表状态。复制链表中的指针都不应指向原链表中的节点 。
例如，如果原链表中有 X 和 Y 两个节点，其中 X.random --> Y 。那么在复制链表中对应的两个节点 x 和 y ，同样有 x.random --> y 。
返回复制链表的头节点。
用一个由 n 个节点组成的链表来表示输入/输出中的链表。每个节点用一个 [val, random_index] 表示：
● val：一个表示 Node.val 的整数。
● random_index：随机指针指向的节点索引（范围从 0 到 n-1）；如果不指向任何节点，则为  null 。
你的代码 只 接受原链表的头节点 head 作为传入参数。
示例 1：

输入：head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
输出：[[7,null],[13,0],[11,4],[10,2],[1,0]]
示例 2：

输入：head = [[1,1],[2,1]]
输出：[[1,1],[2,1]]
示例 3：

输入：head = [[3,null],[3,0],[3,null]]
输出：[[3,null],[3,0],[3,null]]
func copyRandomList(head *Node) *Node {
    if head == nil {
        return nil
    }

    // 复制每个节点，把新节点直接插到原节点的后面
    for cur := head; cur != nil; cur = cur.Next.Next {
        cur.Next = &Node{Val: cur.Val, Next: cur.Next}
    }

    // 遍历交错链表中的原链表节点
    for cur := head; cur != nil; cur = cur.Next.Next {
        if cur.Random != nil {
            // 要复制的 random 是 cur.Random 的下一个节点
            cur.Next.Random = cur.Random.Next
        }
    }

    // 把交错链表分离成两个链表
    newHead := head.Next
    cur := head
    for ; cur.Next.Next != nil; cur = cur.Next {
        clone := cur.Next
        cur.Next = clone.Next        // 恢复原节点的 next
        clone.Next = clone.Next.Next // 设置新节点的 next
    }
    cur.Next = nil // 恢复原节点的 next
    return newHead
}

新节点的地址不一样
循环终止时，cur=3,但是原链表
还是1->2->3->3',如果不cur.Next = nil，会破坏原链表的结构
148. 排序链表（中等）
给你链表的头结点 head ，请将其按 升序 排列并返回 排序后的链表 。
示例 1：

输入：head = [4,2,1,3]
输出：[1,2,3,4]
示例 2：

输入：head = [-1,5,3,4,0]
输出：[-1,0,3,4,5]
示例 3：
输入：head = []
输出：[]
// 获取链表长度
func getListLength(head *ListNode) (length int) {
    for head != nil {
        length++
        head = head.Next
    }
    return
}

// 分割链表
// 如果链表长度 <= size，不做任何操作，返回空节点
// 如果链表长度 > size，把链表的前 size 个节点分割出来（断开连接），并返回剩余链表的头节点
func splitList(head *ListNode, size int) *ListNode {
    // 先找到 nextHead 的前一个节点
    cur := head
    for i := 0; i < size-1 && cur != nil; i++ {
        cur = cur.Next
    }

    // 如果链表长度 <= size
    if cur == nil || cur.Next == nil {
        return nil // 不做任何操作，返回空节点
    }

    nextHead := cur.Next
    cur.Next = nil // 断开 nextHead 的前一个节点和 nextHead 的连接
    return nextHead
}

// 21. 合并两个有序链表（双指针）
// 返回合并后的链表的头节点和尾节点
func mergeTwoLists(list1, list2 *ListNode) (head, tail *ListNode) {
    dummy := ListNode{} // 用哨兵节点简化代码逻辑
    cur := &dummy // cur 指向新链表的末尾
    for list1 != nil && list2 != nil {
        if list1.Val < list2.Val {
            cur.Next = list1 // 把 list1 加到新链表中
            list1 = list1.Next
        } else { // 注：相等的情况加哪个节点都是可以的
            cur.Next = list2 // 把 list2 加到新链表中
            list2 = list2.Next
        }
        cur = cur.Next
    }
    // 拼接剩余链表
    if list1 != nil {
        cur.Next = list1
    } else {
        cur.Next = list2
    }

    for cur.Next != nil {
        cur = cur.Next
    }
    // 循环结束后，cur 是合并后的链表的尾节点
    return dummy.Next, cur
}

func sortList(head *ListNode) *ListNode {
    length := getListLength(head) // 获取链表长度
    dummy := ListNode{Next: head} // 用哨兵节点简化代码逻辑
    // step 为步长，即参与合并的链表长度
    for step := 1; step < length; step *= 2 {
        newListTail := &dummy // 新链表的末尾
        cur := dummy.Next // 每轮循环的起始节点
        for cur != nil {
            // 从 cur 开始，分割出两段长为 step 的链表，头节点分别为 head1 和 head2
            head1 := cur
            head2 := splitList(head1, step)
            cur = splitList(head2, step) // 下一轮循环的起始节点
            // 合并两段长为 step 的链表
            head, tail := mergeTwoLists(head1, head2)
            // 合并后的头节点 head，插到 newListTail 的后面
            newListTail.Next = head
            newListTail = tail // tail 现在是新链表的末尾
        }
    }
    return dummy.Next
}
归并排序，按步长分割链表
空间复杂度优化成 O(1)。自底向上的意思是：
首先，归并长度为 1 的子链表。例如 [4,2,1,3]，把第一个节点和第二个节点归并，第三个节点和第四个节点归并，得到 [2,4,1,3]。
然后，归并长度为 2 的子链表。例如 [2,4,1,3]，把前两个节点和后两个节点归并，得到 [1,2,3,4]。
然后，归并长度为 4 的子链表。
依此类推，直到归并的长度大于等于链表长度为止，此时链表已经是有序的了。
具体算法：
遍历链表，获取链表长度 length。
初始化步长 step=1。
循环直到 step≥length。
每轮循环，从链表头节点开始。
分割出两段长为 step 的链表，合并，把合并后的链表插到新链表的末尾。重复该步骤，直到链表遍历完毕。
把 step 扩大一倍。回到第 4 步。
23. 合并 K 个升序链表（困难）
给你一个链表数组，每个链表都已经按升序排列。
请你将所有链表合并到一个升序链表中，返回合并后的链表。
示例 1：
输入：lists = [[1,4,5],[1,3,4],[2,6]]
输出：[1,1,2,3,4,4,5,6]
解释：链表数组如下：
[
  1->4->5,
  1->3->4,
  2->6
]
将它们合并到一个有序链表中得到。
1->1->2->3->4->4->5->6
示例 2：
输入：lists = []
输出：[]
示例 3：
输入：lists = [[]]
输出：[]
// 21. 合并两个有序链表
func mergeTwoLists(list1, list2 *ListNode) *ListNode {
    dummy := &ListNode{} // 用哨兵节点简化代码逻辑
    cur := dummy // cur 指向新链表的末尾
    for list1 != nil && list2 != nil {
        if list1.Val < list2.Val {
            cur.Next = list1 // 把 list1 加到新链表中
            list1 = list1.Next
        } else { // 注：相等的情况加哪个节点都是可以的
            cur.Next = list2 // 把 list2 加到新链表中
            list2 = list2.Next
        }
        cur = cur.Next
    }
    // 拼接剩余链表
    if list1 != nil {
        cur.Next = list1
    } else {
        cur.Next = list2
    }
    return dummy.Next
}

func mergeKLists(lists []*ListNode) *ListNode {
    m := len(lists)
    if m == 0 {
        return nil
    }
    for step := 1; step < m; step *= 2 {
        for i := 0; i < m-step; i += step * 2 {
            lists[i] = mergeTwoLists(lists[i], lists[i+step])
        }
    }
    return lists[0]
}
迭代
直接自底向上合并链表：
两两合并：把 lists[0] 和 lists[1] 合并，合并后的链表保存在 lists[0] 中；把 lists[2] 和 lists[3] 合并，合并后的链表保存在 lists[2] 中；依此类推。
四四合并：把 lists[0] 和 lists[2] 合并（相当于合并前四条链表），合并后的链表保存在 lists[0] 中；把 lists[4] 和 lists[6] 合并，合并后的链表保存在 lists[4] 中；依此类推。
八八合并：把 lists[0] 和 lists[4] 合并（相当于合并前八条链表），合并后的链表保存在 lists[0] 中；把 lists[8] 和 lists[12] 合并，合并后的链表保存在 lists[8] 中；依此类推。
依此类推，直到所有链表都合并到 lists[0] 中。最后返回 lists[0]。
i < m-step 是为了防止数组越界，确保安全访问 lists[i+step]
时间复杂度：O(Llogm)，其中 m 为 lists 的长度，L 为所有链表的长度之和。外层关于 step 的循环是 O(logm) 次，内层相当于把每个链表节点都遍历了一遍，是 O(L) 的，所以总的时间复杂度为 O(Llogm)。
空间复杂度：O(1)。
146. LRU 缓存（中等）
请你设计并实现一个满足  LRU (最近最少使用) 缓存 约束的数据结构。
实现 LRUCache 类：
● LRUCache(int capacity) 以 正整数 作为容量 capacity 初始化 LRU 缓存
● int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
● void put(int key, int value) 如果关键字 key 已经存在，则变更其数据值 value ；如果不存在，则向缓存中插入该组 key-value 。如果插入操作导致关键字数量超过 capacity ，则应该 逐出 最久未使用的关键字。
函数 get 和 put 必须以 O(1) 的平均时间复杂度运行。
示例：
输入
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
输出
[null, null, null, 1, null, -1, null, -1, 3, 4]

解释
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // 缓存是 {1=1}
lRUCache.put(2, 2); // 缓存是 {1=1, 2=2}
lRUCache.get(1);    // 返回 1

lRUCache.put(3, 3); // 该操作会使得关键字 2 作废，缓存是 {1=1, 3=3}
lRUCache.get(2);    // 返回 -1 (未找到)
lRUCache.put(4, 4); // 该操作会使得关键字 1 作废，缓存是 {4=4, 3=3}
lRUCache.get(1);    // 返回 -1 (未找到)
lRUCache.get(3);    // 返回 3
lRUCache.get(4);    // 返回 4
type Node struct {
    key, value int
    prev, next *Node
}

type LRUCache struct {
    capacity  int
    dummy     *Node // 哨兵节点
    keyToNode map[int]*Node
}

func Constructor(capacity int) LRUCache {
    dummy := &Node{}
    dummy.prev = dummy
    dummy.next = dummy
    return LRUCache{
        capacity:  capacity,
        dummy:     dummy,
        keyToNode: map[int]*Node{},
    }
}

// 删除一个节点（抽出一本书）
func (c *LRUCache) remove(x *Node) {
    x.prev.next = x.next
    x.next.prev = x.prev
}

// 在链表头添加一个节点（把一本书放到最上面）
func (c *LRUCache) pushFront(x *Node) {
    x.prev = c.dummy
    x.next = c.dummy.next
    x.prev.next = x
    x.next.prev = x
}

// 获取 key 对应的节点，同时把该节点移到链表头部
func (c *LRUCache) getNode(key int) *Node {
    node := c.keyToNode[key]
    if node == nil { // 没有这本书
        return nil
    }
    c.remove(node)    // 把这本书抽出来
    c.pushFront(node) // 放到最上面
    return node
}

func (c *LRUCache) Get(key int) int {
    node := c.getNode(key) // getNode 会把对应节点移到链表头部
    if node == nil {
        return -1
    }
    return node.value
}

func (c *LRUCache) Put(key, value int) {
    node := c.getNode(key) // getNode 会把对应节点移到链表头部
    if node != nil {       // 有这本书
        node.value = value // 更新 value
        return
    }
    node = &Node{key: key, value: value} // 新书
    c.keyToNode[key] = node
    c.pushFront(node) // 放到最上面
    if len(c.keyToNode) > c.capacity { // 书太多了
        backNode := c.dummy.prev
        delete(c.keyToNode, backNode.key)
        c.remove(backNode) // 去掉最后一本书
    }
}
🎯 核心思想：双向链表 + 哈希表
📚 图书馆比喻：
● 链表：代表书架上的书（最近使用的在最前面）
● 哈希表：快速找到某本书的位置
● 哨兵节点：简化边界处理（像书架的固定端点）
一开始哨兵节点 dummy 的 prev 和 next 都指向 dummy。随着节点的插入，dummy 的 next 指向链表的第一个节点（最上面的书），prev 指向链表的最后一个节点（最下面的书）。
复杂度分析
● 时间复杂度：所有操作均为 O(1)。
● 空间复杂度：O(min(p,capacity))，其中 p 为 put 的调用次数。
二叉树
94. 二叉树的中序遍历（简单）
示例 1：

输入：root = [1,null,2,3]
输出：[1,3,2]
示例 2：
输入：root = []
输出：[]
示例 3：
输入：root = [1]
输出：[1]
func inorderTraversal(root *TreeNode) (res []int) {
	for root != nil {
		if root.Left != nil {
			// predecessor 节点表示当前 root 节点向左走一步，然后一直向右走至无法走为止的节点
			predecessor := root.Left
			for predecessor.Right != nil && predecessor.Right != root {
				// 有右子树且没有设置过指向 root，则继续向右走
				predecessor = predecessor.Right
			}
			if predecessor.Right == nil {
				// 将 predecessor 的右指针指向 root，这样后面遍历完左子树 root.Left 后，就能通过这个指向回到 root
				predecessor.Right = root
				// 遍历左子树
				root = root.Left
			} else { // predecessor 的右指针已经指向了 root，则表示左子树 root.Left 已经访问完了
				res = append(res, root.Val)
				// 恢复原样
				predecessor.Right = nil
				// 遍历右子树
				root = root.Right
			}
		} else { // 没有左子树
			res = append(res, root.Val)
			// 若有右子树，则遍历右子树
			// 若没有右子树，则整颗左子树已遍历完，root 会通过之前设置的指向回到这颗子树的父节点
			root = root.Right
		}
	}
	return
}

复杂度分析
时间复杂度：O(n)，其中 n 为二叉树的节点个数。Morris 遍历中每个节点会被访问两次，因此总时间复杂度为 O(2n)=O(n)。
空间复杂度：O(1)。
104. 二叉树的最大深度（简单）
给定一个二叉树 root ，返回其最大深度。
二叉树的 最大深度 是指从根节点到最远叶子节点的最长路径上的节点数。
示例 1：


输入：root = [3,9,20,null,null,15,7]
输出：3
示例 2：
输入：root = [1,null,2]
输出：2
func maxDepth(root *TreeNode) int {
    if root == nil {
        return 0
    }
    lDepth := maxDepth(root.Left)
    rDepth := maxDepth(root.Right)
    return max(lDepth, rDepth) + 1
}
复杂度分析
● 时间复杂度：O(n)，其中 n 为二叉树的节点个数。
● 空间复杂度：O(n)。最坏情况下，二叉树退化成一条链，递归需要 O(n) 的栈空间。
226. 翻转二叉树
给你一棵二叉树的根节点 root ，翻转这棵二叉树，并返回其根节点。
示例 1：

输入：root = [4,2,7,1,3,6,9]
输出：[4,7,2,9,6,3,1]
示例 2：

输入：root = [2,1,3]
输出：[2,3,1]
示例 3：
输入：root = []
输出：[]
func invertTree(root *TreeNode) *TreeNode {
    if root == nil {
        return nil
    }
    left := invertTree(root.Left) // 翻转左子树
    right := invertTree(root.Right) // 翻转右子树
    root.Left = right // 交换左右儿子
    root.Right = left
    return root
}

101. 对称二叉树（简单）
给你一个二叉树的根节点 root ， 检查它是否轴对称。
示例 1：

输入：root = [1,2,2,3,4,4,3]
输出：true
示例 2：

输入：root = [1,2,2,null,3,null,3]
输出：false
// 在【100. 相同的树】的基础上稍加改动
func isSameTree(p, q *TreeNode) bool {
    if p == nil || q == nil {
        return p == q
    }
    return p.Val == q.Val && isSameTree(p.Left, q.Right) && isSameTree(p.Right, q.Left)
}

func isSymmetric(root *TreeNode) bool {
    return isSameTree(root.Left, root.Right)
}
对于对称的两个节点：
● 左子树的左节点 vs 右子树的右节点（外侧）
● 左子树的右节点 vs 右子树的左节点（内侧）
复杂度分析
● 时间复杂度：O(n)，其中 n 为二叉树的节点个数。
● 空间复杂度：O(n)。最坏情况下，二叉树退化成一条链，递归需要 O(n) 的栈空间。
543. 二叉树的直径（简单）
给你一棵二叉树的根节点，返回该树的 直径 。
二叉树的 直径 是指树中任意两个节点之间最长路径的 长度 。这条路径可能经过也可能不经过根节点 root 。
两节点之间路径的 长度 由它们之间边数表示。

示例 1：

输入：root = [1,2,3,4,5]
输出：3
解释：3 ，取路径 [4,2,1,3] 或 [5,2,1,3] 的长度。
示例 2：
输入：root = [1,2]
输出：1
\\空节点深度=-1 ⇒ 叶子深度=0：因为计算的是边数量
func diameterOfBinaryTree(root *TreeNode) (ans int) {
    var dfs func(*TreeNode) int
    dfs = func(node *TreeNode) int {
        if node == nil {
            return -1 // 对于叶子来说，链长就是 -1+1=0
        }
        lLen := dfs(node.Left) + 1  // 左子树最大链长+1
        rLen := dfs(node.Right) + 1 // 右子树最大链长+1
        ans = max(ans, lLen+rLen)   // 两条链拼成路径
        return max(lLen, rLen)      // 当前子树最大链长
    }
    dfs(root)
    return
}

\\空节点深度=0 ⇒ 叶子深度=1：因为计算的是节点数量
func diameterOfBinaryTree(root *TreeNode) int {
    ans:=0
    var dfs func (*TreeNode) int
    dfs=func (node *TreeNode) int{
        if node==nil{
            return 0
        }
        LLen:=dfs(node.Left)
        Rlen:=dfs(node.Right)
        ans=max(ans,LLen+Rlen)
        return max(LLen,Rlen)+1
    }
    dfs(root)
    return ans
}
102. 二叉树的层序遍历(中等)
给你二叉树的根节点 root ，返回其节点值的 层序遍历 。 （即逐层地，从左到右访问所有节点）。

示例 1：

输入：root = [3,9,20,null,null,15,7]
输出：[[3],[9,20],[15,7]]
示例 2：
输入：root = [1]
输出：[[1]]
示例 3：
输入：root = []
输出：[]
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if root is None:
            return []
        ans = []
        q = deque([root])
        while q:
            vals = []
            for _ in range(len(q)):
                node = q.popleft()
                vals.append(node.val)
                if node.left:  q.append(node.left)
                if node.right: q.append(node.right)
            ans.append(vals)
        return ans
复杂度分析
时间复杂度：O(n)，其中 n 为二叉树的节点个数。
空间复杂度：O(n)。满二叉树（每一层都填满）最后一层有大约 n/2 个节点，因此队列中最多有 O(n) 个元素，所以空间复杂度是 O(n) 的。
108. 将有序数组转换为二叉搜索树（简单）
给你一个整数数组 nums ，其中元素已经按 升序 排列，请你将其转换为一棵 平衡 二叉搜索树。

示例 1：

输入：nums = [-10,-3,0,5,9]
输出：[0,-3,9,-10,null,5]
解释：[0,-10,5,null,-3,null,9] 也将被视为正确答案：
示例 2：

输入：nums = [1,3]
输出：[3,1]
解释：[1,null,3] 和 [3,1] 都是高度平衡二叉搜索树。
func sortedArrayToBST(nums []int) *TreeNode {
    if len(nums) == 0 {
        return nil
    }
    m := len(nums) / 2
    return &TreeNode{
        Val:   nums[m],
        Left:  sortedArrayToBST(nums[:m]),
        Right: sortedArrayToBST(nums[m+1:]),
    }
}
复杂度分析
时间复杂度：O(n)，其中 n 是 nums 的长度。每次递归要么返回空节点，要么把 nums 的一个数转成一个节点，所以递归次数是 O(n) 的，所以时间复杂度是 O(n)。需要注意，Python 的第一种写法有切片的复制开销，二叉树的每一层都需要花费 O(n) 的时间，一共有 O(logn) 层，所以时间复杂度是 O(nlogn)；第二种写法避免了切片的复制开销，时间复杂度是 O(n)。
空间复杂度：O(n)。如果不计入返回值和切片的空间，那么空间复杂度为 O(logn)，即递归栈的开销。
98. 验证二叉搜索树（中等）
给你一个二叉树的根节点 root ，判断其是否是一个有效的二叉搜索树。
有效 二叉搜索树定义如下：
● 节点的左子树只包含 严格小于 当前节点的数。
● 节点的右子树只包含 严格大于 当前节点的数。
● 所有左子树和右子树自身必须也是二叉搜索树。

示例 1：

输入：root = [2,1,3]
输出：true
示例 2：

输入：root = [5,1,4,null,null,3,6]
输出：false
解释：根节点的值是 5 ，但是右子节点的值是 4 。
func dfs(node *TreeNode, left, right int) bool {
    if node == nil {
        return true
    }
    x := node.Val
    return left < x && x < right &&
        dfs(node.Left, left, x) &&
        dfs(node.Right, x, right)
}

func isValidBST(root *TreeNode) bool {
    return dfs(root, math.MinInt, math.MaxInt)
}
复杂度分析
时间复杂度：O(n)，其中 n 为二叉搜索树的节点个数。
空间复杂度：O(n)。最坏情况下，二叉搜索树退化成一条链（注意题目没有保证它是平衡树），因此递归需要 O(n) 的栈空间。
if node == nil { return true } 是因为：
1. 空树定义：空树被认为是有效的BST
2. 递归基础：为递归提供终止条件
3. 逻辑正确：确保叶子节点能通过验证
4. 数学一致性：符合BST的数学定义
230. 二叉搜索树中第 K 小的元素（中等）
给定一个二叉搜索树的根节点 root ，和一个整数 k ，请你设计一个算法查找其中第 k 小的元素（从 1 开始计数）。

示例 1：

输入：root = [3,1,4,null,2], k = 1
输出：1
示例 2：

输入：root = [5,3,6,2,4,null,null,1], k = 3
输出：3
func kthSmallest(root *TreeNode, k int) (ans int) {
    var dfs func(*TreeNode)
    dfs = func(node *TreeNode) {
        if node == nil || k == 0 {
            return
        }
        dfs(node.Left) // 左
        k--
        if k == 0 {
            ans = node.Val // 根
        }
        dfs(node.Right) // 右
    }
    dfs(root)
    return
}
由于中序遍历就是在从小到大遍历节点值，所以遍历到的第 k 个节点值就是答案。
在中序遍历，即「左-根-右」的过程中，每次递归完左子树，就把 k 减少 1，表示我们按照中序遍历访问到了一个节点。如果减一后 k 变成 0，那么答案就是当前节点的值，用一个外部变量 ans 记录。
复杂度分析
时间复杂度：O(n)，其中 n 是二叉树的节点个数。
空间复杂度：O(h)，其中 h 是二叉树的高度。递归需要 O(h) 的栈空间。最坏情况下，二叉树退化成一条链，递归需要 O(n) 的栈空间。
199. 二叉树的右视图（中等）
给定一个二叉树的 根节点 root，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。
示例 1：
输入：root = [1,2,3,null,5,null,4]
输出：[1,3,4]
解释：

示例 2：
输入：root = [1,2,3,4,null,null,null,5]
输出：[1,3,4,5]
解释：

示例 3：
输入：root = [1,null,3]
输出：[1,3]
示例 4：
输入：root = []
输出：[]
func rightSideView(root *TreeNode) (ans []int) {
    var dfs func(*TreeNode, int)
    dfs = func(node *TreeNode, depth int) {
        if node == nil {
            return
        }
        if depth == len(ans) { // 这个深度首次遇到
            ans = append(ans, node.Val)
        }
        dfs(node.Right, depth+1) // 先递归右子树，保证首次遇到的一定是最右边的节点
        dfs(node.Left, depth+1)
    }
    dfs(root, 0)
    return
}
思路：先递归右子树，再递归左子树，当某个深度首次到达时，对应的节点就在右视图中。
复杂度分析
时间复杂度：O(n)，其中 n 是二叉树的节点个数。
空间复杂度：O(h)，其中 h 是二叉树的高度。递归需要 O(h) 的栈空间。最坏情况下，二叉树退化成一条链，递归需要 O(n) 的栈空间。
114. 二叉树展开为链表（中等）
给你二叉树的根结点 root ，请你将它展开为一个单链表：
● 展开后的单链表应该同样使用 TreeNode ，其中 right 子指针指向链表中下一个结点，而左子指针始终为 null 。
● 展开后的单链表应该与二叉树 先序遍历 顺序相同。

示例 1：

输入：root = [1,2,5,3,4,null,6]
输出：[1,null,2,null,3,null,4,null,5,null,6]
示例 2：
输入：root = []
输出：[]
示例 3：
输入：root = [0]
输出：[0]
func flatten(root *TreeNode) {
    var head *TreeNode
    var dfs func(*TreeNode)
    dfs = func(node *TreeNode) {
        if node == nil {
            return
        }
        dfs(node.Right)
        dfs(node.Left)
        node.Left = nil
        node.Right = head // 头插法，相当于链表的 node.Next = head
        head = node       // 现在链表头节点是 node
    }
    dfs(root)
}
头插法

采用头插法构建链表，也就是从节点 6 开始，在 6 的前面插入 5，在 5 的前面插入 4，依此类推。
为此，要按照 6→5→4→3→2→1 的顺序访问节点。如何遍历这棵树，才能实现这个顺序？
按照右子树 - 左子树 - 根的顺序 DFS 这棵树。
DFS 的同时，记录当前链表的头节点为 head。一开始 head 是空节点。
具体来说：
如果当前节点为空，返回。
递归右子树。
递归左子树。
把 root.left 置为空。
头插法，把 root 插在 head 的前面，也就是 root.right=head。
现在 root 是链表的头节点，把 head 更新为 root。
复杂度分析
● 时间复杂度：O(n)，其中 n 是二叉树的节点个数。
● 空间复杂度：O(n)。递归需要 O(n) 的栈空间。
105. 从前序与中序遍历序列构造二叉树（中等）
给定两个整数数组 preorder 和 inorder ，其中 preorder 是二叉树的先序遍历， inorder 是同一棵树的中序遍历，请构造二叉树并返回其根节点。

示例 1:

输入: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
输出: [3,9,20,null,null,15,7]
示例 2:
输入: preorder = [-1], inorder = [-1]
输出: [-1]
func buildTree(preorder, inorder []int) *TreeNode {
    n := len(preorder)
    index := make(map[int]int, n)
    for i, x := range inorder {
        index[x] = i
    }

    var dfs func(int, int, int, int) *TreeNode
    dfs = func(preL, preR, inL, inR int) *TreeNode {
        if preL == preR { // 空节点
            return nil
        }
        leftSize := index[preorder[preL]] - inL // 左子树的大小
        left := dfs(preL+1, preL+1+leftSize, inL, inL+leftSize)
        right := dfs(preL+1+leftSize, preR, inL+1+leftSize, inR)
        return &TreeNode{preorder[preL], left, right}
    }
    return dfs(0, n, 0, n) // 左闭右开区间
}
left和right分别表示preorder和inorder的左子树和右子树，inL+1+leftSize跳过inorder中的根节点
复杂度分析
● 时间复杂度：O(n)，其中 n 为 preorder 的长度。递归 O(n) 次，每次只需要 O(1) 的时间。
● 空间复杂度：O(n)。
437. 路径总和 III（中等）
给定一个二叉树的根节点 root ，和一个整数 targetSum ，求该二叉树里节点值之和等于 targetSum 的 路径 的数目。
路径 不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。

示例 1：

输入：root = [10,5,-3,3,2,null,11,3,-2,null,1], targetSum = 8
输出：3
解释：和等于 8 的路径有 3 条，如图所示。
示例 2：
输入：root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
输出：3
func pathSum(root *TreeNode, targetSum int) (ans int) {
    cnt := map[int]int{0: 1}
    var dfs func(*TreeNode, int)
    dfs = func(node *TreeNode, s int) {
        if node == nil {
            return
        }

        s += node.Val
        // 把 node 当作路径的终点，统计有多少个起点
        ans += cnt[s-targetSum]

        cnt[s]++
        dfs(node.Left, s)
        dfs(node.Right, s)
        cnt[s]-- // 恢复现场
    }
    dfs(root, 0)
    return ans
}
和 560 题一样的套路：一边遍历二叉树，一边用哈希表 cnt 统计前缀和（从根节点开始的路径和）的出现次数。设从根到终点 node 的路径和为 s，那么起点的个数就是 cnt[s−targetSum]，加入答案。对比 560 题，我们在枚举子数组的右端点（终点），统计有多少个左端点（起点），做法完全一致
问：代码中的「恢复现场」用意何在？
答：如果不恢复现场，当我们递归完左子树，要递归右子树时，cnt 中还保存着左子树的数据。但递归到右子树，要计算的路径并不涉及到左子树的任何节点，如果不恢复现场，cnt 中统计的前缀和个数会更多，我们算出来的答案可能比正确答案更大。

问：为什么要在递归完右子树后才能恢复现场？能否在递归完左子树后就恢复现场呢？换句话说，代码中的恢复现场能否写在 dfs(node.left, s) 和 dfs(node.right, s) 之间？
答：看代码，恢复现场恢复的是什么？是去掉当前节点 node 的信息。递归 node 的右子树时，需不需要 node 的信息？需要，因为 node 在路径中。所以要在递归完右子树后，才能恢复现场。恢复现场的代码不能写在 dfs(node.left, s) 和 dfs(node.right, s) 之间。

问：为什么递归参数 s 不需要恢复现场？
答：s 是基本类型，在函数调用的时候会复制一份往下传递，s += node.val 修改的仅仅是当前递归函数中的 s 参数，并不会影响到其他递归函数中的 s。注：如果把 s 放在递归函数外，此时只有一个 s，执行 s += node.val 就会影响全局了。
复杂度分析
● 时间复杂度：O(n)，其中 n 是二叉树的节点个数。
● 空间复杂度：O(n)。
236. 二叉树的最近公共祖先（中等）
给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
百度百科中最近公共祖先的定义为：“对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

示例 1：

输入：root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
输出：3
解释：节点 5 和节点 1 的最近公共祖先是节点 3 。
示例 2：

输入：root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
输出：5
解释：节点 5 和节点 4 的最近公共祖先是节点 5 。因为根据定义最近公共祖先节点可以为节点本身。
示例 3：
输入：root = [1,2], p = 1, q = 2
输出：1
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
    if root == nil || root == p || root == q {
        return root // 找到 p 或 q 就不往下递归了，原因见上面答疑
    }
    left := lowestCommonAncestor(root.Left, p, q)
    right := lowestCommonAncestor(root.Right, p, q)
    if left != nil && right != nil { // 左右都找到
        return root // 当前节点是最近公共祖先
    }
    // 如果只有左子树找到，就返回左子树的返回值
    // 如果只有右子树找到，就返回右子树的返回值
    // 如果左右子树都没有找到，就返回 nil（注意此时 right = nil）
    if left != nil {
        return left
    }
    return right
}
问：lowestCommonAncestor 函数的返回值是什么意思？
答：返回值的准确含义是「最近公共祖先的候选项」。对于最外层的递归调用者来说，返回值是最近公共祖先的意思。但是，在递归过程中，返回值可能是最近公共祖先，也可能是空节点（表示子树内没找到任何有用信息）、节点 p 或者节点 q（可能成为最近公共祖先，或者用来辅助判断上面的某个节点是否为最近公共祖先）。

问：为什么发现当前节点是 p 或者 q 就不再往下递归了？万一下面有 q 或者 p 呢？
答：如果下面有 q 或者 p，那么当前节点就是最近公共祖先，直接返回当前节点。如果下面没有 q 和 p，那既然都没有要找的节点了，也不需要递归，直接返回当前节点。
复杂度分析
● 时间复杂度：O(n)，其中 n 为二叉树的节点个数。
● 空间复杂度：O(n)。最坏情况下，二叉树是一条链，因此递归需要 O(n) 的栈空间。
124. 二叉树中的最大路径和（困难）
二叉树中的 路径 被定义为一条节点序列，序列中每对相邻节点之间都存在一条边。同一个节点在一条路径序列中 至多出现一次 。该路径 至少包含一个 节点，且不一定经过根节点。
路径和 是路径中各节点值的总和。
给你一个二叉树的根节点 root ，返回其 最大路径和 。

示例 1：

输入：root = [1,2,3]
输出：6
解释：最优路径是 2 -> 1 -> 3 ，路径和为 2 + 1 + 3 = 6
示例 2：

输入：root = [-10,9,20,null,null,15,7]
输出：42
解释：最优路径是 15 -> 20 -> 7 ，路径和为 15 + 20 + 7 = 42
func maxPathSum(root *TreeNode) int {
    ans := math.MinInt
    var dfs func(*TreeNode) int
    dfs = func(node *TreeNode) int {
        if node == nil {
            return 0 // 没有节点，和为 0
        }
        lVal := dfs(node.Left)  // 左子树最大链和
        rVal := dfs(node.Right) // 右子树最大链和
        ans = max(ans, lVal+rVal+node.Val) // 两条链拼成路径
        return max(max(lVal, rVal)+node.Val, 0) // 当前子树最大链和（注意这里和 0 取最大值了）
    }
    dfs(root)
    return ans
}
本题有两个关键概念：
链：从下面的某个节点（不一定是叶子）到当前节点的路径。把这条链的节点值之和，作为 dfs 的返回值。如果节点值之和是负数，则返回 0（和 0 取最大值）。这个思想和 53. 最大子数组和 是一样的，如果左侧子数组的元素和是负数，就不和当前元素拼起来。
直径：等价于由两条（或者一条）链拼成的路径。我们枚举每个 node，假设直径在这里「拐弯」，也就是计算由左右两条从下面的某个节点（不一定是叶子）到 node 的链的节点值之和，去更新答案的最大值。
⚠注意：dfs 返回的是链的节点值之和，不是直径的节点值之和。
复杂度分析
● 时间复杂度：O(n)，其中 n 为二叉树的节点个数。
● 空间复杂度：O(n)。最坏情况下，二叉树退化成一条链，递归需要 O(n) 的栈空间。
图论
200. 岛屿数量（中等）
给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。
此外，你可以假设该网格的四条边均被水包围。
示例 1：
输入：grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
输出：1
示例 2：
输入：grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
输出：3
func numIslands(grid [][]byte) (ans int) {
    m, n := len(grid), len(grid[0])
    var dfs func(int, int)
    dfs = func(i, j int) {
        // 出界，或者不是 '1'，就不再往下递归
        if i < 0 || i >= m || j < 0 || j >= n || grid[i][j] != '1' {
            return
        }
        grid[i][j] = '2' // 插旗！避免来回横跳无限递归
        dfs(i, j-1)      // 往左走
        dfs(i, j+1)      // 往右走
        dfs(i-1, j)      // 往上走
        dfs(i+1, j)      // 往下走
    }

    for i, row := range grid {
        for j, c := range row {
            if c == '1' { // 找到了一个新的岛
                dfs(i, j) // 把这个岛插满旗子，这样后面遍历到的 '1' 一定是新的岛
                ans++
            }
        }
    }
    return
}
遇到岛屿插旗，遍历
复杂度分析
● 时间复杂度：O(mn)，其中 m 和 n 分别为 grid 的行数和列数。
● 空间复杂度：O(mn)。最坏情况下，递归需要 O(mn) 的栈空间。
