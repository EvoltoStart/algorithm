package main

import "math"

// 152. 乘积最大子数组（中等）
func maxProduct(nums []int) int {
	ans := math.MinInt // 注意答案可能是负数
	fMax, fMin := 1, 1
	for _, x := range nums {
		fMax, fMin = max(fMax*x, fMin*x, x),
			min(fMax*x, fMin*x, x)
		ans = max(ans, fMax)
	}
	return ans
}

// 416. 分割等和子集（中等）
func canPartition(nums []int) bool {
	s := 0
	for _, x := range nums {
		s += x
	}
	if s%2 != 0 {
		return false
	}
	s /= 2 // 注意这里把 s 减半了
	f := make([]bool, s+1)
	f[0] = true
	s2 := 0
	for _, x := range nums {
		s2 = min(s2+x, s)          //动态缩小内层循环的遍历范围，避免在前期做大量无用功。
		for j := s2; j >= x; j-- { //想要凑出的目标和j比x小，就不可能去选
			f[j] = f[j] || f[j-x]
		}
		if f[s] {
			return true
		}
	}
	return false
}

// 62. 不同路径（中等）
func uniquePaths(m, n int) int {
	f := make([]int, n+1)
	f[1] = 1      //第一列
	for range m { //循环的每行
		for j := range n { //循环的每列                           f[j+1]代表当前行当前列
			f[j+1] += f[j] //f[j+1]没有更新前存放的是上一行的数据，f[j]代表前一列的数据，j从0开始循环，这里我们用f[1]为第一列，所以偏移一位
		}
	}
	return f[n]
}

// 64. 最小路径和(中等)
func minPathSum(grid [][]int) int {
	n := len(grid[0])
	f := make([]int, n+1)
	for j := range f {
		f[j] = math.MaxInt
	}
	f[1] = 0                   //左上角，边界
	for _, row := range grid { //循环中的每一行
		for j, x := range row { //循环中的每一列
			f[j+1] = min(f[j], f[j+1]) + x //f[j+1]当前行的当前列，右边的是上一行的当前列，f[j]当前列的左边一列
		}
	}
	return f[n]
}

// 5. 最长回文子串(中等)
func longestPalindrome(s string) string {
	n := len(s)
	ansLeft, ansRight := 0, 0

	// 奇回文串
	for i := range n {
		l, r := i, i
		for l >= 0 && r < n && s[l] == s[r] {
			l--
			r++
		}
		if r-l-1 > ansRight-ansLeft { //r-l-1==r-1-l+1+1,l 和 r 所指的字符，
			//已经是不符合回文要求的了。真正的回文串边界，是 l 的右边一位（l+1）到 r 的左边一位（r-1）。
			ansLeft = l + 1
			ansRight = r // 左闭右开区间
		}
	}

	// 偶回文串
	for i := range n - 1 {
		l, r := i, i+1
		for l >= 0 && r < n && s[l] == s[r] {
			l--
			r++
		}
		if r-l-1 > ansRight-ansLeft {
			ansLeft = l + 1
			ansRight = r // 左闭右开区间
		}
	}

	return s[ansLeft:ansRight]
}

// 1143. 最长公共子序列
func longestCommonSubsequence(text1 string, text2 string) int {
	n := len(text2)
	f := make([]int, n+1)
	for _, x := range text1 {
		pre := 0
		for j, y := range text2 {
			if x == y {
				f[j+1], pre = pre+1, f[j+1]
			} else {
				f[j+1], pre = max(f[j+1], f[j]), f[j+1]
			}
		}
	}
	return f[n]
}

// 72. 编辑距离
func minDistance(word1 string, word2 string) int {
	n := len(word2)
	f := make([]int, n+1)
	for j := range n {
		f[j+1] = j + 1
	}
	for _, x := range word1 {
		pre := f[0]
		f[0]++
		for j, y := range word2 {
			if x == y {
				f[j+1], pre = pre, f[j+1]
			} else {
				f[j+1], pre = min(f[j+1], f[j], pre)+1, f[j+1]
			}
		}
	}
	return f[n]
}
