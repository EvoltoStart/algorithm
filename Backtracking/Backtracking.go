package main

import (
	"fmt"
	"slices"
	"strings"
)

func main() {
	fmt.Println("xxx")
	fmt.Println("<>")

	fmt.Println("<>")

}

func subsets(nums []int) [][]int {
	ans := make([][]int, 1<<len(nums))
	for i := range ans { // 枚举全集 U 的所有子集 i，i就是一个子集
		for j, x := range nums {
			if i>>j&1 == 1 { // j 在集合 i 中，j是一个元素
				ans[i] = append(ans[i], x)
			}
		}
	}
	return ans
}

func permute(nums []int) (ans [][]int) {
	n := len(nums)
	path := make([]int, n)
	onPath := make([]bool, n)
	var dfs func(int)
	dfs = func(i int) {
		if i == n {
			ans = append(ans, append([]int(nil), path...))
			return
		}
		for j, on := range onPath {
			if !on {
				path[i] = nums[j]
				onPath[j] = true //访问后
				dfs(i + 1)
				onPath[j] = false //恢复未访问的状态
			}
		}
	}
	dfs(0)
	return
}

var mapping = [...]string{"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"}

func letterCombinations(digits string) (ans []string) {
	n := len(digits)
	if n == 0 {
		return
	}

	path := make([]byte, n) // 注意 path 长度一开始就是 n，不是空列表

	var dfs func(int)
	dfs = func(i int) {
		if i == n {
			ans = append(ans, string(path))
			return
		}
		for _, c := range mapping[digits[i]-'0'] {
			path[i] = byte(c) // 直接覆盖
			dfs(i + 1)
		}
	}

	dfs(0)
	return
}

func combinationSum(candidates []int, target int) (ans [][]int) {
	n := len(candidates)
	// 完全背包
	f := make([][]bool, n+1) //n+1 target+1 考虑边界情况
	f[0] = make([]bool, target+1)
	f[0][0] = true
	for i, x := range candidates {
		f[i+1] = make([]bool, target+1)
		for j, b := range f[i] {
			f[i+1][j] = b || j >= x && f[i+1][j-x] //(b || j >= x && f[i+1][j-x]),‘=’优先级最低
		}
	}

	path := []int{}
	var dfs func(int, int)
	dfs = func(i, left int) {
		if left == 0 {
			// 找到一个合法组合
			ans = append(ans, slices.Clone(path))
			return
		}

		// 无法用下标在 [0, i] 中的数字组合出 left
		if left < 0 || !f[i+1][left] {
			return
		}

		// 不选
		dfs(i-1, left)

		// 选
		path = append(path, candidates[i]) // 把当前数加入路径
		dfs(i, left-candidates[i])         // 继续用 i（因为可以重复选）
		path = path[:len(path)-1]          // 回溯，撤销选择

	}

	// 倒着递归，这样参数符合 f 数组的定义
	dfs(n-1, target)
	return ans
}

func generateParenthesis(n int) (ans []string) {
	path := make([]byte, n*2) // 所有括号长度都是一样的 2n

	// 目前填了 left 个左括号，right 个右括号
	var dfs func(int, int)
	dfs = func(left, right int) {
		if right == n { // 填完 2n 个括号
			ans = append(ans, string(path)) // 加入答案
			return
		}
		if left < n { // 可以填左括号
			path[left+right] = '(' // 直接覆盖
			dfs(left+1, right)
		}
		if right < left { // 可以填右括号
			path[left+right] = ')' // 直接覆盖
			dfs(left, right+1)
		}
	}

	dfs(0, 0) // 一开始没有填括号
	return
}

var dirs = []struct{ x, y int }{{0, -1}, {0, 1}, {-1, 0}, {1, 0}}

func exist(board [][]byte, word string) bool {
	cnt := map[byte]int{}
	for _, row := range board {
		for _, c := range row {
			cnt[c]++
		}
	}

	// 优化一
	w := []byte(word)
	wordCnt := map[byte]int{}
	for _, c := range w {
		wordCnt[c]++
		if wordCnt[c] > cnt[c] {
			return false
		}
	}

	// 优化二
	if cnt[w[len(w)-1]] < cnt[w[0]] {
		slices.Reverse(w)
	}

	m, n := len(board), len(board[0])
	var dfs func(int, int, int) bool
	dfs = func(i, j, k int) bool {
		if board[i][j] != w[k] { // 匹配失败
			return false
		}
		if k == len(w)-1 { // 匹配成功，单词没有查到最后一个继续查找
			return true
		}
		board[i][j] = 0 // 标记访问过 同一个单元格内的字母不允许被重复使用
		for _, d := range dirs {
			x, y := i+d.x, j+d.y // 相邻格子
			if 0 <= x && x < m && 0 <= y && y < n && dfs(x, y, k+1) {
				return true // 搜到了！
			}
		}
		board[i][j] = w[k] // 恢复现场
		return false       // 没搜到
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if dfs(i, j, 0) {
				return true // 搜到了！
			}
		}
	}
	return false // 没搜到
}

func isPalindrome(s string) bool {
	n := len(s)
	for i := range n / 2 {
		if s[i] != s[n-1-i] {
			return false
		}
	}
	return true
}

func partition(s string) (ans [][]string) {
	path := []string{}

	// 考虑 s 怎么分割
	var dfs func(string)
	dfs = func(s string) {
		n := len(s)
		if n == 0 { // s 分割完毕
			ans = append(ans, slices.Clone(path))
			return
		}
		for i := 1; i <= n; i++ { // 枚举子串长度
			substr := s[:i]
			if isPalindrome(substr) {
				path = append(path, substr) // 分割！
				// 考虑剩余的 s[i:] 怎么分割
				dfs(s[i:])
				path = path[:len(path)-1] // 恢复现场
			}
		}
	}

	dfs(s)
	return
}

func solveNQueens(n int) (ans [][]string) {
	queens := make([]int, n) // 皇后放在 (r,queens[r])
	col := make([]bool, n)
	diag1 := make([]bool, n*2-1)
	diag2 := make([]bool, n*2-1)
	var dfs func(int)
	dfs = func(r int) {
		if r == n {
			board := make([]string, n)
			for i, c := range queens {
				board[i] = strings.Repeat(".", c) + "Q" + strings.Repeat(".", n-1-c)
			}
			ans = append(ans, board)
			return
		}
		// 在 (r,c) 放皇后
		for c, ok := range col {
			rc := r - c + n - 1
			if !ok && !diag1[r+c] && !diag2[rc] { // 判断能否放皇后
				queens[r] = c                                    // 直接覆盖，无需恢复现场
				col[c], diag1[r+c], diag2[rc] = true, true, true // 皇后占用了 c 列和两条斜线
				dfs(r + 1)
				col[c], diag1[r+c], diag2[rc] = false, false, false // 恢复现场
			}
		}
	}
	dfs(0)
	return
}
