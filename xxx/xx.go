package xxx

import (
	"fmt"
	"slices"
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

