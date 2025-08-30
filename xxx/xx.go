package xxx

import "fmt"

func main() {
	fmt.Println("xxx")
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
