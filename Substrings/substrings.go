package main

import "fmt"

// 和为 K 的子数组
func subarraySum(nums []int, k int) int {
	maps := make(map[int]int)
	pre := 0
	count := 0
	maps[0] = 1
	for _, v := range nums {
		pre += v
		count = count + maps[pre-k]
		maps[pre]++
	}
	return count
}

// 滑动窗口最大值
func maxSlidingWindow(nums []int, k int) []int {
	n := len(nums)
	prefixMax := make([]int, n)
	suffixMax := make([]int, n)
	for i, j := 0, n-1; i < n && j >= 0; i, j = i+1, j-1 {
		if i%k == 0 {
			prefixMax[i] = nums[i]
		} else {
			prefixMax[i] = max(nums[i], prefixMax[i-1])
		}
		if (j+1)%k == 0 || j == n-1 {
			suffixMax[j] = nums[j]
		} else {
			suffixMax[j] = max(nums[j], suffixMax[j+1])
		}
	}

	result := make([]int, n-k+1)
	for i := range result {
		result[i] = max(suffixMax[i], prefixMax[i+k-1])
	}
	return result
}

// 76. 最小覆盖子串
func minWindow(s, t string) string {
	less := 0
	cnt := [128]int{}
	ansLeft, ansRight := -1, len(s)
	for _, ch := range t {
		if cnt[ch] == 0 {
			less++
		}
		cnt[ch]++
	}
	left := 0
	for right, ch := range s {
		cnt[ch]--
		if cnt[ch] == 0 {
			less--
		}
		for less == 0 {
			if right-left < ansRight-ansLeft {
				ansLeft, ansRight = left, right
			}
			x := s[left]
			if cnt[x] == 0 {
				less++
			}
			cnt[x]++
			left++
		}
	}
	if ansLeft < 0 {
		return ""
	}
	return s[ansLeft : ansRight+1]
}
func main() {

	fmt.Println(subarraySum([]int{1, -1, 0}, 0))
	fmt.Println(maxSlidingWindow([]int{1, 3, -1, -3, 5, 3, 6, 7}, 3))

}
