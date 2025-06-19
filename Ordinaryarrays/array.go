package main

import (
	"fmt"
	"sort"
)

// 最大子序和（中等）
func maxSubArray(nums []int) int {
	pre, maxAns := 0, nums[0]
	for i := 0; i < len(nums); i++ {
		pre = max(pre+nums[i], nums[i])
		maxAns = max(maxAns, pre)
	}
	return maxAns
}

// 合并区间(中等)
func merge(intervals [][]int) [][]int {
	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i][0] < intervals[j][0]
	})
	merged := [][]int{}
	for _, interval := range intervals {
		n := len(merged)
		if n == 0 || interval[0] > merged[n-1][1] {
			merged = append(merged, interval)
		} else {
			merged[n-1][1] = max(merged[n-1][1], interval[1])
		}
	}
	return merged
}

// 轮转数组  给定一个整数数组 nums，将数组中的元素向右轮转 k 个位置，其中 k 是非负数。
func reverse(nums []int) {
	for i, j := 0, len(nums)-1; i < j; i, j = i+1, j-1 {
		nums[i], nums[j] = nums[j], nums[i]
	}
}
func rotate(nums []int, k int) {
	k = k % len(nums)
	reverse(nums)
	reverse(nums[:k])
	reverse(nums[k:])

}

// 除自身以外数组的乘积  给你一个整数数组 nums，返回 数组 answer ，其中 answer[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积 。
func productExceptSelf(nums []int) []int {
	n := len(nums)
	answer := make([]int, n)
	answer[0] = 1
	for i := 1; i < n; i++ {
		answer[i] = nums[i-1] * answer[i-1]
	}
	R := 1
	for i := n - 1; i >= 0; i-- {
		answer[i] *= R
		R *= nums[i]
	}
	return answer
}

// 缺失的第一个正数 给你一个未排序的整数数组 nums ，请你找出其中没有出现的最小的正整数。
// 请你实现时间复杂度为 O(n) 并且只使用常数级别额外空间的解决方案。
func firstMissingPositive(nums []int) int {
	n := len(nums)
	for i := 0; i < n; i++ {
		for nums[i] > 0 && nums[i] <= n && nums[i] != nums[nums[i]-1] {
			nums[i], nums[nums[i]-1] = nums[nums[i]-1], nums[i]
		}
	}
	for i := 0; i < n; i++ {
		if nums[i] != i+1 {
			return i + 1
		}
	}
	return n + 1
}
func main() {
	intervals := [][]int{}

	fmt.Println(merge(intervals))

	fmt.Println(firstMissingPositive([]int{3, 4, -1, 1}))
}
