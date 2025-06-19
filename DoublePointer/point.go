package main

import (
	"fmt"
	"slices"
)

// 移动零
func moveZeroes(nums []int) {
	left, right, n := 0, 0, len(nums)
	for right < n {
		if nums[right] != 0 {
			nums[left], nums[right] = nums[right], nums[left]
			left++
		}
		right++

	}
}

// 盛最多水的容器
//
//	func maxArea(height []int) int {
//		left, right := 0, len(height)-1
//		res := 0
//		for left < right {
//			if height[left] > height[right] {
//				res = max(min(height[left], height[right])*(right-left), res)
//				right--
//			} else {
//				res = max(min(height[left], height[right])*(right-left), res)
//				left++
//			}
//		}
//		return res
//	}
func maxArea(height []int) int {
	left, right := 0, len(height)-1
	res := 0
	for left < right {
		res = max(res, min(height[left], height[right])*(right-left))
		if height[left] < height[right] {
			left++
		} else {
			right--
		}
	}
	return res
}

// 三数之和
func threeSum(nums []int) [][]int {
	n := len(nums)
	res := [][]int{}
	slices.Sort(nums)
	for first := 0; first < n; first++ {
		if first > 0 && nums[first] == nums[first-1] {
			continue
		}
		third := n - 1
		target := -nums[first]
		for second := first + 1; second < n; second++ {
			if second > first+1 && nums[second] == nums[second-1] {
				continue
			}
			for second < third && nums[second]+nums[third] > target {
				third--
			}
			if second == third {
				break
			}
			if nums[second]+nums[third] == target {
				res = append(res, []int{nums[first], nums[second], nums[third]})
			}
		}
	}
	return res
}

// 接雨水
func trap(height []int) int {
	left, right := 0, len(height)-1
	leftMax, rightMax := 0, 0
	res := 0
	for left < right {
		leftMax = max(leftMax, height[left])
		rightMax = max(rightMax, height[right])
		if height[left] < height[right] {
			res += leftMax - height[left]
			left++
		} else {
			res += rightMax - height[right]
			right--
		}
	}
	return res
}
func main() {
	moveZeroes([]int{1})
	height := []int{1, 8, 6, 2, 5, 4, 8, 3, 7}
	fmt.Println(maxArea(height))
}
