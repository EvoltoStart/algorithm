package main

import "slices"

// 136. 只出现一次的数字
func singleNumber(nums []int) int {
	ans := 0
	for _, x := range nums {
		ans ^= x
	}
	return ans
}

// 169. 多数元素
func majorityElement(nums []int) int {
	ans, hp := 0, 0
	for _, x := range nums {
		if hp == 0 {
			ans, hp = x, 1
		} else if x == ans {
			hp++
		} else {
			hp--
		}
	}
	return ans
}

// 75. 颜色分类
func sortColors(nums []int) {
	p0, p1 := 0, 0
	for i, x := range nums {
		nums[i] = 2
		if x <= 1 {
			nums[p0] = 1
			p0++
		}
		if x == 0 {
			nums[p1] = 0
			p1++
		}
	}
}

// 31. 下一个排列
func nextPermutation(nums []int) {
	n := len(nums)
	i := n - 2
	for i >= 0 && nums[i] >= nums[i+1] {
		i--
	}
	if i >= 0 {
		j := n - 1
		for nums[i] >= nums[j] {
			j--
		}
		nums[i], nums[j] = nums[j], nums[i]
	}
	slices.Reverse(nums[i+1:])
}

// 287. 寻找重复数
func findDuplicate(nums []int) int {
	fast, slow := 0, 0
	for {
		slow = nums[slow]
		fast = nums[nums[fast]]
		if slow == fast {
			break
		}
	}
	head := 0
	for slow != head {
		slow = nums[slow]
		head = nums[head]
	}
	return slow
}
