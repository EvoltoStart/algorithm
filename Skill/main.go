package main

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
