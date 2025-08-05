package main

import "fmt"

// 和为 K 的子数组
//func subarraySum(nums []int, k int) int {
//	maps := make(map[int]int)
//	pre := 0
//	count := 0
//	maps[0] = 1
//	for _, v := range nums {
//		pre += v
//		count = count + maps[pre-k]
//		maps[pre]++
//	}
//	return count
//}

func subarraySum(nums []int, k int) int {
	ans, pre := 0, 0
	mp := make(map[int]int, len(nums))
	for _, num := range nums {
		mp[pre]++
		pre += num
		ans += mp[pre-k]
	}
	return ans
}

// 滑动窗口最大值
//func maxSlidingWindow(nums []int, k int) []int {
//	n := len(nums)
//	prefixMax := make([]int, n)
//	suffixMax := make([]int, n)
//	for i, j := 0, n-1; i < n && j >= 0; i, j = i+1, j-1 {
//		if i%k == 0 {
//			prefixMax[i] = nums[i]
//		} else {
//			prefixMax[i] = max(nums[i], prefixMax[i-1])
//		}
//		if (j+1)%k == 0 || j == n-1 {
//			suffixMax[j] = nums[j]
//		} else {
//			suffixMax[j] = max(nums[j], suffixMax[j+1])
//		}
//	}
//
//	result := make([]int, n-k+1)
//	for i := range result {
//		result[i] = max(suffixMax[i], prefixMax[i+k-1])
//	}
//	return result
//}

func maxSlidingWindow(nums []int, k int) []int {
	ans := make([]int, len(nums)-k+1)
	q := []int{}
	for i, n := range nums {
		for len(q) > 0 && nums[q[len(q)-1]] < n {
			q = q[:len(q)-1] //弹出队尾元素
		}
		q = append(q, i)
		left := i - k + 1
		if q[0] < left {
			q = q[1:] // 弹出队首元素
		}
		if left >= 0 {
			ans[left] = nums[q[0]]
		}
	}
	return ans
}

// 76. 最小覆盖子串
func minWindow(s, t string) string {
	cnt := [128]int{}
	ansLeft, ansRight := -1, len(s)
	less, left := 0, 0
	for _, ch := range t {
		if cnt[ch] == 0 {
			less++
		}
		cnt[ch]++
	}
	for right, ch := range s {
		cnt[ch]--
		if cnt[ch] == 0 {
			less--
		}
		for less == 0 {
			if ansRight-ansLeft > right-left {
				ansLeft, ansRight = left, right
			}
			if cnt[s[left]] == 0 {
				less++
			}
			cnt[s[left]]++
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
