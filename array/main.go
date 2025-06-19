package main

import (
	"fmt"
	"slices"
)

// 合并两个有序数组
func merge(nums1 []int, m int, nums2 []int, n int) {
	i := m + n - 1
	mm := m - 1
	nn := n - 1
	for {
		if mm < 0 {
			copy(nums1[0:], nums2[0:nn+1])
			return
		}
		if nn < 0 {
			return
		}
		if nums1[mm] > nums2[nn] {
			nums1[i] = nums1[mm]
			mm--
		} else {
			nums1[i] = nums2[nn]
			nn--
		}
		i--
	}

}

// 移除元素
func removeElement(nums []int, val int) int {
	left, right := 0, len(nums)-1
	for right > left {
		if nums[left] == val {
			nums[left] = nums[right]
			right--
		} else {
			left++
		}
	}
	return left + 1
}

//	func twoSum(nums []int, target int) []int {
//		maps := make(map[int]int)
//		for i := 0; i < len(nums); i++ {
//			if h, ok := maps[target-nums[i]]; ok {
//				return []int{h, i}
//			}
//			maps[nums[i]] = i
//		}
//		return nil
//	}
//
// 两数之和
func twoSum(nums []int, target int) []int {
	maps := make(map[int]int)
	for i := 0; i < len(nums); i++ {
		num := target - nums[i]
		if _, ok := maps[num]; ok {
			return []int{maps[num], i}
		}
		maps[nums[i]] = i
	}
	return nil
}

// 字母异位词分组
//
//	func groupAnagrams(strs []string) [][]string {
//		maps := make(map[string][]string)
//		res := make([][]string, 0)
//		for _, str := range strs {
//			s := []byte(str)
//			slices.Sort(s)
//			st := string(s)
//			maps[st] = append(maps[st], str)
//		}
//		for _, strs := range maps {
//			res = append(res, strs)
//		}
//		return res
//	}
func groupAnagrams(strs []string) [][]string {
	maps := make(map[[26]byte][]string)
	for _, str := range strs {
		cnt := [26]byte{}
		for _, ch := range str {
			cnt[ch-'a']++
		}
		maps[cnt] = append(maps[cnt], str)
	}
	var result [][]string
	for _, strs := range maps {
		result = append(result, strs)
	}
	return result
}

type Person struct {
	Name string
	Age  int
}

func set(person Person) {
	person.Age = 18

}

//	func longestConsecutive(nums []int) int {
//		numSet := map[int]bool{}
//		for _, num := range nums {
//			numSet[num] = true
//		}
//		longest := 0
//		for num := range numSet {
//			if !numSet[num-1] {
//				currentNum := num
//				currentLen := 1
//				for numSet[currentNum+1] {
//					currentNum++
//					currentLen++
//				}
//				if currentLen > longest {
//					longest = currentLen
//				}
//			}
//
//		}
//		return longest
//	}
//
// 最长连续序列
func longestConsecutive(nums []int) int {
	slices.Sort(nums)
	if len(nums) == 0 {
		return 0
	}
	pre := nums[0]
	currentLen := 1
	result := 1
	if len(nums) == 0 {
		return 0
	}
	for i := 1; i < len(nums); i++ {
		if pre == nums[i] {
			continue
		}
		if nums[i] == pre+1 {
			currentLen++
		} else {
			if currentLen > result {
				result = currentLen
			}
			currentLen = 1
		}
		pre = nums[i]
	}
	if result < currentLen {
		result = currentLen
	}
	return result
}

func main() {
	nums1 := []int{5, 6, 0, 0, 0}
	nums2 := []int{2, 5, 6}
	nums3 := []int{0, 2, 3, 5, 7, 10}
	merge(nums1, 2, nums2, 3)
	fmt.Println(nums1)
	removeElement(nums1, 2)
	fmt.Println(nums1)
	fmt.Println(twoSum(nums3, 3))
	a := 10
	ptr := &a
	*ptr = 100
	fmt.Println(a)
	fmt.Printf("ptr: %p\n,T:%T,V:%v", *ptr, *ptr, *ptr)
	fmt.Println(twoSum(nums3, 5))

}
