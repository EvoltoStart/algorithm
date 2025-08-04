package main

import (
	"fmt"
)

// 无重复字符的最长子串
//func lengthOfLongestSubstring(s string) int {
//	maps := make(map[byte]int)
//	right, n := 0, len(s)
//	result := 0
//	for left := 0; left < n; left++ {
//		if left != 0 {
//			delete(maps, s[left-1])
//		}
//		if right >= n {
//			break
//		}
//		for right < n && maps[s[right]] == 0 {
//			maps[s[right]]++
//			right++
//		}
//		result = max(result, right-left)
//	}
//	return result
//
//}

func lengthOfLongestSubstring(s string) int {
	ans := 0
	cnt := [128]int{}
	left := 0
	for right, c := range s {
		cnt[c]++
		for cnt[c] > 1 {
			cnt[s[left]]--
			left++
		}
		ans = max(ans, right-left+1)
	}
	return ans
}

// 找到字符串中所有字母异位词
//func findAnagrams(s string, p string) []int {
//	sLen, pLen := len(s), len(p)
//	sCount, pCount := [26]int{}, [26]int{}
//	res := make([]int, 0)
//	if sLen < pLen {
//		return res
//	}
//	for i, str := range p {
//		sCount[s[i]-'a']++
//		pCount[str-'a']++
//	}
//	if sCount == pCount {
//		res = append(res, 0)
//	}
//	for i, str := range s[:sLen-pLen] {
//		sCount[str-'a']--
//		sCount[s[i+pLen]-'a']++
//		if sCount == pCount {
//			res = append(res, i+1)
//		}
//	}
//	return res
//}

func findAnagrams(s string, p string) []int {
	cntS := [26]int{}
	cntP := [26]int{}
	n := len(p)
	ans := []int{}
	for _, ch := range p {
		cntP[ch-'a']++
	}
	for right, ch := range s {
		cntS[ch-'a']++
		left := right - n + 1
		if left < 0 {
			continue
		}
		if cntP == cntS {
			ans = append(ans, left)
		}
		cntS[s[left]-'a']--
	}
	return ans
}

func main() {
	fmt.Println(lengthOfLongestSubstring("pwwkew"))
	fmt.Println(findAnagrams("cbacbabacd", "abc"))

}
