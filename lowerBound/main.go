package main

// lowerBound 返回最小的满足 nums[i] >= target 的 i
// 如果数组为空，或者所有数都 < target，则返回 nums.length
// 要求 nums 是非递减的，即 nums[i] <= nums[i + 1]

// 闭区间写法
func lowerBound(nums []int, target int) int {
	left, right := 0, len(nums)-1 // 闭区间 [left, right]
	for left <= right {           // 区间不为空
		// 循环不变量：
		// nums[left-1] < target
		// nums[right+1] >= target
		mid := left + (right-left)/2
		if nums[mid] < target {
			left = mid + 1 // 范围缩小到 [mid+1, right]
		} else {
			right = mid - 1 // 范围缩小到 [left, mid-1]
		}
	}
	return left
}

// 左闭右开区间写法
func lowerBound2(nums []int, target int) int {
	left, right := 0, len(nums) // 左闭右开区间 [left, right)
	for left < right {          // 区间不为空
		// 循环不变量：
		// nums[left-1] < target
		// nums[right] >= target
		mid := left + (right-left)/2
		if nums[mid] < target {
			left = mid + 1 // 范围缩小到 [mid+1, right)
		} else {
			right = mid // 范围缩小到 [left, mid)
		}
	}
	return left // 或者 right
}

// 开区间写法
func lowerBound3(nums []int, target int) int {
	left, right := -1, len(nums) // 开区间 (left, right)
	for left+1 < right {         // 区间不为空
		// 循环不变量：
		// nums[left] < target
		// nums[right] >= target
		mid := left + (right-left)/2
		if nums[mid] < target {
			left = mid // 范围缩小到 (mid, right)
		} else {
			right = mid // 范围缩小到 (left, mid)
		}
	}
	return right
}

func searchInsert(nums []int, target int) int {
	return lowerBound(nums, target) // 选择其中一种写法即可
}

func searchMatrix(matrix [][]int, target int) bool {
	m, n := len(matrix), len(matrix[0])
	i, j := 0, n-1
	for i < m && j >= 0 { // 还有剩余元素
		if matrix[i][j] == target {
			return true // 找到 target
		}
		if matrix[i][j] < target {
			i++ // 这一行剩余元素全部小于 target，排除
		} else {
			j-- // 这一列剩余元素全部大于 target，排除
		}
	}
	return false
}

// lowerBound 返回最小的满足 nums[i] >= target 的下标 i
// 如果数组为空，或者所有数都 < target，则返回 len(nums)
// 要求 nums 是非递减的，即 nums[i] <= nums[i + 1]
func lowerBound(nums []int, target int) int {
	left, right := 0, len(nums)-1 // 闭区间 [left, right]
	for left <= right {           // 区间不为空
		// 循环不变量：
		// nums[left-1] < target
		// nums[right+1] >= target
		mid := left + (right-left)/2
		if nums[mid] >= target {
			right = mid - 1 // 范围缩小到 [left, mid-1]
		} else {
			left = mid + 1 // 范围缩小到 [mid+1, right]
		}
	}
	// 循环结束后 left = right+1
	// 此时 nums[left-1] < target 而 nums[left] = nums[right+1] >= target
	// 所以 left 就是第一个 >= target 的元素下标
	return left
}

func searchRange(nums []int, target int) []int {
	start := lowerBound(nums, target)
	if start == len(nums) || nums[start] != target {
		return []int{-1, -1} // nums 中没有 target
	}
	// 如果 start 存在，那么 end 必定存在
	end := lowerBound(nums, target+1) - 1
	return []int{start, end}
}
