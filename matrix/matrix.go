package main

//矩阵

// 73. 矩阵置零 给定一个 m x n 的矩阵，如果一个元素为 0 ，则将其所在行和列的所有元素都设为 0 。请使用 原地 算法。
func setZeroes(matrix [][]int) {
	row, col := len(matrix), len(matrix[0])
	row_flag, col_flag := false, false
	//判断第一行是否包含0
	for i := 0; i < col; i++ {
		if matrix[0][i] == 0 {
			row_flag = true
			break
		}
	}
	//判断第一列是否包含0
	for i := 0; i < row; i++ {
		if matrix[i][0] == 0 {
			col_flag = true
			break
		}
	}
	//判断matrix[i][j]是否为0，将第一行和第一列对应标志位置0
	for i := 1; i < row; i++ {
		for j := 1; j < col; j++ {
			if matrix[i][j] == 0 {
				matrix[0][j], matrix[i][0] = 0, 0
			}
		}
	}
	//置0
	for i := 1; i < row; i++ {
		for j := 1; j < col; j++ {
			if matrix[0][j] == 0 || matrix[i][0] == 0 {
				matrix[i][j] = 0
			}
		}
	}
	if row_flag {
		for i := 0; i < col; i++ {
			matrix[0][i] = 0
		}
	}
	if col_flag {
		for i := 0; i < row; i++ {
			matrix[i][0] = 0
		}
	}
}

// 54. 螺旋矩阵 给你一个 m 行 n 列的矩阵 matrix ，请按照 顺时针螺旋顺序 ，返回矩阵中的所有元素。
func spiralOrder(matrix [][]int) []int {
	if len(matrix) == 0 || len(matrix[0]) == 0 {
		return []int{}
	}
	rows, cols := len(matrix), len(matrix[0])
	left, right, top, bottom := 0, cols-1, 0, rows-1
	order := make([]int, rows*cols)
	index := 0
	for left <= right && top <= bottom {
		for col := left; col <= right; col++ {
			order[index] = matrix[top][col]
			index++
		}
		for row := top + 1; row <= bottom; row++ {
			order[index] = matrix[row][right]
			index++
		}
		if left < right && top < bottom {
			for col := right - 1; col > left; col-- {
				order[index] = matrix[bottom][col]
				index++
			}
			for row := bottom; row > top; row-- {
				order[index] = matrix[row][left]
				index++
			}
		}
		left++
		right--
		bottom--
		top++
	}
	return order
}

// 48. 旋转图像 给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。
// 你必须在 原地 旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要 使用另一个矩阵来旋转图像。

func rotate(matrix [][]int) {
	n := len(matrix)
	for row := 0; row < n/2; row++ {
		for col := 0; col < (n+1)/2; col++ {
			matrix[row][col], matrix[n-col-1][row], matrix[n-row-1][n-col-1], matrix[col][n-row-1] = matrix[n-col-1][row], matrix[n-row-1][n-col-1], matrix[col][n-row-1], matrix[row][col]
		}
	}
}

// 搜索二维矩阵 II 编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性：
// 每行的元素从左到右升序排列。
// 每列的元素从上到下升序排列。
// 方法三：Z 字形查找
func searchMatrix(matrix [][]int, target int) bool {
	m, n := len(matrix), len(matrix[0])
	x, y := 0, n-1
	for x < m && y >= 0 {
		if matrix[x][y] == target {
			return true
		} else if matrix[x][y] > target {
			y--
		} else {
			x++
		}
	}
	return false
}
func main() {

}
