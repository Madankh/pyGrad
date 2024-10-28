In order to construct a tensor library, the first concept you need to learn obviously is: what is a tensor?

You may have an intuitive idea that a tensor is a mathematical concept of a n-dimensional data structure that contains some numbers. But here we need to understand how to model this data structure from a computational perspective. We can think of a tensor as consisting of the data itself and also some metadata describing aspects of the tensor such as its shape or the device it lives in (i.e. CPU memory, GPU memory…). There is also a less popular metadata that you may have never heard of, called stride. This concept is very important to understand the internals of tensor data rearrangement, so we need to discuss it a little more.

Imagine a 2-D tensor with shape [4, 8], The data (i.e. float numbers) of a tensor is actually stored as a 1-dimensional array on memory We have a matrix with 4 rows and 8 columns. Considering that all of its elements are organized by rows on the 1-dimensional array, if we want to access the value at position [2, 3], we need to traverse 2 rows (of 8 elements each) plus 3 positions. In mathematical terms, we need to traverse 3 + 2 * 8 elements on the 1-dimensional array. So this ‘8’ is the stride of the second dimension. In this case, it is the information of how many elements I need to traverse on the array to “jump” to other positions on the second dimension. Thus, for accessing the element [i, j] of a 2-dimensional tensor with shape [shape_0, shape_1], we basically need to access the element at position j + i * shape_1
