// Temp.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#include<iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
//#include <opencv2/opencv.hpp>

namespace py = pybind11;

int add(int i, int j) {
	return i + j;
}

py::array_t<int> add_arrays_1d(py::array_t<int>& input1, py::array_t<int>& input2) {
	py::buffer_info buf1 = input1.request();
	py::buffer_info buf2 = input2.request();

	if (buf1.ndim != 1 || buf2.ndim != 1) {
		throw std::runtime_error("number if dim must be 1");
	}

	if (buf1.size != buf2.size) {
		throw std::runtime_error("Input shape must match");
	}

	auto result = py::array_t<int>(buf1.size); // allocate the space
	py::buffer_info buf3 = result.request(); // get the struct

	double* ptr1 = (double*)buf1.ptr;
	double* ptr2 = (double*)buf2.ptr;
	double* ptr3 = (double*)buf3.ptr;

	//usr pointers to access ndarray
	for (int i = 0; i < buf1.shape[0]; i++) {
		ptr3[i] = ptr1[i] + ptr2[i];
	}
	return result;
}


py::list concat_arrays(py::array_t<int>& input1, py::array_t<int>& input2) {
	py::buffer_info buf1 = input1.request();
	py::buffer_info buf2 = input2.request();
    py::list result;
    result.append<py::array_t<int>&>(input1);
    result.append<py::array_t<int>&>(input2);
	return result;
}


PYBIND11_MODULE(example, m) {
   m.doc() = "pybind 11 example plugin"; // optional module docstring
   m.def("add_arrays_1d", &add_arrays_1d, "A function which adds two 1d array");
   m.def("concat_arrays",&concat_arrays, "concat two pyarray into a stl vector");
}