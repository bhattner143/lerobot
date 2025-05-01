#include "pch.h"

// mymath.cpp
extern "C" {
    __declspec(dllexport) int add(int a, int b) {
        return a + b;
    }
}