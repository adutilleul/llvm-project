if(LLVM_INCLUDE_TESTS)
  add_subdirectory(MPFRWrapper)
endif()
if(LIBC_TARGET_OS_IS_GPU)
  add_subdirectory(gpu)
endif()
