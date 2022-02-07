/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \brief Example code on load and run TVM module.s
 * \file cpp_deploy.cc
 */
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <sys/time.h>
#include <cstdio>

void Verify(tvm::runtime::Module mod, std::string fname, int mr, int nr) {
  // Get the function from the module.
  tvm::runtime::PackedFunc f = mod.GetFunction(fname);
  ICHECK(f != nullptr);
  // Allocate the DLPack data structures.
  //
  // Note that we use TVM runtime API to allocate the DLTensor in this example.
  // TVM accept DLPack compatible DLTensors, so function can be invoked
  // as long as we pass correct pointer to DLTensor array.
  //
  // For more information please refer to dlpack.
  // One thing to notice is that DLPack contains alignment requirement for
  // the data pointer and TVM takes advantage of that.
  // If you plan to use your customized data container, please
  // make sure the DLTensor you pass in meet the alignment requirement.
  //
  DLTensor* A;
  DLTensor* B;
  DLTensor* C;
  int ndim = 2;
  int dtype_code = kDLFloat;
  int dtype_bits = 32;
  int dtype_lanes = 1;
  int device_type = kDLCPU;
  int device_id = 0;
    int m=mr;
    int n=nr;
    int k=512;
  int64_t shapeA[2] = {m,k};
  int64_t shapeB[2] = {k,n};
  int64_t shapeC[2] = {m,n};

  TVMArrayAlloc(shapeA, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &A);
  TVMArrayAlloc(shapeB, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &B);
  TVMArrayAlloc(shapeC, ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &C);
  //for (int i = 0; i < shape[0]; ++i) {
  //  static_cast<float*>(x->data)[i] = i;
  //}
  // Invoke the function
  // PackedFunc is a function that can be invoked via positional argument.
  struct timeval t_ini, t_end;
  // The signature of the function is specified in tvm.build
  int reps=10000000;
  gettimeofday (&t_ini, NULL);
  for(int i = 0; i < reps; i++){
      f(A, B, C);
  }
  gettimeofday (&t_end, NULL);
  double total =(t_end.tv_sec - t_ini.tv_sec) * 1000000 + t_end.tv_usec - t_ini.tv_usec;
  total=(total/1e6)/reps;
    double gflops = (2 * m * n * k)/(total *1e9);
    std::cout<<m<<"x"<<n<<" Time:"<<total<<" GFLOPS: "<<gflops<<std::endl;
  // Print out the output
  //for (int i = 0; i < shape[0]; ++i) {
  //  ICHECK_EQ(static_cast<float*>(y->data)[i], i + 1.0f);
  //}
  LOG(INFO) << "Finish verification...";
  TVMArrayFree(A);
  TVMArrayFree(B);
  TVMArrayFree(C);

}

void DeploySingleOp(int mr, int nr) {
  // Normally we can directly
  //tvm::runtime::Module mod_dylib = tvm::runtime::Module::LoadFromFile("lib/test_addone_dll.so");
  //LOG(INFO) << "Verify dynamic loading from test_addone_dll.so";
  //Verify(mod_dylib, "addone");
  // For libraries that are directly packed as system lib and linked together with the app
  // We can directly use GetSystemLib to get the system wide library.
  LOG(INFO) << "Verify load function from system lib";
  tvm::runtime::Module mod_syslib = (*tvm::runtime::Registry::Get("runtime.SystemLib"))();
  std::string name="microkernel_mult";
  
  Verify(mod_syslib, name, mr, nr);
}



int main(int argc ,char *argv[]) {
  DeploySingleOp(atoi(argv[1]), atoi(argv[2]));
  return 0;
}
