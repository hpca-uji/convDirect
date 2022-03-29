/*
 * void gemm_microkernel_Cresident_neon_7x12_fixed_nopackA_unroll_4_fp32(int mr, int nr, int kc, float alpha,
                                                                         const float *Ar, int ldA,
                                                                         const float *Br,
                                                                         float beta,
                                                                         float *C, int ldC) {
*/
    // input operands
   __asm__ volatile
  (
  "                                \n\t"
  " ldr x30, %[ukc]                \n\t" // Load kc
  "                                \n\t"
  " ldr x18, %[uldA]               \n\t" // Load ldA
  " lsl x18, x18, #2               \n\t" // Actual stride is to be multiplied by 4 (sizeof(FP32))
  " ldr x17, %[Aaddr]              \n\t" // Load A address
  "                                \n\t"
  " ldr x19, %[Baddr]              \n\t" // Load B address
  "                                \n\t"
  " .LOOP_ITER:                    \n\t"
  "                                \n\t"
  " add x8,  x17, #0               \n\t" // Load address row 0 of A
  " add x9,  x8,  x18              \n\t" // Load address row 1 of A
  " add x10, x9,  x18              \n\t" // Load address row 2 of A
  " add x11, x10, x18              \n\t" // Load address row 3 of A
  " add x12, x11, x18              \n\t" // Load address row 4 of A
  " add x13, x12, x18              \n\t" // Load address row 5 of A
  " add x14, x13, x18              \n\t" // Load address row 6 of A
  "                                \n\t"
  " ldr q22, [x8]                  \n\t" // Load row 0 of A
  " ldr q23, [x9]                  \n\t" // Load row 1 of A
  " ldr q24, [x10]                 \n\t" // Load row 2 of A
  " ldr q25, [x11]                 \n\t" // Load row 3 of A
  " ldr q26, [x12]                 \n\t" // Load row 4 of A
  " ldr q27, [x13]                 \n\t" // Load row 5 of A
  " ldr q28, [x14]                 \n\t" // Load row 6 of A
"                                  \n\t"
"                                  \n\t" // Load row kc of B 
" ldr q29, [x19]                   \n\t" 
" ldr q30, [x19,#16]               \n\t"
" ldr q31, [x19,#32]               \n\t"
"                                  \n\t"
" fmla v1.4s, v29.4s, v22.4s[0]    \n\t" 
" fmla v2.4s, v30.4s, v22.4s[0]    \n\t" 
" fmla v3.4s, v31.4s, v22.4s[0]    \n\t" 
"                                  \n\t"
" fmla v4.4s, v29.4s, v23.4s[0]    \n\t" 
" fmla v5.4s, v30.4s, v23.4s[0]    \n\t" 
" fmla v6.4s, v31.4s, v23.4s[0]    \n\t" 
"                                  \n\t"
" fmla v7.4s, v29.4s, v24.4s[0]    \n\t" 
" fmla v8.4s, v30.4s, v24.4s[0]    \n\t" 
" fmla v9.4s, v31.4s, v24.4s[0]    \n\t" 
"                                  \n\t"
" fmla v10.4s, v29.4s, v25.4s[0]   \n\t" 
" fmla v11.4s, v30.4s, v25.4s[0]   \n\t" 
" fmla v12.4s, v31.4s, v25.4s[0]   \n\t" 
"                                  \n\t"
" fmla v13.4s, v29.4s, v26.4s[0]   \n\t" 
" fmla v14.4s, v30.4s, v26.4s[0]   \n\t" 
" fmla v15.4s, v31.4s, v26.4s[0]   \n\t" 
"                                  \n\t"
" fmla v16.4s, v29.4s, v27.4s[0]   \n\t" 
" fmla v17.4s, v30.4s, v27.4s[0]   \n\t" 
" fmla v18.4s, v31.4s, v27.4s[0]   \n\t" 
"                                  \n\t"
" fmla v19.4s, v29.4s, v28.4s[0]   \n\t" 
" fmla v20.4s, v30.4s, v28.4s[0]   \n\t" 
" fmla v21.4s, v31.4s, v28.4s[0]   \n\t" 
"                                  \n\t"
" add x19, x19, #48                \n\t"
" ldr q29, [x19]                   \n\t"  // Load row kc+1 of B 
" ldr q30, [x19,#16]               \n\t"
" ldr q31, [x19,#32]               \n\t"
"                                  \n\t"
" fmla v1.4s, v29.4s, v22.4s[1]    \n\t" 
" fmla v2.4s, v30.4s, v22.4s[1]    \n\t" 
" fmla v3.4s, v31.4s, v22.4s[1]    \n\t" 
"                                  \n\t"
" fmla v4.4s, v29.4s, v23.4s[1]    \n\t" 
" fmla v5.4s, v30.4s, v23.4s[1]    \n\t" 
" fmla v6.4s, v31.4s, v23.4s[1]    \n\t" 
"                                  \n\t"
" fmla v7.4s, v29.4s, v24.4s[1]    \n\t" 
" fmla v8.4s, v30.4s, v24.4s[1]    \n\t" 
" fmla v9.4s, v31.4s, v24.4s[1]    \n\t" 
"                                  \n\t"
" fmla v10.4s, v29.4s, v25.4s[1]   \n\t" 
" fmla v11.4s, v30.4s, v25.4s[1]   \n\t" 
" fmla v12.4s, v31.4s, v25.4s[1]   \n\t" 
"                                  \n\t"
" fmla v13.4s, v29.4s, v26.4s[1]   \n\t" 
" fmla v14.4s, v30.4s, v26.4s[1]   \n\t" 
" fmla v15.4s, v31.4s, v26.4s[1]   \n\t" 
"                                  \n\t"
" fmla v16.4s, v29.4s, v27.4s[1]   \n\t" 
" fmla v17.4s, v30.4s, v27.4s[1]   \n\t" 
" fmla v18.4s, v31.4s, v27.4s[1]   \n\t" 
"                                  \n\t"
" fmla v19.4s, v29.4s, v28.4s[1]   \n\t" 
" fmla v20.4s, v30.4s, v28.4s[1]   \n\t" 
" fmla v21.4s, v31.4s, v28.4s[1]   \n\t" 
"                                  \n\t"
" add x19, x19, #48                \n\t"
" ldr q29, [x19]                   \n\t"  // Load row kc+2 of B 
" ldr q30, [x19,#16]               \n\t"
" ldr q31, [x19,#32]               \n\t"
"                                  \n\t"
" fmla v1.4s, v29.4s, v22.4s[2]    \n\t" 
" fmla v2.4s, v30.4s, v22.4s[2]    \n\t" 
" fmla v3.4s, v31.4s, v22.4s[2]    \n\t" 
"                                  \n\t"
" fmla v4.4s, v29.4s, v23.4s[2]    \n\t" 
" fmla v5.4s, v30.4s, v23.4s[2]    \n\t" 
" fmla v6.4s, v31.4s, v23.4s[2]    \n\t" 
"                                  \n\t"
" fmla v7.4s, v29.4s, v24.4s[2]    \n\t" 
" fmla v8.4s, v30.4s, v24.4s[2]    \n\t" 
" fmla v9.4s, v31.4s, v24.4s[2]    \n\t" 
"                                  \n\t"
" fmla v10.4s, v29.4s, v25.4s[2]   \n\t" 
" fmla v11.4s, v30.4s, v25.4s[2]   \n\t" 
" fmla v12.4s, v31.4s, v25.4s[2]   \n\t" 
"                                  \n\t"
" fmla v13.4s, v29.4s, v26.4s[2]   \n\t" 
" fmla v14.4s, v30.4s, v26.4s[2]   \n\t" 
" fmla v15.4s, v31.4s, v26.4s[2]   \n\t" 
"                                  \n\t"
" fmla v16.4s, v29.4s, v27.4s[2]   \n\t" 
" fmla v17.4s, v30.4s, v27.4s[2]   \n\t" 
" fmla v18.4s, v31.4s, v27.4s[2]   \n\t" 
"                                  \n\t"
" fmla v19.4s, v29.4s, v28.4s[2]   \n\t" 
" fmla v20.4s, v30.4s, v28.4s[2]   \n\t" 
" fmla v21.4s, v31.4s, v28.4s[2]   \n\t" 
"                                  \n\t"
" add x19, x19, #48                \n\t"
" ldr q29, [x19]                   \n\t"  // Load row kc+3 of B 
" ldr q30, [x19,#16]               \n\t"
" ldr q31, [x19,#32]               \n\t"
"                                  \n\t"
" fmla v1.4s, v29.4s, v22.4s[3]    \n\t" 
" fmla v2.4s, v30.4s, v22.4s[3]    \n\t" 
" fmla v3.4s, v31.4s, v22.4s[3]    \n\t" 
"                                  \n\t"
" fmla v4.4s, v29.4s, v23.4s[3]    \n\t" 
" fmla v5.4s, v30.4s, v23.4s[3]    \n\t" 
" fmla v6.4s, v31.4s, v23.4s[3]    \n\t" 
"                                  \n\t"
" fmla v7.4s, v29.4s, v24.4s[3]    \n\t" 
" fmla v8.4s, v30.4s, v24.4s[3]    \n\t" 
" fmla v9.4s, v31.4s, v24.4s[3]    \n\t" 
"                                  \n\t"
" fmla v10.4s, v29.4s, v25.4s[3]   \n\t" 
" fmla v11.4s, v30.4s, v25.4s[3]   \n\t" 
" fmla v12.4s, v31.4s, v25.4s[3]   \n\t" 
"                                  \n\t"
" fmla v13.4s, v29.4s, v26.4s[3]   \n\t" 
" fmla v14.4s, v30.4s, v26.4s[3]   \n\t" 
" fmla v15.4s, v31.4s, v26.4s[3]   \n\t" 
"                                  \n\t"
" fmla v16.4s, v29.4s, v27.4s[3]   \n\t" 
" fmla v17.4s, v30.4s, v27.4s[3]   \n\t" 
" fmla v18.4s, v31.4s, v27.4s[3]   \n\t" 
"                                  \n\t"
" fmla v19.4s, v29.4s, v28.4s[3]   \n\t" 
" fmla v20.4s, v30.4s, v28.4s[3]   \n\t" 
" fmla v21.4s, v31.4s, v28.4s[3]   \n\t" 
"                                  \n\t"
  " add x17, x17, #16              \n\t" // Update address of A for next iteration
  " add x19, x19, #48              \n\t" // Update address of B for next iteration
  " sub x30, x30, #4               \n\t" // Decrease iteration count by 4 (unroll factor)
  " cmp x30, 0                     \n\t" // Check end of iteration count
  " b.ne .LOOP_ITER                \n\t"
  : // output operands (none)
  : 
    [ukc]   "m" (ukc),   // 0
    [Aaddr] "m" (Ar),    // 1
    [Baddr] "m" (Br),    // 2
    [uldA]  "m" (uldA)   // 5
  : // Register clobber list
    "x8",  "x9",  "x10", "x11", "x12", "x13", "x14", 
    "x17", "x18", "x19", "x30",
    "v1",  "v2",  "v3",  "v4",  "v5",  "v6",  "v7",
    "v8",  "v9",  "v10", "v11", "v12", "v13", "v14",
    "v15", "v16", "v17", "v18", "v19", "v20", "v21",
    "v22", "v23", "v24", "v25", "v26", "v27", "v28",
    "v29", "v30", "v31" 
  );
