    __asm__ volatile
  (
    // input operands
  " ldr x16, %[uldC]               \n\t" // Load ldC
  " lsl x16, x16, #2               \n\t" // Actual stride is to be multiplied by 4 (sizeof(FP32))
  " ldr x1,  %[Caddr]              \n\t" // Load address row 0 of C
  " add x2,  x1,  x16              \n\t" // Load address row 1 of C
  " add x3,  x2,  x16              \n\t" // Load address row 2 of C
  " add x4,  x3,  x16              \n\t" // Load address row 3 of C
  " add x5,  x4,  x16              \n\t" // Load address row 4 of C
  " add x6,  x5,  x16              \n\t" // Load address row 5 of C
  " add x7,  x6,  x16              \n\t" // Load address row 6 of C
  "                                \n\t"
  " ldr q1,  [x1]                  \n\t" // Load row 0 of C 
  " ldr q2,  [x1,#16]              \n\t"
  " ldr q3,  [x1,#32]              \n\t"
  " ldr q4,  [x2]                  \n\t" // Load row 1 of C 
  " ldr q5,  [x2,#16]              \n\t"
  " ldr q6,  [x2,#32]              \n\t"
  " ldr q7,  [x3]                  \n\t" // Load row 2 of C 
  " ldr q8,  [x3,#16]              \n\t"
  " ldr q9,  [x3,#32]              \n\t"
  " ldr q10, [x4]                  \n\t" // Load row 3 of C 
  " ldr q11, [x4,#16]              \n\t"
  " ldr q12, [x4,#32]              \n\t"
  " ldr q13, [x5]                  \n\t" // Load row 4 of C 
  " ldr q14, [x5,#16]              \n\t"
  " ldr q15, [x5,#32]              \n\t"
  " ldr q16, [x6]                  \n\t" // Load row 5 of C 
  " ldr q17, [x6,#16]              \n\t"
  " ldr q18, [x6,#32]              \n\t"
  " ldr q19, [x7]                  \n\t" // Load row 6 of C 
  " ldr q20, [x7,#16]              \n\t"
  " ldr q21, [x7,#32]              \n\t"
  "                                \n\t"
  : // output operands (none)
  : 
    [Caddr] "m" (Cptr),  // 0
    [uldC]  "m" (uldC)   // 1
  : // Register clobber list
    "x1",  "x2",  "x3",  "x4",  "x5",  "x6",  "x7", "x16"
  );
