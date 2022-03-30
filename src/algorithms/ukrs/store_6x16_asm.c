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
  "                                \n\t"
  " str q1,  [x1]                  \n\t" // Store row 0 of C 
  " str q2,  [x1,#16]              \n\t"
  " str q3,  [x1,#32]              \n\t"
  " str q4,  [x1,#48]              \n\t"
  " str q5,  [x2]                  \n\t" // Store row 1 of C 
  " str q6,  [x2,#16]              \n\t"
  " str q7,  [x2,#32]              \n\t"
  " str q8,  [x2,#48]              \n\t"
  " str q9,  [x3]                  \n\t" // Store row 2 of C 
  " str q10, [x3,#16]              \n\t"
  " str q11, [x3,#32]              \n\t"
  " str q12, [x3,#48]              \n\t"
  " str q13, [x4]                  \n\t" // Store row 3 of C 
  " str q14, [x4,#16]              \n\t"
  " str q15, [x4,#32]              \n\t"
  " str q16, [x4,#48]              \n\t"
  " str q17, [x5]                  \n\t" // Store row 4 of C 
  " str q18, [x5,#16]              \n\t"
  " str q19, [x5,#32]              \n\t"
  " str q20, [x5,#48]              \n\t"
  " str q21, [x6]                  \n\t" // Store row 5 of C 
  " str q22, [x6,#16]              \n\t"
  " str q23, [x6,#32]              \n\t"
  " str q24, [x6,#48]              \n\t"
  "                                \n\t"
  : // output operands (none)
  : 
    [Caddr] "m" (Cptr),  // 0
    [uldC]  "m" (uldC)   // 1
  : // Register clobber list
    "x1",  "x2",  "x3",  "x4",  "x5",  "x6",  "x16"
  );
