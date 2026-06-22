/*
 * llvm_jit_test.c — Minimal LLVM ORC JIT demo for conv1d kernel.
 *
 * Demonstrates: build LLVM IR for conv1d at runtime,
 * JIT compile, call the result as a function pointer.
 *
 * Compile:
 *   clang -O3 llvm_jit_test.c $(llvm-config --cflags --ldflags --libs core orcjit native) -o llvm_jit_test
 *   ./llvm_jit_test
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <llvm-c/Core.h>
#include <llvm-c/ExecutionEngine.h>
#include <llvm-c/Target.h>
#include <llvm-c/Analysis.h>

/* The JIT'd conv1d signature */
typedef void (*conv1d_fn)(float*, const float*, const float*, const float*,
                           int, int, int, int);

int main(void)
{
    LLVMInitializeNativeTarget();
    LLVMInitializeNativeAsmPrinter();

    LLVMContextRef ctx = LLVMContextCreate();
    LLVMModuleRef mod = LLVMModuleCreateWithNameInContext("conv1d_jit", ctx);

    /* Build function type: void(float*, float*, float*, float*, int, int, int, int) */
    LLVMTypeRef param_types[] = {
        LLVMPointerType(LLVMFloatType(), 0), /* out */
        LLVMPointerType(LLVMFloatType(), 0), /* in  */
        LLVMPointerType(LLVMFloatType(), 0), /* w   */
        LLVMPointerType(LLVMFloatType(), 0), /* bias */
        LLVMInt32Type(),  /* T  */
        LLVMInt32Type(),  /* K  */
        LLVMInt32Type(),  /* Ci */
        LLVMInt32Type(),  /* Co */
    };
    LLVMTypeRef fn_type = LLVMFunctionType(LLVMVoidType(), param_types, 8, 0);
    LLVMValueRef fn = LLVMAddFunction(mod, "conv1d", fn_type);
    LLVMSetLinkage(fn, LLVMExternalLinkage);

    /* Set parameter names */
    const char *names[] = {"out","in","w","bias","T","K","Ci","Co"};
    for (int i = 0; i < 8; i++)
        LLVMSetValueName(LLVMGetParam(fn, i), names[i]);

    /* Build a minimal IR: just copy input to output (placeholder) */
    /* TODO: generate actual conv1d loop IR */
    LLVMBasicBlockRef entry = LLVMAppendBasicBlock(fn, "entry");
    LLVMBuilderRef builder = LLVMCreateBuilder();
    LLVMPositionBuilderAtEnd(builder, entry);

    /* out[0] = in[0] + bias[0] — just verify JIT works */
    LLVMValueRef in0 = LLVMBuildLoad2(builder, LLVMFloatType(),
        LLVMGetParam(fn, 1), "in0");
    LLVMValueRef b0 = LLVMBuildLoad2(builder, LLVMFloatType(),
        LLVMGetParam(fn, 3), "b0");
    LLVMValueRef sum = LLVMBuildFAdd(builder, in0, b0, "sum");
    LLVMBuildStore(builder, sum, LLVMGetParam(fn, 0));

    LLVMBuildRetVoid(builder);

    /* Verify */
    char *err = NULL;
    LLVMVerifyModule(mod, LLVMAbortProcessAction, &err);
    LLVMDisposeMessage(err);

    /* JIT via MCJIT (simpler than ORC for this demo) */
    LLVMExecutionEngineRef ee;
    err = NULL;
    LLVMCreateExecutionEngineForModule(&ee, mod, &err);
    if (err) { fprintf(stderr, "JIT error: %s\n", err); return 1; }

    conv1d_fn jit_fn = (conv1d_fn)LLVMGetFunctionAddress(ee, "conv1d");

    /* Test */
    float out[4] = {}, in[] = {1,2,3,4}, w[] = {1}, bias[] = {5};
    jit_fn(out, in, w, bias, 4, 1, 1, 1);
    printf("JIT conv1d: out[0] = %f (expected %f)\n", out[0], in[0] + bias[0]);

    /* One-time cleanup */
    LLVMDisposeExecutionEngine(ee);
    LLVMContextDispose(ctx);

    printf("LLVM JIT: SUCCESS\n");
    return 0;
}
