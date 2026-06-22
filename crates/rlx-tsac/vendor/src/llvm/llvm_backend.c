/*
 * llvm_backend.c — LLVM JIT backend for TSAC-ng (experimental).
 *
 * Uses LLVM MCJIT to runtime-compile kernel IR tuned for the
 * current CPU. Provides tsac_llvm_init / tsac_llvm_decode /
 * tsac_llvm_shutdown.
 *
 * Build: cmake -DUSE_LLVM=ON (requires llvm-dev)
 */

#include "../tsac_codec.h"
#include "../dac_model.h"
#include <llvm-c/Core.h>
#include <llvm-c/ExecutionEngine.h>
#include <llvm-c/Target.h>
#include <llvm-c/Analysis.h>
#include <llvm-c/Transforms/PassBuilder.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef void (*jit_conv1d_fn)(float*, const float*, const float*, const float*,
                               int, int, int, int);
typedef void (*jit_convt_fn)(float*, const float*, const float*,
                              int, int, int, int, int);
typedef void (*jit_snake_fn)(float*, const float*, const float*, int, int);
typedef void (*jit_add_fn)(float*, const float*, const float*, int);

typedef struct {
    LLVMExecutionEngineRef ee;
    LLVMModuleRef         mod;
    int                   initialized;

    jit_conv1d_fn conv1d_jit;
    jit_convt_fn  convt_jit;
    jit_snake_fn  snake_jit;
    jit_add_fn    add_jit;
} LLVMBackend;

static LLVMValueRef ir_build_conv1d(LLVMModuleRef mod, LLVMBuilderRef b)
{
    LLVMTypeRef p_f32 = LLVMPointerType(LLVMFloatType(), 0);
    LLVMTypeRef i32   = LLVMInt32Type();
    LLVMTypeRef param_types[] = {p_f32, p_f32, p_f32, p_f32, i32, i32, i32, i32};
    LLVMTypeRef fn_type = LLVMFunctionType(LLVMVoidType(), param_types, 8, 0);
    LLVMValueRef fn = LLVMAddFunction(mod, "conv1d", fn_type);
    LLVMSetLinkage(fn, LLVMExternalLinkage);

    LLVMValueRef out  = LLVMGetParam(fn, 0);
    LLVMValueRef x    = LLVMGetParam(fn, 1);
    LLVMValueRef w    = LLVMGetParam(fn, 2);
    LLVMValueRef bias = LLVMGetParam(fn, 3);
    LLVMValueRef T    = LLVMGetParam(fn, 4);
    LLVMValueRef K    = LLVMGetParam(fn, 5);
    LLVMValueRef Ci   = LLVMGetParam(fn, 6);
    LLVMValueRef Co   = LLVMGetParam(fn, 7);

    LLVMBasicBlockRef entry = LLVMAppendBasicBlock(fn, "entry");
    LLVMPositionBuilderAtEnd(b, entry);

    LLVMValueRef P = LLVMBuildSDiv(b, K, LLVMConstInt(i32, 2, 0), "P");

    LLVMValueRef oc_init = LLVMConstInt(i32, 0, 0);
    LLVMValueRef oc_phi;
    LLVMBasicBlockRef oc_loop = LLVMAppendBasicBlock(fn, "oc_loop");
    LLVMBasicBlockRef oc_body = LLVMAppendBasicBlock(fn, "oc_body");
    LLVMBasicBlockRef oc_exit = LLVMAppendBasicBlock(fn, "oc_exit");

    LLVMBuildBr(b, oc_loop);
    LLVMPositionBuilderAtEnd(b, oc_loop);
    oc_phi = LLVMBuildPhi(b, i32, "oc");
    LLVMAddIncoming(oc_phi, &oc_init, &entry, 1);
    LLVMValueRef oc_cmp = LLVMBuildICmp(b, LLVMIntSLT, oc_phi, Co, "oc_cmp");
    LLVMBuildCondBr(b, oc_cmp, oc_body, oc_exit);

    LLVMPositionBuilderAtEnd(b, oc_body);
    LLVMValueRef bias_v = LLVMBuildLoad2(b, LLVMFloatType(),
        LLVMBuildGEP2(b, LLVMFloatType(), bias, &oc_phi, 1, ""), "bias_v");

    LLVMValueRef oi_init = LLVMConstInt(i32, 0, 0);
    LLVMValueRef oi_phi;
    LLVMBasicBlockRef oi_loop = LLVMAppendBasicBlock(fn, "oi_loop");
    LLVMBasicBlockRef oi_body = LLVMAppendBasicBlock(fn, "oi_body");
    LLVMBasicBlockRef oi_exit = LLVMAppendBasicBlock(fn, "oi_exit");

    LLVMBuildBr(b, oi_loop);
    LLVMPositionBuilderAtEnd(b, oi_loop);
    oi_phi = LLVMBuildPhi(b, i32, "oi");
    LLVMAddIncoming(oi_phi, &oi_init, &oc_body, 1);
    LLVMValueRef oi_cmp = LLVMBuildICmp(b, LLVMIntSLT, oi_phi, T, "oi_cmp");
    LLVMBuildCondBr(b, oi_cmp, oi_body, oi_exit);

    LLVMPositionBuilderAtEnd(b, oi_body);
    LLVMValueRef sum_phi;
    LLVMBasicBlockRef sum_init_bb = LLVMGetInsertBlock(b);

    LLVMValueRef ic_init = LLVMConstInt(i32, 0, 0);
    LLVMBasicBlockRef ic_loop = LLVMAppendBasicBlock(fn, "ic_loop");
    LLVMBasicBlockRef ic_body = LLVMAppendBasicBlock(fn, "ic_body");
    LLVMBasicBlockRef ic_exit = LLVMAppendBasicBlock(fn, "ic_exit");

    LLVMBuildBr(b, ic_loop);
    LLVMPositionBuilderAtEnd(b, ic_loop);
    LLVMValueRef ic_phi = LLVMBuildPhi(b, i32, "ic");
    LLVMAddIncoming(ic_phi, &ic_init, &sum_init_bb, 1);
    LLVMValueRef ic_cmp = LLVMBuildICmp(b, LLVMIntSLT, ic_phi, Ci, "ic_cmp");
    LLVMBuildCondBr(b, ic_cmp, ic_body, ic_exit);

    LLVMPositionBuilderAtEnd(b, ic_body);
    LLVMValueRef j_init = LLVMConstInt(i32, 0, 0);
    LLVMBasicBlockRef j_loop = LLVMAppendBasicBlock(fn, "j_loop");
    LLVMBasicBlockRef j_body = LLVMAppendBasicBlock(fn, "j_body");
    LLVMBasicBlockRef j_exit = LLVMAppendBasicBlock(fn, "j_exit");

    LLVMBuildBr(b, j_loop);
    LLVMPositionBuilderAtEnd(b, j_loop);
    LLVMValueRef j_phi = LLVMBuildPhi(b, i32, "j");
    LLVMAddIncoming(j_phi, &j_init, &ic_body, 1);
    LLVMValueRef j_cmp = LLVMBuildICmp(b, LLVMIntSLT, j_phi, K, "j_cmp");
    LLVMBuildCondBr(b, j_cmp, j_body, j_exit);

    LLVMPositionBuilderAtEnd(b, j_body);
    LLVMValueRef ii = LLVMBuildAdd(b, oi_phi, j_phi, "ii_tmp");
    ii = LLVMBuildSub(b, ii, P, "ii");
    LLVMValueRef valid = LLVMBuildAnd(b,
        LLVMBuildICmp(b, LLVMIntSGE, ii, LLVMConstInt(i32, 0, 0), "ge0"),
        LLVMBuildICmp(b, LLVMIntSLT, ii, T, "ltT"), "valid");

    LLVMBasicBlockRef valid_then = LLVMAppendBasicBlock(fn, "valid_then");
    LLVMBasicBlockRef valid_else = LLVMAppendBasicBlock(fn, "valid_else");
    LLVMBuildCondBr(b, valid, valid_then, valid_else);

    LLVMPositionBuilderAtEnd(b, valid_then);
    LLVMValueRef x_idx = LLVMBuildAdd(b,
        LLVMBuildMul(b, ic_phi, T, ""), ii, "x_idx");
    LLVMValueRef x_v = LLVMBuildLoad2(b, LLVMFloatType(),
        LLVMBuildGEP2(b, LLVMFloatType(), x, &x_idx, 1, ""), "x_v");
    LLVMValueRef w_idx1 = LLVMBuildMul(b, oc_phi, Ci, "");
    LLVMValueRef w_idx2 = LLVMBuildMul(b, w_idx1, K, "");
    LLVMValueRef w_idx3 = LLVMBuildAdd(b,
        LLVMBuildMul(b, ic_phi, K, ""), j_phi, "");
    LLVMValueRef w_idx = LLVMBuildAdd(b, w_idx2, w_idx3, "w_idx");
    LLVMValueRef w_v = LLVMBuildLoad2(b, LLVMFloatType(),
        LLVMBuildGEP2(b, LLVMFloatType(), w, &w_idx, 1, ""), "w_v");
    LLVMValueRef prod = LLVMBuildFMul(b, x_v, w_v, "prod");

    LLVMBuildBr(b, valid_else);
    LLVMPositionBuilderAtEnd(b, valid_else);
    LLVMValueRef prod_phi = LLVMBuildPhi(b, LLVMFloatType(), "prod_phi");
    LLVMAddIncoming(prod_phi, &prod, &valid_then, 1);
    LLVMValueRef zero = LLVMConstReal(LLVMFloatType(), 0.0);
    LLVMAddIncoming(prod_phi, &zero, &j_body, 1);

    LLVMValueRef j_next = LLVMBuildAdd(b, j_phi, LLVMConstInt(i32, 1, 0), "j_next");
    LLVMBuildBr(b, j_loop);
    LLVMAddIncoming(j_phi, &j_next, &valid_else, 1);

    LLVMPositionBuilderAtEnd(b, j_exit);
    LLVMValueRef ic_next = LLVMBuildAdd(b, ic_phi, LLVMConstInt(i32, 1, 0), "ic_next");
    LLVMBuildBr(b, ic_loop);
    LLVMAddIncoming(ic_phi, &ic_next, &j_exit, 1);

    LLVMPositionBuilderAtEnd(b, ic_exit);
    LLVMValueRef out_idx = LLVMBuildAdd(b,
        LLVMBuildMul(b, oc_phi, T, ""), oi_phi, "out_idx");
    LLVMBuildStore(b, bias_v, LLVMBuildGEP2(b, LLVMFloatType(), out, &out_idx, 1, ""));
    LLVMValueRef oi_next = LLVMBuildAdd(b, oi_phi, LLVMConstInt(i32, 1, 0), "oi_next");
    LLVMBuildBr(b, oi_loop);
    LLVMAddIncoming(oi_phi, &oi_next, &ic_exit, 1);

    LLVMPositionBuilderAtEnd(b, oi_exit);
    LLVMValueRef oc_next = LLVMBuildAdd(b, oc_phi, LLVMConstInt(i32, 1, 0), "oc_next");
    LLVMBuildBr(b, oc_loop);
    LLVMAddIncoming(oc_phi, &oc_next, &oi_exit, 1);

    LLVMPositionBuilderAtEnd(b, oc_exit);
    LLVMBuildRetVoid(b);

    return fn;
}

static LLVMValueRef ir_build_snake(LLVMModuleRef mod, LLVMBuilderRef b)
{
    LLVMTypeRef p_f32 = LLVMPointerType(LLVMFloatType(), 0);
    LLVMTypeRef i32   = LLVMInt32Type();
    LLVMTypeRef param_types[] = {p_f32, p_f32, p_f32, i32, i32};
    LLVMTypeRef fn_type = LLVMFunctionType(LLVMVoidType(), param_types, 5, 0);
    LLVMValueRef fn = LLVMAddFunction(mod, "snake", fn_type);

    LLVMValueRef o = LLVMGetParam(fn, 0);
    LLVMValueRef x = LLVMGetParam(fn, 1);
    LLVMValueRef a_ptr = LLVMGetParam(fn, 2);
    LLVMValueRef n = LLVMGetParam(fn, 3);
    LLVMValueRef C = LLVMGetParam(fn, 4);

    LLVMBasicBlockRef entry = LLVMAppendBasicBlock(fn, "entry");
    LLVMPositionBuilderAtEnd(b, entry);

    LLVMValueRef i_init = LLVMConstInt(i32, 0, 0);
    LLVMBasicBlockRef loop = LLVMAppendBasicBlock(fn, "loop");
    LLVMBasicBlockRef body = LLVMAppendBasicBlock(fn, "body");
    LLVMBasicBlockRef exit = LLVMAppendBasicBlock(fn, "exit");

    LLVMBuildBr(b, loop);
    LLVMPositionBuilderAtEnd(b, loop);
    LLVMValueRef i_phi = LLVMBuildPhi(b, i32, "i");
    LLVMAddIncoming(i_phi, &i_init, &entry, 1);
    LLVMValueRef cmp = LLVMBuildICmp(b, LLVMIntSLT, i_phi, n, "cmp");
    LLVMBuildCondBr(b, cmp, body, exit);

    LLVMPositionBuilderAtEnd(b, body);
    LLVMValueRef v_val = LLVMBuildLoad2(b, LLVMFloatType(),
        LLVMBuildGEP2(b, LLVMFloatType(), x, &i_phi, 1, ""), "v");
    LLVMValueRef a_idx = LLVMBuildURem(b, i_phi, C, "a_idx");
    LLVMValueRef al = LLVMBuildLoad2(b, LLVMFloatType(),
        LLVMBuildGEP2(b, LLVMFloatType(), a_ptr, &a_idx, 1, ""), "al");
    LLVMValueRef eps = LLVMConstReal(LLVMFloatType(), 1e-6);
    al = LLVMBuildSelect(b,
        LLVMBuildFCmp(b, LLVMRealOLT, al, eps, ""), eps, al, "al_clamp");
    LLVMValueRef av = LLVMBuildFMul(b, al, v_val, "av");
    /* Declare sinf */
    LLVMTypeRef sinf_param[] = {LLVMFloatType()};
    LLVMValueRef sinf_fn = LLVMGetNamedFunction(mod, "sinf");
    if (!sinf_fn) {
        LLVMTypeRef sinf_type = LLVMFunctionType(LLVMFloatType(), sinf_param, 1, 0);
        sinf_fn = LLVMAddFunction(mod, "sinf", sinf_type);
    }
    LLVMValueRef sin_args[] = {av};
    LLVMValueRef s = LLVMBuildCall2(b, LLVMGetElementType(LLVMTypeOf(sinf_fn)),
                                     sinf_fn, sin_args, 1, "s");
    LLVMValueRef s2 = LLVMBuildFMul(b, s, s, "s2");
    LLVMValueRef s2_div = LLVMBuildFDiv(b, s2, al, "s2a");
    LLVMValueRef res = LLVMBuildFAdd(b, v_val, s2_div, "res");
    LLVMBuildStore(b, res, LLVMBuildGEP2(b, LLVMFloatType(), o, &i_phi, 1, ""));

    LLVMValueRef i_next = LLVMBuildAdd(b, i_phi, LLVMConstInt(i32, 1, 0), "i_next");
    LLVMBuildBr(b, loop);
    LLVMAddIncoming(i_phi, &i_next, &body, 1);

    LLVMPositionBuilderAtEnd(b, exit);
    LLVMBuildRetVoid(b);
    return fn;
}

static LLVMValueRef ir_build_add(LLVMModuleRef mod, LLVMBuilderRef b)
{
    LLVMTypeRef p_f32 = LLVMPointerType(LLVMFloatType(), 0);
    LLVMTypeRef i32   = LLVMInt32Type();
    LLVMTypeRef param_types[] = {p_f32, p_f32, p_f32, i32};
    LLVMTypeRef fn_type = LLVMFunctionType(LLVMVoidType(), param_types, 4, 0);
    LLVMValueRef fn = LLVMAddFunction(mod, "add_jit", fn_type);

    LLVMValueRef o = LLVMGetParam(fn, 0);
    LLVMValueRef a = LLVMGetParam(fn, 1);
    LLVMValueRef b_p = LLVMGetParam(fn, 2);
    LLVMValueRef n = LLVMGetParam(fn, 3);

    LLVMBasicBlockRef entry = LLVMAppendBasicBlock(fn, "entry");
    LLVMPositionBuilderAtEnd(b, entry);

    LLVMValueRef i_init = LLVMConstInt(i32, 0, 0);
    LLVMBasicBlockRef loop = LLVMAppendBasicBlock(fn, "loop");
    LLVMBasicBlockRef body = LLVMAppendBasicBlock(fn, "body");
    LLVMBasicBlockRef exit_b = LLVMAppendBasicBlock(fn, "exit");

    LLVMBuildBr(b, loop);
    LLVMPositionBuilderAtEnd(b, loop);
    LLVMValueRef i_phi = LLVMBuildPhi(b, i32, "i");
    LLVMAddIncoming(i_phi, &i_init, &entry, 1);
    LLVMValueRef cmp = LLVMBuildICmp(b, LLVMIntSLT, i_phi, n, "cmp");
    LLVMBuildCondBr(b, cmp, body, exit_b);

    LLVMPositionBuilderAtEnd(b, body);
    LLVMValueRef av = LLVMBuildLoad2(b, LLVMFloatType(),
        LLVMBuildGEP2(b, LLVMFloatType(), a, &i_phi, 1, ""), "av");
    LLVMValueRef bv = LLVMBuildLoad2(b, LLVMFloatType(),
        LLVMBuildGEP2(b, LLVMFloatType(), b_p, &i_phi, 1, ""), "bv");
    LLVMValueRef sum = LLVMBuildFAdd(b, av, bv, "sum");
    LLVMBuildStore(b, sum, LLVMBuildGEP2(b, LLVMFloatType(), o, &i_phi, 1, ""));

    LLVMValueRef i_next = LLVMBuildAdd(b, i_phi, LLVMConstInt(i32, 1, 0), "i_next");
    LLVMBuildBr(b, loop);
    LLVMAddIncoming(i_phi, &i_next, &body, 1);

    LLVMPositionBuilderAtEnd(b, exit_b);
    LLVMBuildRetVoid(b);
    return fn;
}

static LLVMValueRef ir_build_convt(LLVMModuleRef mod, LLVMBuilderRef b)
{
    LLVMTypeRef p_f32 = LLVMPointerType(LLVMFloatType(), 0);
    LLVMTypeRef i32   = LLVMInt32Type();
    LLVMTypeRef param_types[] = {p_f32, p_f32, p_f32, i32, i32, i32, i32, i32};
    LLVMTypeRef fn_type = LLVMFunctionType(LLVMVoidType(), param_types, 8, 0);
    LLVMValueRef fn = LLVMAddFunction(mod, "convt", fn_type);

    LLVMValueRef o = LLVMGetParam(fn, 0);
    LLVMValueRef x = LLVMGetParam(fn, 1);
    LLVMValueRef w = LLVMGetParam(fn, 2);
    LLVMValueRef Ti = LLVMGetParam(fn, 3);
    LLVMValueRef To = LLVMGetParam(fn, 4);
    LLVMValueRef K = LLVMGetParam(fn, 5);
    LLVMValueRef Ci = LLVMGetParam(fn, 6);
    LLVMValueRef Co = LLVMGetParam(fn, 7);

    LLVMBasicBlockRef entry = LLVMAppendBasicBlock(fn, "entry");
    LLVMPositionBuilderAtEnd(b, entry);
    LLVMValueRef P = LLVMBuildSDiv(b, K, LLVMConstInt(i32, 2, 0), "P");
    LLVMValueRef S = LLVMConstInt(i32, 2, 0); /* stride=2 */

    /* For each ic, ii: scatter v * w[oc*Ci*K + ic*K + j] to o[oc*To + oi] */
    LLVMValueRef ic_init = LLVMConstInt(i32, 0, 0);
    LLVMBasicBlockRef ic_loop = LLVMAppendBasicBlock(fn, "ic_loop");
    LLVMBasicBlockRef ic_body = LLVMAppendBasicBlock(fn, "ic_body");
    LLVMBasicBlockRef ic_exit = LLVMAppendBasicBlock(fn, "ic_exit");
    LLVMBuildBr(b, ic_loop);

    LLVMPositionBuilderAtEnd(b, ic_loop);
    LLVMValueRef ic = LLVMBuildPhi(b, i32, "ic");
    LLVMAddIncoming(ic, &ic_init, &entry, 1);
    LLVMValueRef icc = LLVMBuildICmp(b, LLVMIntSLT, ic, Ci, "icc");
    LLVMBuildCondBr(b, icc, ic_body, ic_exit);

    LLVMPositionBuilderAtEnd(b, ic_body);
    LLVMValueRef ii_init = LLVMConstInt(i32, 0, 0);
    LLVMBasicBlockRef ii_loop = LLVMAppendBasicBlock(fn, "ii_loop");
    LLVMBasicBlockRef ii_body = LLVMAppendBasicBlock(fn, "ii_body");
    LLVMBasicBlockRef ii_exit = LLVMAppendBasicBlock(fn, "ii_exit");
    LLVMBuildBr(b, ii_loop);

    LLVMPositionBuilderAtEnd(b, ii_loop);
    LLVMValueRef ii = LLVMBuildPhi(b, i32, "ii");
    LLVMAddIncoming(ii, &ii_init, &ic_body, 1);
    LLVMValueRef iic = LLVMBuildICmp(b, LLVMIntSLT, ii, Ti, "iic");
    LLVMBuildCondBr(b, iic, ii_body, ii_exit);

    LLVMPositionBuilderAtEnd(b, ii_body);
    LLVMValueRef x_idx = LLVMBuildAdd(b, LLVMBuildMul(b, ic, Ti, ""), ii, "x_idx");
    LLVMValueRef v = LLVMBuildLoad2(b, LLVMFloatType(),
        LLVMBuildGEP2(b, LLVMFloatType(), x, &x_idx, 1, ""), "v");

    LLVMValueRef j_init = LLVMConstInt(i32, 0, 0);
    LLVMBasicBlockRef j_loop = LLVMAppendBasicBlock(fn, "j_loop2");
    LLVMBasicBlockRef j_body = LLVMAppendBasicBlock(fn, "j_body2");
    LLVMBasicBlockRef j_exit = LLVMAppendBasicBlock(fn, "j_exit2");
    LLVMBuildBr(b, j_loop);

    LLVMPositionBuilderAtEnd(b, j_loop);
    LLVMValueRef j = LLVMBuildPhi(b, i32, "j");
    LLVMAddIncoming(j, &j_init, &ii_body, 1);
    LLVMValueRef jc = LLVMBuildICmp(b, LLVMIntSLT, j, K, "jc");
    LLVMBuildCondBr(b, jc, j_body, j_exit);

    LLVMPositionBuilderAtEnd(b, j_body);
    LLVMValueRef oi_0 = LLVMBuildMul(b, ii, S, "");
    LLVMValueRef oi   = LLVMBuildAdd(b, LLVMBuildSub(b, oi_0, j, ""), P, "oi");
    LLVMValueRef oi_ok = LLVMBuildAnd(b,
        LLVMBuildICmp(b, LLVMIntSGE, oi, LLVMConstInt(i32, 0, 0), ""),
        LLVMBuildICmp(b, LLVMIntSLT, oi, To, ""), "oi_ok");

    LLVMBasicBlockRef oi_then = LLVMAppendBasicBlock(fn, "oi_then");
    LLVMBasicBlockRef oi_else = LLVMAppendBasicBlock(fn, "oi_else");
    LLVMBuildCondBr(b, oi_ok, oi_then, oi_else);

    LLVMPositionBuilderAtEnd(b, oi_then);
    LLVMValueRef oc_init2 = LLVMConstInt(i32, 0, 0);
    LLVMBasicBlockRef oc_loop2 = LLVMAppendBasicBlock(fn, "oc_loop2");
    LLVMBasicBlockRef oc_body2 = LLVMAppendBasicBlock(fn, "oc_body2");
    LLVMBasicBlockRef oc_exit2 = LLVMAppendBasicBlock(fn, "oc_exit2");
    LLVMBuildBr(b, oc_loop2);

    LLVMPositionBuilderAtEnd(b, oc_loop2);
    LLVMValueRef oc = LLVMBuildPhi(b, i32, "oc");
    LLVMAddIncoming(oc, &oc_init2, &oi_then, 1);
    LLVMValueRef occ = LLVMBuildICmp(b, LLVMIntSLT, oc, Co, "occ");
    LLVMBuildCondBr(b, occ, oc_body2, oc_exit2);

    LLVMPositionBuilderAtEnd(b, oc_body2);
    LLVMValueRef w_idx_a = LLVMBuildMul(b, oc, Ci, "");
    LLVMValueRef w_idx_b = LLVMBuildMul(b, w_idx_a, K, "");
    LLVMValueRef w_idx_c = LLVMBuildAdd(b, LLVMBuildMul(b, ic, K, ""), j, "");
    LLVMValueRef w_idx2 = LLVMBuildAdd(b, w_idx_b, w_idx_c, "w_idx");
    LLVMValueRef wv = LLVMBuildLoad2(b, LLVMFloatType(),
        LLVMBuildGEP2(b, LLVMFloatType(), w, &w_idx2, 1, ""), "wv");
    LLVMValueRef prod = LLVMBuildFMul(b, v, wv, "prod");
    LLVMValueRef out_idx2 = LLVMBuildAdd(b, LLVMBuildMul(b, oc, To, ""), oi, "out_idx");
    LLVMValueRef old = LLVMBuildLoad2(b, LLVMFloatType(),
        LLVMBuildGEP2(b, LLVMFloatType(), o, &out_idx2, 1, ""), "old");
    LLVMValueRef upd = LLVMBuildFAdd(b, old, prod, "upd");
    LLVMBuildStore(b, upd, LLVMBuildGEP2(b, LLVMFloatType(), o, &out_idx2, 1, ""));
    LLVMValueRef oc_next = LLVMBuildAdd(b, oc, LLVMConstInt(i32, 1, 0), "oc_next");
    LLVMBuildBr(b, oc_loop2);
    LLVMAddIncoming(oc, &oc_next, &oc_body2, 1);

    LLVMPositionBuilderAtEnd(b, oc_exit2);
    LLVMBuildBr(b, oi_else);
    LLVMPositionBuilderAtEnd(b, oi_else);
    LLVMValueRef j_next = LLVMBuildAdd(b, j, LLVMConstInt(i32, 1, 0), "j_next");
    LLVMBuildBr(b, j_loop);
    LLVMAddIncoming(j, &j_next, &oi_else, 1);

    LLVMPositionBuilderAtEnd(b, j_exit);
    LLVMValueRef ii_next = LLVMBuildAdd(b, ii, LLVMConstInt(i32, 1, 0), "ii_next");
    LLVMBuildBr(b, ii_loop);
    LLVMAddIncoming(ii, &ii_next, &j_exit, 1);

    LLVMPositionBuilderAtEnd(b, ii_exit);
    LLVMValueRef ic_next = LLVMBuildAdd(b, ic, LLVMConstInt(i32, 1, 0), "ic_next");
    LLVMBuildBr(b, ic_loop);
    LLVMAddIncoming(ic, &ic_next, &ii_exit, 1);

    LLVMPositionBuilderAtEnd(b, ic_exit);
    LLVMBuildRetVoid(b);
    return fn;
}

int tsac_llvm_init(void **priv)
{
    LLVMBackend *b = (LLVMBackend *)calloc(1, sizeof(LLVMBackend));
    if (!b) return TSAC_ERR_MEMORY;

    LLVMInitializeNativeTarget();
    LLVMInitializeNativeAsmPrinter();

    LLVMContextRef ctx = LLVMContextCreate();
    b->mod = LLVMModuleCreateWithNameInContext("tsac_kernels", ctx);
    LLVMBuilderRef builder = LLVMCreateBuilder();

    ir_build_conv1d(b->mod, builder);
    ir_build_convt(b->mod, builder);
    ir_build_snake(b->mod, builder);
    ir_build_add(b->mod, builder);

    LLVMDisposeBuilder(builder);

    /* Note: LLVM pass manager API changed in LLVM 17+.
     * Optimization passes can be added when building with older LLVM. */

    char *err = NULL;
    if (LLVMCreateExecutionEngineForModule(&b->ee, b->mod, &err)) {
        fprintf(stderr, "[llvm] JIT engine creation failed: %s\n", err ? err : "unknown");
        LLVMDisposeMessage(err);
        LLVMContextDispose(ctx);
        free(b);
        return TSAC_ERR_BACKEND;
    }

    b->conv1d_jit = (jit_conv1d_fn)LLVMGetFunctionAddress(b->ee, "conv1d");
    b->convt_jit  = (jit_convt_fn)LLVMGetFunctionAddress(b->ee, "convt");
    b->snake_jit  = (jit_snake_fn)LLVMGetFunctionAddress(b->ee, "snake");
    b->add_jit    = (jit_add_fn)LLVMGetFunctionAddress(b->ee, "add_jit");

    b->initialized = 1;
    *priv = b;

    fprintf(stderr, "[llvm] JIT backend initialized (conv1d=%p convt=%p snake=%p add=%p)\n",
            (void*)b->conv1d_jit, (void*)b->convt_jit,
            (void*)b->snake_jit, (void*)b->add_jit);

    return (b->conv1d_jit && b->snake_jit) ? TSAC_OK : TSAC_ERR_BACKEND;
}

int tsac_llvm_decode(void *priv, void *model,
                      const int *codebook_indices, int n_frames,
                      int n_codebooks, int block_len, int channels,
                      float *pcm, int n_samples)
{
    LLVMBackend *b = (LLVMBackend *)priv;
    (void)model; (void)codebook_indices; (void)n_codebooks; (void)channels;
    (void)pcm;

    if (!b || !b->initialized) return TSAC_ERR_BACKEND;

    /* Sanity test: run JIT conv1d on small buffer */
    float test_out[4] = {0}, test_in[] = {1,2,3,4}, test_w[] = {0.5f}, test_bias[] = {0.1f};
    b->conv1d_jit(test_out, test_in, test_w, test_bias, 4, 1, 1, 1);
    fprintf(stderr, "[llvm] JIT conv1d test: out[0]=%f (expect %f)\n",
            test_out[0], test_in[0] * test_w[0] + test_bias[0]);

    fprintf(stderr, "[llvm] decode: frames=%d samples=%d block_len=%d\n",
            n_frames, n_samples, block_len);

    /* Full decode graph not yet implemented — CPU fallback in tsac_codec.c handles this */
    return TSAC_ERR_BACKEND;
}

int tsac_llvm_encode(void *priv, void *model,
                      const float *pcm, int n_samples, int channels,
                      int n_codebooks, int block_len,
                      int **codebook_indices, int *n_frames)
{
    (void)priv; (void)model; (void)pcm; (void)n_samples; (void)channels;
    (void)n_codebooks; (void)block_len; (void)codebook_indices; (void)n_frames;
    return TSAC_ERR_BACKEND;
}

void tsac_llvm_shutdown(void *priv)
{
    LLVMBackend *b = (LLVMBackend *)priv;
    if (!b) return;
    if (b->ee) LLVMDisposeExecutionEngine(b->ee);
    free(b);
}
