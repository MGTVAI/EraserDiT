// Isolated cuBLASLt heuristic screen. Same FP8/scales/BF16 bias/output/fast-accum.
#include <cublasLt.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
static std::string error;
static void check(cublasStatus_t s,const char*where){if(s)throw std::runtime_error(std::string(where)+":"+std::to_string(s));}
#define CK(x) check(x,#x)
struct Plan {
 cublasLtHandle_t h=nullptr;
 cublasLtMatmulDesc_t op=nullptr;
 cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr,d=nullptr;
 cublasLtMatmulPreference_t pref=nullptr;
 cublasLtMatmulHeuristicResult_t algos[128]; int count=0;
 size_t workspace=32ULL*1024*1024;
 ~Plan(){if(pref)cublasLtMatmulPreferenceDestroy(pref);if(a)cublasLtMatrixLayoutDestroy(a);if(b)cublasLtMatrixLayoutDestroy(b);if(c)cublasLtMatrixLayoutDestroy(c);if(d)cublasLtMatrixLayoutDestroy(d);if(op)cublasLtMatmulDescDestroy(op);if(h)cublasLtDestroy(h);}
};
extern "C" const char* lt_error(){return error.c_str();}
extern "C" size_t lt_version(){return cublasLtGetVersion();}
extern "C" void* lt_create(int m,int n,int k,void* sa,void* sb,void* bias,void* sd,int mode){
 Plan*p=new Plan;
 try{
 CK(cublasLtCreate(&p->h));
 CK(cublasLtMatmulDescCreate(&p->op,CUBLAS_COMPUTE_32F,CUDA_R_32F));
 cublasOperation_t ta=CUBLAS_OP_T,tb=CUBLAS_OP_N;
 CK(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));
 CK(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb)));
 CK(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&sa,sizeof(sa)));
 CK(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&sb,sizeof(sb)));
 int8_t fast=1;
 CK(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_FAST_ACCUM,&fast,sizeof(fast)));
 cublasLtEpilogue_t ep=mode==0 ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_GELU_BIAS;
 if(mode==2) CK(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_D_SCALE_POINTER,&sd,sizeof(sd)));
 CK(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_EPILOGUE,&ep,sizeof(ep)));
 CK(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_BIAS_POINTER,&bias,sizeof(bias)));
 cudaDataType_t bias_type=CUDA_R_16BF;
 CK(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,&bias_type,sizeof(bias_type)));
 // D^T = W * X^T; all actual allocations remain contiguous row-major.
 CK(cublasLtMatrixLayoutCreate(&p->a,CUDA_R_8F_E4M3,k,n,k));
 CK(cublasLtMatrixLayoutCreate(&p->b,CUDA_R_8F_E4M3,k,m,k));
 CK(cublasLtMatrixLayoutCreate(&p->c,CUDA_R_16BF,n,m,n));
 CK(cublasLtMatrixLayoutCreate(&p->d,mode==2 ? CUDA_R_8F_E4M3 : CUDA_R_16BF,n,m,n));
 CK(cublasLtMatmulPreferenceCreate(&p->pref));
 CK(cublasLtMatmulPreferenceSetAttribute(p->pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&p->workspace,sizeof(p->workspace)));
 CK(cublasLtMatmulAlgoGetHeuristic(p->h,p->op,p->a,p->b,p->c,p->d,p->pref,128,p->algos,&p->count));
 return p;
 }catch(const std::exception&e){error=e.what();delete p;return nullptr;}
}
extern "C" int lt_count(void*handle){return ((Plan*)handle)->count;}
extern "C" int lt_info(void*handle,int idx,int*info,float*waves,size_t*workspace){
 Plan*p=(Plan*)handle;if(idx<0||idx>=p->count)return -1;
 auto&r=p->algos[idx];*waves=r.wavesCount;*workspace=r.workspaceSize;
 cublasLtMatmulAlgoConfigAttributes_t fields[]={CUBLASLT_ALGO_CONFIG_ID,CUBLASLT_ALGO_CONFIG_TILE_ID,CUBLASLT_ALGO_CONFIG_STAGES_ID,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION};
 for(int j=0;j<7;++j){size_t size=0;info[j]=-1;cublasLtMatmulAlgoConfigGetAttribute(&r.algo,fields[j],info+j,sizeof(int),&size);}
 return (int)r.state;
}
extern "C" int lt_run(void*handle,int idx,void*w,void*x,void*y,void*ws,void*stream,int reps){
 Plan*p=(Plan*)handle;if(idx<0||idx>=p->count)return -1;
 const float alpha=1,beta=0;
 for(int i=0;i<reps;++i){auto s=cublasLtMatmul(p->h,p->op,&alpha,w,p->a,x,p->b,&beta,y,p->c,y,p->d,&p->algos[idx].algo,ws,p->workspace,(cudaStream_t)stream);if(s)return (int)s;}
 return 0;
}
extern "C" void lt_destroy(void*handle){delete (Plan*)handle;}
