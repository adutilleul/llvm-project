#ifndef LLVM_TRANSFORMS_UTILS_MARKALIASINGLOADSTOREPAIRS_H
#define LLVM_TRANSFORMS_UTILS_MARKALIASINGLOADSTOREPAIRS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class AAResults;
class AllocaInst;
class BatchAAResults;
class AssumptionCache;
class CallBase;
class CallInst;
class DominatorTree;
class Function;
class Instruction;
class LoadInst;
class MemCpyInst;
class MemMoveInst;
class MemorySSA;
class MemorySSAUpdater;
class MemSetInst;
class PostDominatorTree;
class StoreInst;
class TargetLibraryInfo;
class Value;

class MarkAliasingLoadStorePairs : public PassInfoMixin<MarkAliasingLoadStorePairs> {
  AAResults *AA = nullptr;
  DominatorTree *DT = nullptr;
  MemorySSA *MSSA = nullptr;
  MemorySSAUpdater *MSSAU = nullptr;
public:
  MarkAliasingLoadStorePairs() = default;

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  void runImpl(Function &F, AAResults*AA_, DominatorTree *DT_, MemorySSA *MSSA_);
  void applyMSSA(Function &F);
  void applyAA(Function &F);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_MARKALIASINGLOADSTOREPAIRS_H