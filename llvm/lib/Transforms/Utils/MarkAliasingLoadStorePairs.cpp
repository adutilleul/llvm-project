#include "llvm/Transforms/Utils/MarkAliasingLoadStorePairs.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Local.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>

#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

PreservedAnalyses MarkAliasingLoadStorePairs::run(Function &F,
                                                  FunctionAnalysisManager &AM) {

  auto *MSSA = &AM.getResult<MemorySSAAnalysis>(F);
  auto *DT = &AM.getResult<DominatorTreeAnalysis>(F);
  auto *AA = &AM.getResult<AAManager>(F);
  // get AAManager

  runImpl(F, AA, DT, &MSSA->getMSSA());
  llvm::errs() << F.getName() << "\n";
  return PreservedAnalyses::all();
}

void MarkAliasingLoadStorePairs::runImpl(Function &F, AAResults *AA_,
                                         DominatorTree *DT_, MemorySSA *MSSA_) {
  AA = AA_;
  DT = DT_;
  MSSA = MSSA_;
  MemorySSAUpdater MSSAU_(MSSA_);
  MSSAU = &MSSAU_;
  // applyMSSA(F);
  applyAA(F);
}


void MarkAliasingLoadStorePairs::applyAA(Function &F) {
  auto kindToString = [](AliasResult::Kind kind) {
    switch (kind) {
    case AliasResult::NoAlias:
      return "NoAlias";
    case AliasResult::MayAlias:
      return "MayAlias";
    case AliasResult::PartialAlias:
      return "PartialAlias";
    case AliasResult::MustAlias:
      return "MustAlias";
    }
    llvm_unreachable("Unknown alias kind");
  };

  for (auto &I : instructions(F)) {
    if (StoreInst *SI = dyn_cast<StoreInst>(&I)) {

      for (auto &ID : instructions(F)) {
        if (DT->dominates(&ID, &I)) {
          if (LoadInst *LI = dyn_cast<LoadInst>(&ID)) {
            AliasResult alias =
                AA->alias(MemoryLocation::get(SI), MemoryLocation::get(LI));
            AliasResult::Kind kind = alias;
            if(kind == AliasResult::NoAlias) {
            } 
            else 
            {
                llvm::errs() << "========================\n";
                llvm::errs() << "Kind: " << kindToString(kind) << "\n";
                llvm::errs() << "Load found\n";
                LI->print(llvm::errs());
                llvm::errs() << "\n";
                llvm::errs() << "Store found\n";
                SI->print(llvm::errs());
                llvm::errs() << "\n";

                if (alias.hasOffset()) {
                    llvm::errs() << "Offset: " << alias.getOffset() << "\n";
                }
                llvm::errs() << "========================\n";
            }
          }
        }
      }
    }
  }
}

void MarkAliasingLoadStorePairs::applyMSSA(Function &F) {
  for (auto &I : instructions(F)) {
    if (LoadInst *SI = dyn_cast<LoadInst>(&I)) {
      // MemoryLocation Loc = MemoryLocation::get(LI);
      // if (Loc.Size == 0)
      //   continue;
      llvm::errs() << "========================\n";
      llvm::errs() << "Store found\n";
      SI->print(llvm::errs());
      llvm::errs() << "\n";

      MemoryUseOrDef *MA = MSSA->getMemoryAccess(&I);
      MemoryAccess *Clobber = MSSA->getWalker()->getClobberingMemoryAccess(MA);
      // inst from memoryaccess
      llvm::errs() << "MemoryAccess: ";
      // MemoryAccess

      if (Clobber) {
        // @todo: insert and link call to function rn
        llvm::errs() << "Clobbering access found\n";
        Clobber->print(llvm::errs());
        llvm::errs() << "\n";
        if (auto *MU = dyn_cast<MemoryUse>(Clobber)) {
          llvm::errs() << "MemoryUse\n";
          llvm::errs() << "Defining instruction: ";
          if (auto *MI = MU->getMemoryInst()) {
            MI->print(llvm::errs());
          }
          llvm::errs() << "\n";
          //   llvm::errs() << "Defining access: ";
          //   if (auto *MD = MUD->getDefiningAccess()) {
          //     MD->print(llvm::errs());
          //   }

          //   llvm::errs() << "\n";
        }
        if (auto *MD = dyn_cast<MemoryDef>(Clobber)) {
          llvm::errs() << "MemoryDef\n";
          llvm::errs() << "Defining instruction: ";
          if (auto *MI = MD->getMemoryInst()) {
            MI->print(llvm::errs());
          }
          llvm::errs() << "\n";
        }
      }

      llvm::errs() << "========================\n";
    }
  }
}