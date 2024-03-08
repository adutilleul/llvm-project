#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/InitializePasses.h"

// #include "llvm/CodeGen/MachinePostDominators.h"
using namespace llvm;

#define DEBUG_TYPE "mark-aliasing-load-store"

#include <limits>
#include <optional>
#include <tuple>

constexpr uint64_t MarkerLoop = std::numeric_limits<uint64_t>::max();
using AliasInformation = std::tuple<uint64_t, MachineInstr const *>;

namespace {
class MarkAliasingLoadStore : public MachineFunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid
  MarkAliasingLoadStore() : MachineFunctionPass(ID) {}

private:
  bool runOnMachineFunction(MachineFunction &MF) override;

  llvm::SmallVector<MachineInstr *> findStores(MachineFunction &MF);
  AliasInformation wrapComputeAliasDistance(const MachineInstr *StoreInstr,
                                            const MachineDominatorTree *MDT,
                                            const MachineLoopInfo *MLI,
                                            AAResults *AA,
                                            const TargetInstrInfo *TII) {
    llvm::DenseSet<MachineBasicBlock *> visited;
    return computeAliasDistance(StoreInstr, visited, StoreInstr, MDT, MLI, AA,
                                TII);
  }
  AliasInformation computeAliasDistance(
      const MachineInstr *CurrentBlockInstr,
      llvm::DenseSet<MachineBasicBlock *> &VisitedMBB,
      const MachineInstr *StoreInstr, const MachineDominatorTree *MDT,
      const MachineLoopInfo *MLI, AAResults *AA, const TargetInstrInfo *TII);

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<MachineDominatorTree>();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<MachineLoopInfo>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
} // end anonymous namespace

char MarkAliasingLoadStore::ID = 0;
char &llvm::MarkAliasingLoadStoreID = MarkAliasingLoadStore::ID;
INITIALIZE_PASS(MarkAliasingLoadStore, DEBUG_TYPE,
                "Annotate aliasing load store pairs", false, false)

bool MarkAliasingLoadStore::runOnMachineFunction(MachineFunction &MF) {

  bool Changed = false;

  MachineDominatorTree* MDT = &getAnalysis<MachineDominatorTree>();
  AAResults* AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();
  MachineLoopInfo* MLI = &getAnalysis<MachineLoopInfo>();
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();

  llvm::errs() << "MarkAliasingLoadStore Pass\n";
  llvm::errs() << MF.getName() << "\n";

  for (MachineInstr *store : findStores(MF)) {
    const auto [distance, load] = wrapComputeAliasDistance(store, MDT, MLI, AA, TII);

    if (load != nullptr) {

      llvm::errs() << "========================\n";

      llvm::errs() << "Store: \n";
      store->print(llvm::errs());

      llvm::errs() << "Distance: " << distance << "\n";
      llvm::errs() << "Aliasing load: \n";
      load->print(llvm::errs());
      llvm::errs() << "========================\n";
    }
  }

  return Changed;
}

llvm::SmallVector<MachineInstr *>
MarkAliasingLoadStore::findStores(MachineFunction &MF) {
  llvm::SmallVector<MachineInstr *> stores;
  for (auto I = MF.begin(), E = MF.end(); I != E; ++I) {
    MachineBasicBlock *MBB = &*I;
    for (auto MBBI = MBB->begin(), MBBE = MBB->end(); MBBI != MBBE;) {
      MachineInstr &MI = *MBBI++;
      if (!MI.memoperands_empty() && MI.mayStore()) {
        stores.push_back(&MI);
      }
    }
  }
  return stores;
}

AliasInformation MarkAliasingLoadStore::computeAliasDistance(
    const MachineInstr *CurrentBlockInstr,
    DenseSet<MachineBasicBlock *> &VisitedMBB, const MachineInstr *StoreInstr,
    const MachineDominatorTree *MDT, const MachineLoopInfo *MLI, AAResults *AA,
    const TargetInstrInfo *TII) {

  // TODO@: Derecursify this function using a worklist

  uint64_t AliasDistance = 0;
  MachineInstr const *AliasingLoad = nullptr;

  const MachineBasicBlock *MBB = CurrentBlockInstr->getParent();
  auto IsInstructionMatch = [&](MachineInstr const *I) {
    // Avoids matching X86 MFence and LFence instructions
    bool HasMemOp = !I->memoperands_empty();
    return HasMemOp && I->mayLoad() && I->mustAlias(AA, *StoreInstr, false);
  };

  for (auto &I : make_range(CurrentBlockInstr->getReverseIterator(),
                            CurrentBlockInstr->getParent()->instr_rend())) {
    if (IsInstructionMatch(&I)) {
      AliasingLoad = &I;
      return {AliasDistance, AliasingLoad};
    } else {
      const MachineLoop *CurLoop = MLI->getLoopFor(MBB);
      const MachineLoop *StoreLoop = MLI->getLoopFor(StoreInstr->getParent());

      for (MachineMemOperand *MMO : I.memoperands()) {
        if (MMO->isLoad()) {
          // If any load on the path is within a loop, we cannot
          // determine the trip count of the loop, so we conservatively
          // mark it with a special value.
          if ((CurLoop && !StoreLoop) ||
              (CurLoop && StoreLoop &&
               (CurLoop->getLoopDepth() > StoreLoop->getLoopDepth()))) {
            AliasDistance = MarkerLoop;
          } else {
            // llvm::errs() << "Insn: ";
            // I.print(llvm::errs());
            AliasDistance++;
          }
        }
      }
    }
  }

  const MachineLoop *CurLoop = MLI->getLoopFor(MBB);
  uint64_t MaxDist = std::numeric_limits<uint64_t>::min();

  llvm::SmallVector<std::tuple<MachineBasicBlock *, uint64_t>> PredDistanceVec;

  for (MachineBasicBlock *Pred : MBB->predecessors()) {
    const bool IsBackEdge = CurLoop && MBB == CurLoop->getHeader();
    const bool HasNotVisited = !VisitedMBB.count(Pred);
    const bool IsReachable = MDT->dominates(Pred, MBB);

    if (!IsBackEdge || (HasNotVisited && IsReachable)) {
      VisitedMBB.insert(Pred);

      const auto [PredDistance, PredAliasingLoad] = computeAliasDistance(
          &Pred->instr_back(), VisitedMBB, StoreInstr, MDT, MLI, AA, TII);

      PredDistanceVec.push_back({Pred, PredDistance});

      if (PredAliasingLoad != nullptr) {
        AliasingLoad = PredAliasingLoad;
        MaxDist = PredDistance;
      }
    }
  }

  for (const auto &[Pred, PredDistance] : PredDistanceVec) {
    if (AliasingLoad != nullptr &&
        MDT->dominates(AliasingLoad->getParent(), Pred)) {
      MaxDist = std::max(MaxDist, PredDistance);
    }
  }

  return {MaxDist + AliasDistance, AliasingLoad};
}
