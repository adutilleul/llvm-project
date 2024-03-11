#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "mark-aliasing-load-store"

#include <limits>
#include <optional>
#include <tuple>

constexpr uint64_t MarkerLoop = std::numeric_limits<uint64_t>::max();

using AliasInformation = std::tuple<uint64_t, MachineInstr *>;
using PredDistanceVectors =
    SmallVector<std::tuple<MachineBasicBlock *, uint64_t>>;
using PredDistanceMap = DenseMap<MachineBasicBlock *, PredDistanceVectors>;
using MachineInstrOrBlock = std::variant<MachineInstr *, MachineBasicBlock *>;

namespace {
class MarkAliasingLoadStore : public MachineFunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid
  MarkAliasingLoadStore() : MachineFunctionPass(ID) {}

private:
  bool runOnMachineFunction(MachineFunction &MF) override;

  bool insertMarker(MachineBasicBlock *MBB, MachineInstr *InsertBefore,
                    MachineInstr *LoadInst);

  llvm::SmallVector<MachineInstr *>
  filterInstructions(MachineFunction &MF,
                     std::function<bool(MachineInstr *)> pred);

  llvm::SmallVector<MachineInstr *> findStores(MachineFunction &MF) {
    return filterInstructions(MF, [](MachineInstr *I) {
      return !I->memoperands_empty() && I->mayStore();
    });
  }

  llvm::SmallVector<MachineInstr *> findLoads(MachineFunction &MF) {
    return filterInstructions(MF, [](MachineInstr *I) {
      return !I->memoperands_empty() && I->mayLoad();
    });
  }

  bool hasLoopWithinPath(const MachineLoop *StoreML,
                         const MachineLoop *LoadML) {
    const bool IsLoadWithinLoop = (LoadML && !StoreML);
    const bool BothLoops = (StoreML && LoadML);

    if (IsLoadWithinLoop) {
      return true;
    }

    if (BothLoops && StoreML != LoadML) {
      return true;
    }

    return false;
  }

  AliasInformation wrapComputeAliasDistance(
      MachineInstr *StoreInstr, DominatorTree *DT, LoopInfo *LI,
      MachineDominatorTree *MDT, MachinePostDominatorTree *MPDT,
      const MachineLoopInfo *MLI, AAResults *AA, const TargetInstrInfo *TII);

  AliasInformation computeAliasDistance(
      MachineInstrOrBlock InstrOrBlock,
      llvm::DenseSet<MachineBasicBlock *> &VisitedMBB,
      PredDistanceMap &PredDistanceMap, const MachineInstr *StoreInstr,
      const MachineDominatorTree *MDT, const MachineLoopInfo *MLI,
      AAResults *AA, const TargetInstrInfo *TII);

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<MachineDominatorTree>();
    AU.addRequired<MachinePostDominatorTree>();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<MachineLoopInfo>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
} // end anonymous namespace

char MarkAliasingLoadStore::ID = 0;
char &llvm::MarkAliasingLoadStoreID = MarkAliasingLoadStore::ID;
INITIALIZE_PASS(MarkAliasingLoadStore, DEBUG_TYPE,
                "Annotate aliasing load store pairs", false, false)

bool MarkAliasingLoadStore::insertMarker(MachineBasicBlock *MBB,
                                         MachineInstr *InsertBefore,
                                         MachineInstr *LoadInst) {
  const TargetInstrInfo *TII = MBB->getParent()->getSubtarget().getInstrInfo();
  // const TargetRegisterInfo *TRI =
  //     MBB->getParent()->getSubtarget().getRegisterInfo();
  const DebugLoc &DL = InsertBefore->getDebugLoc();
  const unsigned Marker = TII->get(TargetOpcode::COPY).getOpcode();
  const llvm::Register MarkerReg = LoadInst->getOperand(0).getReg();

  BuildMI(*MBB, InsertBefore, DL, TII->get(Marker))
      .addReg(MarkerReg)
      .addReg(MarkerReg);

  return true;
}

bool MarkAliasingLoadStore::runOnMachineFunction(MachineFunction &MF) {

  bool Changed = false;
  DominatorTree *DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  MachineDominatorTree *MDT = &getAnalysis<MachineDominatorTree>();
  MachinePostDominatorTree *MPDT = &getAnalysis<MachinePostDominatorTree>();
  AAResults *AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();
  MachineLoopInfo *MLI = &getAnalysis<MachineLoopInfo>();
  LoopInfo *LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();

  llvm::errs() << "MarkAliasingLoadStore Pass\n";
  llvm::errs() << MF.getName() << "\n";

  const auto AllLoads = findLoads(MF);
  llvm::errs() << "Loads: " << AllLoads.size() << "\n";

  for (MachineInstr *store : findStores(MF)) {
    const auto [distance, load] =
        wrapComputeAliasDistance(store, DT, LI, MDT, MPDT, MLI, AA, TII);

    if (load != nullptr) {
      llvm::errs() << "========================\n";

      llvm::errs() << "Store: \n";
      store->print(llvm::errs());

      if (distance == MarkerLoop)
        llvm::errs() << "Distance: Unbounded because of a loop\n";
      else
        llvm::errs() << "Distance: " << distance << "\n";

      insertMarker(load->getParent(), load, load);

      llvm::errs() << "Aliasing load: \n";
      load->print(llvm::errs());

      llvm::errs() << "========================\n";
    }
  }

  return Changed;
}

llvm::SmallVector<MachineInstr *> MarkAliasingLoadStore::filterInstructions(
    MachineFunction &MF, std::function<bool(MachineInstr *)> pred) {
  llvm::SmallVector<MachineInstr *> stores;
  for (auto I = MF.begin(), E = MF.end(); I != E; ++I) {
    MachineBasicBlock *MBB = &*I;
    for (auto MBBI = MBB->begin(), MBBE = MBB->end(); MBBI != MBBE;) {
      MachineInstr &MI = *MBBI++;
      if (pred(&MI)) {
        stores.push_back(&MI);
      }
    }
  }
  return stores;
}

AliasInformation MarkAliasingLoadStore::wrapComputeAliasDistance(
    MachineInstr *StoreInstr, DominatorTree *DT, LoopInfo *LI,
    MachineDominatorTree *MDT, MachinePostDominatorTree *MPDT,
    const MachineLoopInfo *MLI, AAResults *AA, const TargetInstrInfo *TII) {
  llvm::DenseSet<MachineBasicBlock *> Visited;
  llvm::DenseMap<MachineBasicBlock *, PredDistanceVectors> PredDistanceMap;
  auto [Distance, AliasingLoad] =
      computeAliasDistance(MachineInstrOrBlock{StoreInstr}, Visited,
                           PredDistanceMap, StoreInstr, MDT, MLI, AA, TII);
  if (AliasingLoad != nullptr) {

    SmallSet<MachineBasicBlock *, 4> IgnoredBranches;
    for (MachineBasicBlock *Pred : AliasingLoad->getParent()->predecessors()) {
      // llvm::errs() << "PredLoad: " << Pred->getName() << "\n";
      if (Pred->empty()) {
        continue;
      }

      for (MachineInstr &I : make_range(Pred->rbegin(), Pred->rend())) {
        // llvm::errs() << "PredInstr: " << I << "\n";
        // I.print(llvm::errs());
        // llvm::errs() << "\n";
        // llvm::errs() << "IsCall: " << I.isCall() << "\n";
        // llvm::errs() << "IsIndirectBranch: " << I.isIndirectBranch() <<
        // "\n"; llvm::errs() << "IsConditionalBranch: " <<
        // I.isConditionalBranch()
        //              << "\n";
        if (!(I.isCall()) &&
            (I.isIndirectBranch() || I.isConditionalBranch())) {
          for (MachineBasicBlock *Succ : Pred->successors()) {
            if (Succ != AliasingLoad->getParent()) {
              IgnoredBranches.insert(Succ);
              // llvm::errs() << "IgnoredBranch: " << Succ->getName() << "\n";
            }
          }
          // llvm::errs() << "IgnoredBranch: " << Pred->getName() << "\n";
          IgnoredBranches.insert(Pred);
          break;
        }
      }
    }

    for (const auto &[Block, Distances] : PredDistanceMap) {
      uint64_t MaxDist = std::numeric_limits<uint64_t>::min();
      for (const auto &[Pred, PredDistance] : Distances) {
        // llvm::errs() << "Pred: " << Pred->getName() << " Distance: "
        //              << PredDistance << "\n";

        // TODO: check if both reacheable sets intersect (or find something
        // similar)
        const bool PrevDominatesStore =
            MPDT->dominates(StoreInstr->getParent(), Pred);
        const llvm::BasicBlock *BBfrom =
            AliasingLoad->getParent()->getBasicBlock();
        const llvm::BasicBlock *BBto = Pred->getBasicBlock();
        // llvm::errs() << "IsReachableFromLoad: " << IsReachableFromLoad <<
        // "\n"; llvm::errs() << "PrevDominatesStore: " << PrevDominatesStore
        // << "\n"; llvm::errs() << "IgnoredBranches: " <<
        // IgnoredBranches.count(Pred) << "\n";
        if (PrevDominatesStore && !IgnoredBranches.count(Pred) &&
            isPotentiallyReachable(BBfrom, BBto, nullptr, DT, LI)) {
          MaxDist = std::max(MaxDist, PredDistance);
          // llvm::errs() << "Dominates: " << Pred->getName() << " Distance: "
          //              << PredDistance << "\n";
          IgnoredBranches.insert(Pred);
        }
      }

      Distance = MaxDist == MarkerLoop ? MarkerLoop : Distance + MaxDist;
    }
  }

  return {Distance, AliasingLoad};
}

AliasInformation MarkAliasingLoadStore::computeAliasDistance(
    MachineInstrOrBlock InstrOrBlock, DenseSet<MachineBasicBlock *> &VisitedMBB,
    PredDistanceMap &PredDistanceMap, const MachineInstr *StoreInstr,
    const MachineDominatorTree *MDT, const MachineLoopInfo *MLI, AAResults *AA,
    const TargetInstrInfo *TII) {

  uint64_t AliasDistance = 0;
  MachineInstr *AliasingLoad = nullptr;

  const bool IsInstr = std::holds_alternative<MachineInstr *>(InstrOrBlock);

  MachineBasicBlock *MBB = nullptr;

  if (IsInstr) {
    MachineInstr *CurrentBlockInstr = std::get<MachineInstr *>(InstrOrBlock);
    MBB = CurrentBlockInstr->getParent();

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
          // StoreLoop->getParentLoop()
          if (MMO->isLoad()) {
            // If any load on the path is within a loop, we cannot
            // determine the trip count of the loop, so we conservatively
            // mark it with a special value.

            if (hasLoopWithinPath(StoreLoop, CurLoop)) {
              AliasDistance = MarkerLoop;
            } else {
              AliasDistance++;
            }
          }
        }
      }
    }
  } else {
    MBB = std::get<MachineBasicBlock *>(InstrOrBlock);
  }

  const MachineLoop *CurLoop = MLI->getLoopFor(MBB);
  PredDistanceVectors PredDistanceVec;

  if (MBB->pred_empty()) {
    return {AliasDistance, AliasingLoad};
  }

  for (MachineBasicBlock *Pred : MBB->predecessors()) {
    const bool IsBackEdge = CurLoop && MBB == CurLoop->getHeader();
    const bool HasNotVisited = !VisitedMBB.count(Pred);
    const bool IsReachable = MDT->dominates(Pred, MBB);

    if (!IsBackEdge || (HasNotVisited && IsReachable)) {
      VisitedMBB.insert(Pred);

      const bool HasAnyInstr = !Pred->empty();
      const MachineInstrOrBlock LastInstr =
          HasAnyInstr ? MachineInstrOrBlock{&Pred->back()}
                      : MachineInstrOrBlock{Pred};
      auto [PredDistance, PredAliasingLoad] =
          computeAliasDistance(LastInstr, VisitedMBB, PredDistanceMap,
                               StoreInstr, MDT, MLI, AA, TII);

      PredDistanceVec.push_back({Pred, PredDistance});

      if (PredAliasingLoad != nullptr) {
        AliasingLoad = PredAliasingLoad;
      }
    }
  }

  PredDistanceMap[MBB] = PredDistanceVec;

  return {AliasDistance, AliasingLoad};
}
