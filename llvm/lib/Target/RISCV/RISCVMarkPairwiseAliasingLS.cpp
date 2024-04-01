#include "RISCVInstrInfo.h"
#include "RISCVMachineFunctionInfo.h"

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
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/InitializePasses.h"

#include <limits>
#include <optional>
#include <tuple>

using namespace llvm;

using MachineInstrOrBlock = std::variant<MachineInstr *, MachineBasicBlock *>;
using AliasInformation = MachineInstr *;

constexpr unsigned int HINT_OPCODE = RISCV::SLLI;
constexpr Register HINT_REG = RISCV::X0;

#define RISCV_MARK_PAIRWISE_ALIASING_LS_NAME                                   \
  "RISC-V Mark Pairwise Aliasing LS pass"

namespace {
struct RISCVMarkPairwiseAliasingLS : public MachineFunctionPass {
  static char ID;

  RISCVMarkPairwiseAliasingLS() : MachineFunctionPass(ID) {}

  bool insertMarker(MachineBasicBlock *MBB, MachineInstr *InsertBefore,
                    MachineInstr *LoadInst, const TargetRegisterInfo *TRI);

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

  AliasInformation computeAliasDistance(MachineInstrOrBlock InstrOrBlock,
                                        const MachineInstr *StoreInstr,
                                        const MachineDominatorTree *MDT,
                                        const MachineLoopInfo *MLI,
                                        AAResults *AA,
                                        const TargetInstrInfo *TII);

  llvm::Register findBaseRegister(MachineInstr *MI) {
    for (const MachineOperand &MO : MI->operands()) {
      if (MO.isReg() && MO.isUse()) {
        return MO.getReg();
      }
    }
    return 0;
  }

  bool runOnMachineFunction(MachineFunction &Fn) override;

  StringRef getPassName() const override {
    return RISCV_MARK_PAIRWISE_ALIASING_LS_NAME;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineDominatorTree>();
    AU.addRequired<MachinePostDominatorTree>();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<MachineLoopInfo>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

char RISCVMarkPairwiseAliasingLS::ID = 0;

} // end of anonymous namespace

INITIALIZE_PASS(RISCVMarkPairwiseAliasingLS, "riscv-mark-pairwise-aliasing-ls",
                RISCV_MARK_PAIRWISE_ALIASING_LS_NAME, false, false)

bool RISCVMarkPairwiseAliasingLS::insertMarker(MachineBasicBlock *MBB,
                                               MachineInstr *InsertBefore,
                                               MachineInstr *LoadInst,
                                               const TargetRegisterInfo *TRI) {
  const TargetInstrInfo *TII = MBB->getParent()->getSubtarget().getInstrInfo();
  const DebugLoc &DL = InsertBefore->getDebugLoc();

  llvm::MachineInstrBuilder builder =
      BuildMI(*MBB, InsertBefore, DL, TII->get(HINT_OPCODE));
  builder.addDef(HINT_REG);
  builder.addUse(findBaseRegister(LoadInst));
  builder.addImm(0);

  return true;
}

llvm::SmallVector<MachineInstr *>
RISCVMarkPairwiseAliasingLS::filterInstructions(
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

bool RISCVMarkPairwiseAliasingLS::runOnMachineFunction(MachineFunction &MF) {
  bool Changed = false;

  MachineDominatorTree *MDT = &getAnalysis<MachineDominatorTree>();
  AAResults *AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();
  MachineLoopInfo *MLI = &getAnalysis<MachineLoopInfo>();
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();

  SmallPtrSet<const MachineInstr *, 4> MarkedInstr;

  for (MachineInstr *Store : findStores(MF)) {
    const AliasInformation AliasingLoad = computeAliasDistance(
        MachineInstrOrBlock{Store}, Store, MDT, MLI, AA, TII);

    if (AliasingLoad != nullptr) {
      if (!MarkedInstr.count(AliasingLoad)) {
        insertMarker(AliasingLoad->getParent(), AliasingLoad, AliasingLoad, TRI);
        MarkedInstr.insert(AliasingLoad);
      }

      llvm::errs() << "Store: " << *Store << "\n";
      DebugLoc StoreDL = Store->getDebugLoc();
      StoreDL.print(llvm::errs());
      llvm::errs() << "\n";
      DebugLoc LoadDL = AliasingLoad->getDebugLoc();
      llvm::errs() << "AliasingLoad: " << *AliasingLoad << "\n";
      LoadDL.print(llvm::errs());
      llvm::errs() << "\n";
      if (Store->getParent() != AliasingLoad->getParent()) {
        llvm::errs() << "Different BB\n";
      }
      llvm::errs() << "==============================\n";
    }
  }

  return Changed;
}

AliasInformation RISCVMarkPairwiseAliasingLS::computeAliasDistance(
    MachineInstrOrBlock InstrOrBlock, const MachineInstr *StoreInstr,
    const MachineDominatorTree *MDT, const MachineLoopInfo *MLI, AAResults *AA,
    const TargetInstrInfo *TII) {

  MachineInstr *AliasingLoad = nullptr;

  SmallVector<MachineInstrOrBlock, 8> WorkList;
  WorkList.push_back(InstrOrBlock);
  SmallPtrSet<MachineBasicBlock *, 4> VisitedMBB;

  do {
    MachineInstrOrBlock CurrentInstrOrBlock = WorkList.pop_back_val();

    const bool IsInstr =
        std::holds_alternative<MachineInstr *>(CurrentInstrOrBlock);
    MachineBasicBlock *MBB = nullptr;

    if (IsInstr) {
      MachineInstr *CurrentBlockInstr =
          std::get<MachineInstr *>(CurrentInstrOrBlock);
      MBB = CurrentBlockInstr->getParent();

      auto IsInstructionMatch = [&](MachineInstr const *I) {
        bool HasMemOp = !I->memoperands_empty();
        return HasMemOp && I->mayLoad() && I->mustAlias(AA, *StoreInstr, false);
      };

      for (auto &I : make_range(CurrentBlockInstr->getReverseIterator(),
                                CurrentBlockInstr->getParent()->instr_rend())) {
        if (IsInstructionMatch(&I)) {
          AliasingLoad = &I;
          return AliasingLoad;
        }
      }
    } else {
      MBB = std::get<MachineBasicBlock *>(CurrentInstrOrBlock);
    }

    const MachineLoop *CurLoop = MLI->getLoopFor(MBB);
    for (MachineBasicBlock *Pred : MBB->predecessors()) {
      const bool IsBackEdge = CurLoop && MBB == CurLoop->getHeader();
      const bool HasNotVisited = !VisitedMBB.count(Pred);
      const bool IsReachable = MDT->dominates(Pred, MBB);

      if (HasNotVisited && (!IsBackEdge || (IsReachable))) {
        VisitedMBB.insert(Pred);

        const bool HasAnyInstr = !Pred->empty();
        const MachineInstrOrBlock LastInstr =
            HasAnyInstr ? MachineInstrOrBlock{&Pred->back()}
                        : MachineInstrOrBlock{Pred};

        WorkList.push_back(LastInstr);
      }
    }
  } while (!WorkList.empty());

  return AliasingLoad;
}

FunctionPass *llvm::createRISCVMarkPairwiseAliasingLSPass() {
  return new RISCVMarkPairwiseAliasingLS();
}
