#include "RISCVInstrInfo.h"
#include "RISCVMachineFunctionInfo.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
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
#include <set>
#include <tuple>

using namespace llvm;

enum class AliasType {
  NoAlias,
  MustAlias,
  CacheLineAlias,
  MayAlias,
};

using BranchingSuccs =
    std::set<std::pair<const MachineBasicBlock *, const MachineBasicBlock *>>;

struct AliasInfo {
  AliasInfo()
      : Type(AliasType::NoAlias), BranchProb(BranchProbability::getUnknown()),
        ControlDeps(BranchingSuccs()) {}
  AliasInfo(AliasType Type, BranchProbability BranchProb,
            BranchingSuccs BranchingSuccs)
      : Type(Type), BranchProb(BranchProb), ControlDeps(BranchingSuccs) {}
  AliasType Type;
  BranchProbability BranchProb;
  BranchingSuccs ControlDeps;
};

using MachineInstrOrBlock = std::variant<MachineInstr *, MachineBasicBlock *>;

using MachineInstrOrBlockWithSucc =
    std::tuple<MachineInstrOrBlock, BranchingSuccs, const MachineBasicBlock *>;
using AliasInformation = std::tuple<MachineInstr *, AliasInfo>;
using AliasInformations = SmallVector<AliasInformation, 4>;
using AliasInformationsMap =
    SmallMapVector<MachineInstr *, AliasInformations, 4>;
using AliasFuncType =
    std::function<AliasType(MachineInstr const &StoreI, const MachineInstr *)>;
using StoreInfo = std::tuple<AliasInfo, MachineInstr *>;
using LoadToStoresMap =
    SmallMapVector<llvm::MachineInstr *, SmallVector<StoreInfo, 4>, 4>;

static cl::opt<unsigned> LoadExclusiveHintThreshold(
    "riscv-load-exclusive-hint-threshold", cl::Hidden,
    cl::desc("Threshold for adding hints to load-exclusive instruction "
             "(default=51)"),
    cl::init(51));

static cl::opt<unsigned> LoadExclusiveCacheLineWindow(
    "riscv-load-exclusive-hint-cache-line-window", cl::Hidden,
    cl::desc("Cache line window for load-exclusive instruction "
             "(default=64)"),
    cl::init(64));

static cl::opt<bool> EnableBlockLevelAlias(
    "riscv-load-exclusive-hint-block-level-alias", cl::Hidden,
    cl::desc("Enable block level aliasing (default=false)"), cl::init(false));

static cl::opt<bool> EnableGreedy(
    "riscv-load-exclusive-hint-greedy", cl::Hidden,
    cl::desc("Enable greedy hinting, don't stop after first aliasing L/S pair "
             "for a given store (default=false)"),
    cl::init(false));

static cl::opt<bool> EnableHintFurthest(
    "riscv-load-exclusive-hint-furthest", cl::Hidden,
    cl::desc("Enable furthest hinting, annotate further aliasing L/S pairs "
             "for a given store (default=false)"),
    cl::init(false));

static cl::opt<bool> EnableBranchProb(
    "riscv-load-exclusive-hint-branch-prob", cl::Hidden,
    cl::desc("Enable branch probability computation (default=false)"),
    cl::init(false));

static cl::opt<bool> EnableVerboseLogging(
    "riscv-load-exclusive-hint-verbose", cl::Hidden,
    cl::desc("Enable verbose logging"),
    cl::init(false));

#define DEBUG_TYPE "RISCVMarkPairwiseAliasingLS"
STATISTIC(NumPairsMustAlias,
          "Number of hints instructions detected using Must Alias");
STATISTIC(NumPairsCacheLineAlias,
          "Number of hints instructions detected using Cache Line Alias");
STATISTIC(NumPairsSimpleAlias,
          "Number of hints instructions detected using Simple Alias");
STATISTIC(NumPairsAnnotated, "Number of hints instructions annotated");
STATISTIC(NumPrunedProbabilitiesLoads,
          "Number of loads pruned due to unknown branch probabilities");
STATISTIC(NumPrunedAlreadyAnnotatedLoads,
          "Number of loads pruned due to already being annotated");
STATISTIC(NumPotentialLoads, "Number of potential loads");
STATISTIC(NumPairsEquivalentStores,
          "Number of equivalent stores pattern detected");
constexpr unsigned int HINT_OPCODE = RISCV::SLLI;
constexpr Register HINT_REG = RISCV::X0;

// Atomic store instructions
// Beware that this does not include pseudo-atomics instructions (e.g:
// PseudoCmpXchg64) that have not been expanded yet Thus the pass must be ran
// after the expansion of pseudo-atomics or this needs to be updated
const std::set<unsigned int> ATOMICS_OPCODES = {
    RISCV::LR_W,      RISCV::LR_W_AQ,      RISCV::LR_W_AQ_RL,
    RISCV::LR_D,      RISCV::LR_D_AQ,      RISCV::LR_D_AQ_RL,
    RISCV::SC_W,      RISCV::SC_W_AQ,      RISCV::SC_W_AQ_RL,
    RISCV::SC_D,      RISCV::SC_D_AQ,      RISCV::SC_D_AQ_RL,
    RISCV::AMOSWAP_W, RISCV::AMOSWAP_W_AQ, RISCV::AMOSWAP_W_AQ_RL,
    RISCV::AMOSWAP_D, RISCV::AMOSWAP_D_AQ, RISCV::AMOSWAP_D_AQ_RL,
    RISCV::AMOADD_W,  RISCV::AMOADD_W_AQ,  RISCV::AMOADD_W_AQ_RL,
    RISCV::AMOADD_D,  RISCV::AMOADD_D_AQ,  RISCV::AMOADD_D_AQ_RL,
    RISCV::AMOXOR_W,  RISCV::AMOXOR_W_AQ,  RISCV::AMOXOR_W_AQ_RL,
    RISCV::AMOXOR_D,  RISCV::AMOXOR_D_AQ,  RISCV::AMOXOR_D_AQ_RL,
    RISCV::AMOAND_W,  RISCV::AMOAND_W_AQ,  RISCV::AMOAND_W_AQ_RL,
    RISCV::AMOAND_D,  RISCV::AMOAND_D_AQ,  RISCV::AMOAND_D_AQ_RL,
    RISCV::AMOOR_W,   RISCV::AMOOR_W_AQ,   RISCV::AMOOR_W_AQ_RL,
    RISCV::AMOOR_D,   RISCV::AMOOR_D_AQ,   RISCV::AMOOR_D_AQ_RL,
    RISCV::AMOMIN_W,  RISCV::AMOMIN_W_AQ,  RISCV::AMOMIN_W_AQ_RL,
    RISCV::AMOMIN_D,  RISCV::AMOMIN_D_AQ,  RISCV::AMOMIN_D_AQ_RL,
    RISCV::AMOMAX_W,  RISCV::AMOMAX_W_AQ,  RISCV::AMOMAX_W_AQ_RL,
    RISCV::AMOMAX_D,  RISCV::AMOMAX_D_AQ,  RISCV::AMOMAX_D_AQ_RL,
    RISCV::AMOMINU_W, RISCV::AMOMINU_W_AQ, RISCV::AMOMINU_W_AQ_RL,
    RISCV::AMOMINU_D, RISCV::AMOMINU_D_AQ, RISCV::AMOMINU_D_AQ_RL,
    RISCV::AMOMAXU_W, RISCV::AMOMAXU_W_AQ, RISCV::AMOMAXU_W_AQ_RL,
    RISCV::AMOMAXU_D, RISCV::AMOMAXU_D_AQ, RISCV::AMOMAXU_D_AQ_RL,
};

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

  AliasInformations computeAliasDistance(
      MachineInstrOrBlock InstrOrBlock, const MachineInstr *StoreInstr,
      const MachineDominatorTree *MDT, const MachinePostDominatorTree *MPDT,
      const MachineLoopInfo *MLI, AAResults *AA,
      const MachineBranchProbabilityInfo *MBPI, const TargetInstrInfo *TII,
      AliasFuncType IsInstructionMatch);

  llvm::Register findBaseRegister(MachineInstr *MI) {
    for (const MachineOperand &MO : MI->operands()) {
      if (MO.isReg() && MO.isUse()) {
        return MO.getReg();
      }
    }
    return 0;
  }

  bool isAtomic(MachineInstr *MI) {
    return ATOMICS_OPCODES.count(MI->getOpcode());
  }

  bool runOnMachineFunction(MachineFunction &Fn) override;

  size_t insertHints(AliasInformations &AliasInfos, MachineInstr *Store,
                     SmallPtrSet<const MachineInstr *, 4> &MarkedInstr,
                     LoadToStoresMap &LoadToStoreMap,
                     const TargetRegisterInfo *TRI,
                     const MachineBranchProbabilityInfo *MBPI,
                     const MachineLoopInfo *MLI,
                     const MachineBlockFrequencyInfo *MBFI) {
    const BranchProbability hotThreshold(LoadExclusiveHintThreshold, 100);

    SmallPtrSet<const MachineInstr *, 4> MarkedLinkedStores;

    for (auto &[AliasingLoad, AliasInfo] : AliasInfos) {
      // don't annotate atomic stores for RISC-V
      if (isAtomic(Store)) {
        continue;
      }

      // hint furthest load
      if (EnableHintFurthest && MarkedLinkedStores.count(Store)) {
        continue;
      }

      // skip cache line aliasing if block level aliasing is disabled
      if (!EnableBlockLevelAlias &&
          AliasInfo.Type == AliasType::CacheLineAlias) {
        continue;
      }

      if (AliasingLoad != nullptr) {
        bool IsEdgeHot = (!AliasInfo.BranchProb.isUnknown() &&
                          AliasInfo.BranchProb > hotThreshold) ||
                         AliasInfo.BranchProb.isUnknown();

        // Already annotated
        if (!MarkedInstr.count(AliasingLoad)) {
          if(EnableVerboseLogging) {
            llvm::errs() << "==============================\n";
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
            if (AliasInfo.Type == AliasType::MustAlias) {
              llvm::errs() << "Must Alias\n";
            } else if (AliasInfo.Type == AliasType::CacheLineAlias) {
              llvm::errs() << "Cache Line Alias\n";
            } else if (AliasInfo.Type == AliasType::MayAlias) {
              llvm::errs() << "May Alias\n";
            }

            llvm::errs() << "Control Dependencies: (cardinality: "
                        << AliasInfo.ControlDeps.size() << ")\n";
            for (auto [U, V] : AliasInfo.ControlDeps) {
              const auto prob = MBPI->getEdgeProbability(U, V);
              auto PredInstr = U->getLastNonDebugInstr();
              auto SuccInstr = V->getLastNonDebugInstr();
              llvm::errs() << "Control Dependent: ";
              llvm::errs() << "Pred: " << *PredInstr << " -> Succ: " << *SuccInstr
                          << " with probability: " << prob << "\n";
              // llvm::errs() << " with probability: " << prob << "\n";
            }
          }

          auto &Stores = LoadToStoreMap[AliasingLoad];
          auto TotalProb = AliasInfo.BranchProb;
          // print other store probabilities
          for (auto &[AI, CurStore] : Stores) {
            if (CurStore == Store) {
              continue;
            }

            if(EnableVerboseLogging) {
              llvm::errs() << "Other Store: " << *CurStore
                          << " -> Probability: " << AI.BranchProb << "\n";

              // print control deps
              llvm::errs() << "Control Dependencies Linked: (cardinality: "
                          << AI.ControlDeps.size() << ")\n";
            }

            for (auto [U, V] : AI.ControlDeps) {
              if(EnableVerboseLogging) {
                DebugLoc UDL = U->getFirstNonDebugInstr()->getDebugLoc();
                DebugLoc VDL = V->getFirstNonDebugInstr()->getDebugLoc();
                llvm::errs() << "Control Dependent Linked: ";
                UDL.print(llvm::errs());
                llvm::errs() << " -> ";
                VDL.print(llvm::errs());
                llvm::errs() << "\n";
              }
            }

            TotalProb += AI.BranchProb;
          }

          const MachineLoop *StoreLoop = MLI->getLoopFor(Store->getParent());
          const MachineLoop *LoadLoop =
              MLI->getLoopFor(AliasingLoad->getParent());

          // bool SameLoopSCC = StoreLoop && LoadLoop &&
          //                   StoreLoop->getParentLoop() ==
          //                   LoadLoop->getParentLoop();

          // case
          if (LoadLoop && !StoreLoop) {
            continue;
          }

          if (StoreLoop && EnableVerboseLogging) {
            llvm::errs() << "Store Loop: " << StoreLoop << "\n";
          }

          if (LoadLoop && EnableVerboseLogging) {
            llvm::errs() << "Load Loop: " << LoadLoop << "\n";
          }

          // if (StoreLoop && LoadStore && SameLoopSCC &&
          // StoreLoop->getLoopDepth() > LoadLoop->getLoopDepth())
          //   continue;
          // }

          if(EnableVerboseLogging)
            llvm::errs() << "Branch Probability: " << TotalProb << "\n";

          if (TotalProb > hotThreshold && !IsEdgeHot) {
            IsEdgeHot = true;
            NumPairsEquivalentStores++;
          }

          if (IsEdgeHot || !EnableBranchProb) {
            insertMarker(AliasingLoad->getParent(), AliasingLoad, AliasingLoad,
                         TRI);
            MarkedInstr.insert(AliasingLoad);
            MarkedLinkedStores.insert(Store);
          } else if (!IsEdgeHot) {
            NumPrunedProbabilitiesLoads++;
          }

          if(EnableVerboseLogging)
            llvm::errs() << "Is edge hot: " << IsEdgeHot << "\n";

          // for (auto *MO : Store->memoperands()) {
          //   // Value *Base = MO->getValue();
          //   if (auto Base = MO->getValue()) {
          //     // print valueid
          //     // check if ConstantExprVal
          //     if (auto *CE = dyn_cast<ConstantExpr>(Base)) {
          //       const Value *CEI = CE->getAsInstruction();
          //       // get as GEP
          //       if (const GetElementPtrInst *GEP =
          //               dyn_cast<GetElementPtrInst>(CEI)) {
          //         const auto Source = GEP->getSourceElementType();
          //         // get alignment of SourceType
          //         const DataLayout &DL = Store->getMF()->getDataLayout();
          //         const auto Alignment = GEP->getPointerAlignment(DL);
          //         llvm::errs() << "Base: " << *Base << " Source: " << *Source
          //                      << " Alignment: " << Alignment.value() <<
          //                      "\n";
          //       }
          //     } else {
          //       llvm::errs() << "Base: " << *Base << "\n";
          //     }

          //   }
          // }

          if(EnableVerboseLogging) {
            llvm::errs() << "Block Frequency: "
                        << MBFI->getBlockFreq(Store->getParent()).getFrequency()
                        << "\n";

            llvm::errs() << "==============================\n";
          }
        } else {
          NumPrunedAlreadyAnnotatedLoads++;
        }
      }
    }

    return MarkedInstr.size();
  }

  StringRef getPassName() const override {
    return RISCV_MARK_PAIRWISE_ALIASING_LS_NAME;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineDominatorTree>();
    AU.addRequired<MachineBranchProbabilityInfo>();
    AU.addRequired<MachineBlockFrequencyInfo>();
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
  for (auto &MBB : make_range(MF.begin(), MF.end())) {
    for (auto &MI : make_range(MBB.begin(), MBB.end())) {
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
  MachinePostDominatorTree *MPDT = &getAnalysis<MachinePostDominatorTree>();
  AAResults *AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();
  MachineLoopInfo *MLI = &getAnalysis<MachineLoopInfo>();
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  const MachineBranchProbabilityInfo *MBPI =
      &getAnalysis<MachineBranchProbabilityInfo>();
  const MachineBlockFrequencyInfo *MBFI =
      &getAnalysis<MachineBlockFrequencyInfo>();
  SmallPtrSet<const MachineInstr *, 4> MarkedInstr;
  size_t MarkedLoads = 0;
  size_t TrivialMatchingLoads = 0;
  size_t DetectedAliasingLoads = 0;
  size_t DetectedCacheAliasingLoads = 0;

  NumPotentialLoads += findLoads(MF).size();

  LoadToStoresMap LoadToStoreMap;
  AliasInformationsMap AliasInfosMap;

  auto IsInstructionMatch = [&](MachineInstr const &StoreI,
                                MachineInstr const *I) {
    bool HasMemOp = !I->memoperands_empty();
    if (HasMemOp && I->mayLoad()) {
      if (I->mustAlias(AA, StoreI, false))
        return AliasType::MustAlias;
      if (I->partialAlias(AA, StoreI, false, LoadExclusiveCacheLineWindow))
        return AliasType::CacheLineAlias;
    }

    return AliasType::NoAlias;
  };

  // auto IsInstructionMatchTrivial = [&](MachineInstr const &StoreI,
  //                                      MachineInstr const *I) {
  //   bool HasMemOp = !I->memoperands_empty();
  //   if (HasMemOp && I->mayLoad()) {
  //     if (I->useSameMemoryRef(StoreI))
  //       return AliasType::MustAlias;
  //   }

  //   return AliasType::NoAlias;
  // };

  for (MachineInstr *Store : findStores(MF)) {
    const auto AliasInfos =
        computeAliasDistance(MachineInstrOrBlock{Store}, Store, MDT, MPDT, MLI,
                             AA, MBPI, TII, IsInstructionMatch);

    // const auto AliasInfosTrivial =
    //     computeAliasDistance(MachineInstrOrBlock{Store}, Store, MDT, MPDT,
    //     MLI,
    //                          AA, MBPI, TII, IsInstructionMatchTrivial);

    TrivialMatchingLoads = 0;
    DetectedAliasingLoads = std::accumulate(
        AliasInfos.begin(), AliasInfos.end(), 0,
        [](size_t acc, const AliasInformation &AI) {
          return std::get<1>(AI).Type == AliasType::MustAlias ? acc + 1 : acc;
        });
    DetectedCacheAliasingLoads = std::accumulate(
        AliasInfos.begin(), AliasInfos.end(), 0,
        [](size_t acc, const AliasInformation &AI) {
          return std::get<1>(AI).Type == AliasType::CacheLineAlias ? acc + 1
                                                                   : acc;
        });

    NumPairsSimpleAlias += TrivialMatchingLoads;
    NumPairsMustAlias += DetectedAliasingLoads;
    NumPairsCacheLineAlias += DetectedCacheAliasingLoads;

    for (auto &[Load, AliasInfo] : AliasInfos) {
      LoadToStoreMap[Load].push_back({AliasInfo, Store});
    }

    // MarkedLoads = insertHints(AliasInfos, Store, MarkedInstr, TRI, MBPI,
    // MBFI);
    AliasInfosMap[Store] = AliasInfos;
  }

  for (auto &[Store, AliasInfos] : AliasInfosMap) {
    MarkedLoads = insertHints(AliasInfos, Store, MarkedInstr, LoadToStoreMap,
                              TRI, MBPI, MLI, MBFI);
  }

  NumPairsAnnotated += MarkedLoads;

  if(EnableVerboseLogging) {
    llvm::errs() << "Marked loads: " << MarkedLoads
                << " as hints on function: " << MF.getName() << " of which "
                << TrivialMatchingLoads << " are trivially matching and "
                << DetectedAliasingLoads << " are detected aliasing\n";
  }

  return Changed;
}

AliasInformations RISCVMarkPairwiseAliasingLS::computeAliasDistance(
    MachineInstrOrBlock InstrOrBlock, const MachineInstr *StoreInstr,
    const MachineDominatorTree *MDT, const MachinePostDominatorTree *MPDT,
    const MachineLoopInfo *MLI, AAResults *AA,
    const MachineBranchProbabilityInfo *MBPI, const TargetInstrInfo *TII,
    AliasFuncType IsInstructionMatch) {

  MachineInstr *AliasingLoad = nullptr;

  SmallVector<MachineInstrOrBlockWithSucc, 8> WorkList;
  WorkList.push_back(
      MachineInstrOrBlockWithSucc{InstrOrBlock, BranchingSuccs(), nullptr});
  SmallPtrSet<MachineBasicBlock *, 4> VisitedMBB;

  AliasInformations AliasInfos;

  do {
    auto [CurrentInstrOrBlock, CurrentSucc, LastDependentBlock] =
        WorkList.pop_back_val();

    const bool IsInstr =
        std::holds_alternative<MachineInstr *>(CurrentInstrOrBlock);
    MachineBasicBlock *MBB = nullptr;
    bool HasEncounteredCallOrAtomic = false;

    if (IsInstr) {
      MachineInstr *CurrentBlockInstr =
          std::get<MachineInstr *>(CurrentInstrOrBlock);
      MBB = CurrentBlockInstr->getParent();

      const bool MBBDominatesStore =
          MDT->dominates(MBB, StoreInstr->getParent());

      for (auto &I : make_range(CurrentBlockInstr->getReverseIterator(),
                                CurrentBlockInstr->getParent()->instr_rend())) {

        // If this block dominates the store, we can stop looking for loads
        // whenever we encounter a call or an atomic instruction.
        if (MBBDominatesStore) {
          if (I.isCall() || isAtomic(&I)) {
            HasEncounteredCallOrAtomic = true;
            break;
          }
        }

        auto IsBlocker = [&](MachineInstr const &StoreI,
                             MachineInstr const *I) {
          if (I == StoreInstr) {
            return false;
          }

          bool HasMemOp = !I->memoperands_empty();
          if (HasMemOp && I->mayStore()) {
            if (I->mustAlias(AA, StoreI, false))
              return true;
            if (I->partialAlias(AA, StoreI, false,
                                LoadExclusiveCacheLineWindow))
              return true;
          }

          return false;
        };

        if (IsBlocker(*StoreInstr, &I)) {
          break;
        }

        const AliasType AT = IsInstructionMatch(*StoreInstr, &I);
        if (AT != AliasType::NoAlias) {
          AliasingLoad = &I;
          auto Prob = BranchProbability::getOne();
          for (auto [U, V] : CurrentSucc) {
            const auto BranchProb = MBPI->getEdgeProbability(U, V);

            if (!Prob.isUnknown()) {
              Prob *= BranchProb;
            }
          }

          if (EnableHintFurthest) {
            // insert at @front
            AliasInfos.insert(AliasInfos.begin(),
                              {AliasingLoad, AliasInfo(AT, Prob, CurrentSucc)});
          } else {
            AliasInfos.push_back(
                {AliasingLoad, AliasInfo(AT, Prob, CurrentSucc)});
          }

          // If we are not greedy, we can stop after the first aliasing L/S pair
          if (!EnableGreedy && !EnableHintFurthest) {
            return AliasInfos;
          }
        }
      }
    } else {
      MBB = std::get<MachineBasicBlock *>(CurrentInstrOrBlock);
    }

    if (!HasEncounteredCallOrAtomic) {
      const auto IsControlDependent =
          [&](MachineBasicBlock *Pred, MachineBasicBlock *Block,
              const MachineBasicBlock *LastDependentBB) {
            return MPDT->dominates(LastDependentBB, Block) &&
                   !MPDT->properlyDominates(LastDependentBB, Pred);
          };

      const MachineLoop *CurLoop = MLI->getLoopFor(MBB);
      for (MachineBasicBlock *Pred : MBB->predecessors()) {
        const bool IsBackEdge = CurLoop && MBB == CurLoop->getHeader();
        const bool HasNotVisited = !VisitedMBB.count(Pred);
        const bool IsReachable = MDT->dominates(Pred, MBB);

        // Check if the store is control dependent on the edge from Pred to MBB
        const MachineBasicBlock *DependentBlock = LastDependentBlock;
        if (LastDependentBlock &&
            IsControlDependent(Pred, MBB, LastDependentBlock)) {
          CurrentSucc.insert(std::make_pair(Pred, MBB));
          DependentBlock = Pred;
        } else if (MBB == StoreInstr->getParent()) {
          if (IsControlDependent(Pred, MBB, MBB)) {
            CurrentSucc.insert(std::make_pair(Pred, MBB));
            DependentBlock = Pred;
          }
        }

        if (HasNotVisited && (!IsBackEdge || (IsReachable))) {
          VisitedMBB.insert(Pred);

          const bool HasAnyInstr = !Pred->empty();
          const MachineInstrOrBlockWithSucc LastInstr =
              HasAnyInstr
                  ? MachineInstrOrBlockWithSucc{MachineInstrOrBlock{
                                                    &Pred->back()},
                                                CurrentSucc, DependentBlock}
                  : MachineInstrOrBlockWithSucc{MachineInstrOrBlock{Pred},
                                                CurrentSucc, DependentBlock};

          WorkList.push_back(LastInstr);
        }
      }
    }
  } while (!WorkList.empty());

  return AliasInfos;
}

FunctionPass *llvm::createRISCVMarkPairwiseAliasingLSPass() {
  return new RISCVMarkPairwiseAliasingLS();
}
