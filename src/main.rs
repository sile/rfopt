use anyhow::anyhow;
use kurobako_core::epi::channel::{MessageReceiver, MessageSender};
use kurobako_core::epi::solver::SolverMessage;
use kurobako_core::solver::{Capability, Solver as _, SolverSpecBuilder};
use kurobako_core::trial::IdGen;
use std::collections::HashMap;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
struct Opt {
    #[structopt(flatten)]
    options: rfopt::Options,
}

fn main() -> anyhow::Result<()> {
    let opt = Opt::from_args();
    let stdout = std::io::stdout();
    let stdin = std::io::stdin();
    let mut tx = MessageSender::new(stdout.lock());
    let mut rx = MessageReceiver::<SolverMessage, _>::new(stdin.lock());

    let spec = SolverSpecBuilder::new("rfopt")
        .capable(Capability::Categorical)
        .capable(Capability::Concurrent)
        .capable(Capability::LogUniformContinuous)
        .capable(Capability::UniformContinuous)
        .capable(Capability::UniformDiscrete)
        .finish();
    tx.send(&SolverMessage::SolverSpecCast { spec })?;

    let mut solvers = HashMap::new();
    loop {
        match rx.recv()? {
            SolverMessage::CreateSolverCast {
                solver_id,
                random_seed,
                problem,
            } => {
                let opt = rfopt::Rfopt::new(random_seed, problem, opt.options.clone());
                solvers.insert(solver_id, opt);
            }
            SolverMessage::AskCall {
                solver_id,
                next_trial_id,
            } => {
                let solver = solvers
                    .get_mut(&solver_id)
                    .ok_or_else(|| anyhow!("unknown solver {:?}", solver_id))?;
                let mut idg = IdGen::from_next_id(next_trial_id);
                let trial = solver.ask(&mut idg)?;
                tx.send(&SolverMessage::AskReply {
                    next_trial_id: idg.peek_id().get(),
                    trial,
                })?;
            }
            SolverMessage::TellCall { solver_id, trial } => {
                let solver = solvers
                    .get_mut(&solver_id)
                    .ok_or_else(|| anyhow!("unknown solver {:?}", solver_id))?;
                solver.tell(trial)?;
                tx.send(&SolverMessage::TellReply {})?;
            }
            SolverMessage::DropSolverCast { solver_id } => {
                solvers.remove(&solver_id);
            }
            other => panic!("unexpected message: {:?}", other),
        }
    }
}
