use kurobako_core::epi::channel::{MessageReceiver, MessageSender};
use kurobako_core::epi::solver::SolverMessage;
use kurobako_core::solver::{Capability, SolverSpecBuilder};
use std::collections::HashMap;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
struct Opt {}

fn main() -> anyhow::Result<()> {
    let _opt = Opt::from_args();
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
                let opt = rfopt::RfOpt::new(random_seed, problem);
                solvers.insert(solver_id, opt);
            }
            SolverMessage::AskCall {
                solver_id,
                next_trial_id,
            } => {
                todo!();
            }
            SolverMessage::TellCall { solver_id, trial } => {
                todo!();
            }
            SolverMessage::DropSolverCast { solver_id } => {
                solvers.remove(&solver_id);
            }
            other => panic!("unexpected message: {:?}", other),
        }
    }
}
