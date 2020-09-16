use kurobako_core::problem::ProblemSpec;
use kurobako_core::solver::Solver;
use kurobako_core::trial::{EvaluatedTrial, IdGen, NextTrial};
use ordered_float::OrderedFloat;
use randomforest::criterion::Mse;
use randomforest::table::{ColumnType, TableBuilder};
use randomforest::RandomForestRegressor;

#[derive(Debug)]
pub struct RfOpt {
    problem: ProblemSpec,
    trials: Vec<Trial>,
}

impl RfOpt {
    pub fn new(problem: ProblemSpec) -> Self {
        Self {
            problem,
            trials: Vec::new(),
        }
    }

    fn fit_rf(&mut self) -> anyhow::Result<RandomForestRegressor> {
        let mut table = TableBuilder::new();
        let columns = self
            .problem
            .params_domain
            .variables()
            .iter()
            .map(|v| {
                if matches!(v.range(), kurobako_core::domain::Range::Categorical{..}) {
                    ColumnType::Categorical
                } else {
                    ColumnType::Numerical
                }
            })
            .collect::<Vec<_>>();
        table.set_feature_column_types(&columns)?;

        self.trials.sort_by_key(|t| OrderedFloat(t.value));
        let n = (self.trials.len() as f64 * 0.8) as usize;
        for t in &self.trials[..n] {
            table.add_row(&t.params, t.value)?;
        }
        for t in &self.trials[n..] {
            table.add_row(&t.params, self.trials[n].value)?;
        }

        Ok(RandomForestRegressor::fit(Mse, table.build()?))
    }
}

impl Solver for RfOpt {
    fn ask(&mut self, idg: &mut IdGen) -> kurobako_core::Result<NextTrial> {
        let rf = self.fit_rf();
        todo!()
    }

    fn tell(&mut self, trial: EvaluatedTrial) -> kurobako_core::Result<()> {
        todo!()
    }
}

#[derive(Debug)]
pub struct Trial {
    pub params: Vec<f64>,
    pub value: f64,
}
