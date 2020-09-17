use kurobako_core::problem::ProblemSpec;
use kurobako_core::rng::ArcRng;
use kurobako_core::solver::Solver;
use kurobako_core::trial::{EvaluatedTrial, IdGen, NextTrial, Params, TrialId};
use ordered_float::OrderedFloat;
use rand::distributions::Distribution as _;
use randomforest::criterion::Mse;
use randomforest::table::{ColumnType, TableBuilder};
use randomforest::RandomForestRegressor;
use std::collections::HashMap;

#[derive(Debug)]
pub struct RfOpt {
    problem: ProblemSpec,
    trials: Vec<Trial>,
    evaluating: HashMap<TrialId, Params>,
    rng: ArcRng,
    // TODO: Phase
}

impl RfOpt {
    pub fn new(seed: u64, problem: ProblemSpec) -> Self {
        Self {
            problem,
            trials: Vec::new(),
            evaluating: HashMap::new(),
            rng: ArcRng::new(seed),
        }
    }

    fn fit_rf(&mut self) -> anyhow::Result<RandomForestRegressor> {
        // TODO: cap outliners
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

    fn calc_params_ranking(&self) -> anyhow::Result<Vec<usize>> {
        let mut features = vec![Vec::new(); self.problem.params_domain.variables().len()];
        let mut target = Vec::new();
        for t in &self.trials {
            for (f, &p) in features.iter_mut().zip(t.params.iter()) {
                f.push(p);
            }
            target.push(t.value);
        }

        let mut fanova =
            fanova::Fanova::fit(features.iter().map(|f| f.as_slice()).collect(), &target)?;
        let mut importances = (0..self.problem.params_domain.variables().len())
            .map(|i| (i, fanova.quantify_importance(&[i])))
            .collect::<Vec<_>>();
        importances.sort_by_key(|x| OrderedFloat(x.1.mean));
        importances.reverse();
        Ok(importances.into_iter().map(|x| x.0).collect::<Vec<_>>())
    }

    fn ask_random(&mut self) -> Params {
        let mut params = Vec::new();
        for p in self.problem.params_domain.variables() {
            let param = p.sample(&mut self.rng);
            params.push(param);
        }
        Params::new(params)
    }

    fn sample(&mut self, params: &[f64], param_index: usize) -> f64 {
        todo!()
    }
}

impl Solver for RfOpt {
    fn ask(&mut self, idg: &mut IdGen) -> kurobako_core::Result<NextTrial> {
        let id = idg.generate();

        let params = if self.trials.len() < 10 {
            self.ask_random()
        } else {
            let rf = self.fit_rf().expect("TODO");
            let ranking = self.calc_params_ranking().expect("TODO");
            let mut params = vec![std::f64::NAN; self.problem.params_domain.variables().len()];
            for i in ranking {
                params[i] = self.sample(&params, i);
            }
            Params::new(params)
        };

        self.evaluating.insert(id, params.clone());
        Ok(NextTrial {
            id,
            params,
            next_step: Some(self.problem.steps.last()),
        })
    }

    fn tell(&mut self, trial: EvaluatedTrial) -> kurobako_core::Result<()> {
        let params = self.evaluating.remove(&trial.id).expect("TODO");
        self.trials.push(Trial {
            params: params.into_vec(),
            value: trial.values[0],
        });
        Ok(())
    }
}

#[derive(Debug)]
pub struct Trial {
    pub params: Vec<f64>,
    pub value: f64,
}
