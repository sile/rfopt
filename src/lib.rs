use kurobako_core::problem::ProblemSpec;
use kurobako_core::rng::ArcRng;
use kurobako_core::solver::Solver;
use kurobako_core::trial::{EvaluatedTrial, IdGen, NextTrial, Params, TrialId};
use ordered_float::OrderedFloat;
use rand::distributions::Distribution as _;
use randomforest::criterion::Mse;
use randomforest::table::{ColumnType, TableBuilder};
use randomforest::{RandomForestRegressor, RandomForestRegressorOptions};
use std::collections::HashMap;
use std::num::NonZeroUsize;

#[derive(Debug)]
pub struct RfOpt {
    problem: ProblemSpec,
    trials: Vec<Trial>,
    evaluating: HashMap<TrialId, Params>,
    rng: ArcRng,
    best_value: f64,
    best_params: Params,
    level: i32, // TODO: rename
}

impl RfOpt {
    pub fn new(seed: u64, problem: ProblemSpec) -> Self {
        Self {
            problem,
            trials: Vec::new(),
            evaluating: HashMap::new(),
            rng: ArcRng::new(seed),
            best_value: std::f64::INFINITY,
            best_params: Params::new(Vec::new()),
            level: 0,
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
        let n = (self.trials.len() as f64 * 0.9) as usize;
        for t in &self.trials[..n] {
            table.add_row(&t.params, t.value)?;
        }
        for t in &self.trials[n..] {
            table.add_row(&t.params, self.trials[n].value)?;
        }

        Ok(RandomForestRegressorOptions::new()
            .trees(NonZeroUsize::new(10).unwrap())
            .fit(Mse, table.build()?))
    }

    fn ask_random(&mut self) -> Params {
        let mut params = Vec::new();
        for p in self.problem.params_domain.variables() {
            let param = p.sample(&mut self.rng);
            params.push(param);
        }
        Params::new(params)
    }

    fn score(&self, fmin: f64, mean: f64, stddev: f64) -> f64 {
        let u = (fmin - mean) / stddev;
        let ei = stddev * (u * cdf(u) + pdf(u));
        // let var = stddev.powi(2);
        // let v = (fmin.ln() - mean) / stddev;
        // let ei = fmin * cdf(v) - (0.5 * var + mean).exp() * cdf(v - stddev);
        -ei
    }
}

fn cdf(x: f64) -> f64 {
    use rustats::distributions::Cdf;

    rustats::distributions::StandardNormal.cdf(&x)
}

fn pdf(x: f64) -> f64 {
    use rustats::distributions::Pdf;

    rustats::distributions::StandardNormal.pdf(&x)
}

impl Solver for RfOpt {
    fn ask(&mut self, idg: &mut IdGen) -> kurobako_core::Result<NextTrial> {
        let id = idg.generate();

        let params = if self.trials.len() < 10 {
            self.ask_random()
        } else {
            let rf = self.fit_rf().expect("TODO");
            let mut best_params = Params::new(vec![0.0]);
            let mut best_score = std::f64::INFINITY;
            let best_mean = self.best_value; //rf.predict(self.best_params.get());
            for _ in 0..5000 {
                let params = self.ask_random();
                let (mean, stddev) = mean_and_stddev(rf.predict_individuals(params.get()));
                let score = self.score(best_mean, mean, stddev); // * 2f64.powi(self.level));
                                                                 //let score = mean - stddev * 2f64.powi(self.level);
                if score < best_score {
                    best_params = params;
                    best_score = score;
                }
            }
            best_params
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
            params: params.clone().into_vec(),
            value: trial.values[0],
        });

        // eprintln!(
        //     "[{}]\t{}\t{}\t{}",
        //     self.trials.len(),
        //     self.level,
        //     trial.values[0],
        //     self.best_value
        // );

        // TODO
        if trial.values[0] < self.best_value {
            self.best_value = trial.values[0];
            self.best_params = params;
            self.level += 1;
        } else {
            self.level -= 1;
            if self.level == -3 {
                self.level = 2;
            }
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct Trial {
    pub params: Vec<f64>,
    pub value: f64,
}

fn mean_and_stddev(xs: impl Iterator<Item = f64>) -> (f64, f64) {
    let xs = xs.collect::<Vec<_>>();

    let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    let var = xs.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / xs.len() as f64;
    (mean, var.sqrt())
}
