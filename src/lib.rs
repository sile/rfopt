use kurobako_core::problem::ProblemSpec;
use kurobako_core::rng::ArcRng;
use kurobako_core::solver::Solver;
use kurobako_core::trial::{EvaluatedTrial, IdGen, NextTrial, Params, TrialId};
use ordered_float::OrderedFloat;
use rand::distributions::Distribution as _;
use rand::Rng;
use randomforest::criterion::Mse;
use randomforest::table::{ColumnType, TableBuilder};
use randomforest::{RandomForestRegressor, RandomForestRegressorOptions};
use rustats::distributions::StandardNormal;
use rustats::distributions::{Cdf, Pdf};
use std::collections::HashMap;
use std::num::NonZeroUsize;
use structopt::StructOpt;

#[derive(Debug, Clone, StructOpt)]
#[structopt(rename_all = "kebab-case")]
pub struct Options {
    #[structopt(long, default_value = "100")]
    pub trees: NonZeroUsize,

    #[structopt(long, default_value = "10000")]
    pub candidates: NonZeroUsize,

    #[structopt(long, default_value = "10")]
    pub warmup: NonZeroUsize,

    #[structopt(long)]
    pub gp: bool,
}

pub struct Gp {
    inner: friedrich::gaussian_process::GaussianProcess<
        friedrich::kernel::Gaussian,
        friedrich::prior::ConstantPrior,
    >,
    problem: ProblemSpec,
}

impl Gp {
    pub fn new(problem: &ProblemSpec, trials: &[Trial]) -> Self {
        let mut xss = Vec::new();
        let mut ys = Vec::new();

        for (rank, trial) in trials.iter().enumerate() {
            let params = Self::normalize(problem, &trial.params);
            xss.push(params);
            ys.push(rank as f64);
        }

        let mut inner = friedrich::gaussian_process::GaussianProcess::default(xss, ys);
        inner.fit_parameters(true, true, 100, 0.05);
        Self {
            problem: problem.clone(),
            inner,
        }
    }

    pub fn predict(&self, params: &[f64]) -> (f64, f64) {
        let params = Self::normalize(&self.problem, params);
        let mean = self.inner.predict(&params);
        let var = self.inner.predict_variance(&params);
        (mean, var.sqrt())
    }

    fn normalize(problem: &ProblemSpec, params: &[f64]) -> Vec<f64> {
        problem
            .params_domain
            .variables()
            .iter()
            .zip(params.iter())
            .map(|(p, v)| {
                if p.distribution() == kurobako_core::domain::Distribution::LogUniform {
                    let size = p.range().high().ln() - p.range().low().ln();
                    v.ln() / size
                } else {
                    let size = p.range().high() - p.range().low();
                    v / size
                }
            })
            .collect()
    }
}

#[derive(Debug)]
pub struct Rfopt {
    problem: ProblemSpec,
    trials: Vec<Trial>,
    evaluating: HashMap<TrialId, Params>,
    rng: ArcRng,
    best_value: f64,
    options: Options,
}

impl Rfopt {
    pub fn new(seed: u64, problem: ProblemSpec, options: Options) -> Self {
        Self {
            problem,
            trials: Vec::new(),
            evaluating: HashMap::new(),
            rng: ArcRng::new(seed),
            best_value: std::f64::INFINITY,
            options,
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
        for (rank, trial) in self.trials.iter().enumerate() {
            table.add_row(&trial.params, rank as f64)?;
        }

        Ok(RandomForestRegressorOptions::new()
            .trees(self.options.trees)
            .seed(self.rng.gen())
            .fit(Mse, table.build()?))
    }

    fn fit_gp(&mut self) -> anyhow::Result<Gp> {
        self.trials.sort_by_key(|t| OrderedFloat(t.value));
        let gp = Gp::new(&self.problem, &self.trials);
        Ok(gp)
    }

    fn ask_random(&mut self) -> Params {
        let rng = &mut self.rng;
        Params::new(
            self.problem
                .params_domain
                .variables()
                .iter()
                .map(|p| p.sample(rng))
                .collect(),
        )
    }
}

impl Solver for Rfopt {
    fn ask(&mut self, idg: &mut IdGen) -> kurobako_core::Result<NextTrial> {
        let id = idg.generate();
        let mut next_params = Params::new(Vec::new());

        if self.trials.len() >= self.options.warmup.get() {
            if self.options.gp {
                let gp = self.fit_gp().expect("TODO: error handling");
                let mut best_ei = std::f64::NEG_INFINITY;
                let best_mean = self.best_value;
                for _ in 0..self.options.candidates.get() {
                    let params = self.ask_random();
                    let (mean, stddev) = gp.predict(params.get());
                    let ei = ei(best_mean, mean, stddev);
                    if ei > best_ei {
                        next_params = params;
                        best_ei = ei;
                    }
                }
            } else {
                let rf = self.fit_rf().expect("TODO: error handling");
                let mut best_ei = std::f64::NEG_INFINITY;
                let best_mean = self.best_value;
                for _ in 0..self.options.candidates.get() {
                    let params = self.ask_random();
                    let (mean, stddev) = mean_and_stddev(rf.predict_individuals(params.get()));
                    let ei = ei(best_mean, mean, stddev);
                    if ei > best_ei {
                        next_params = params;
                        best_ei = ei;
                    }
                }
            }
        }

        if next_params.get().is_empty() {
            next_params = self.ask_random();
        }
        self.evaluating.insert(id, next_params.clone());

        Ok(NextTrial {
            id,
            params: next_params,
            next_step: Some(self.problem.steps.last()),
        })
    }

    fn tell(&mut self, trial: EvaluatedTrial) -> kurobako_core::Result<()> {
        let params = self
            .evaluating
            .remove(&trial.id)
            .expect("TODO: error handling");
        self.trials.push(Trial {
            params: params.into_vec(),
            value: trial.values[0],
        });

        if trial.values[0] < self.best_value {
            self.best_value = trial.values[0];
        }

        eprintln!(
            "[{}]\t{}\t{}",
            self.trials.len(),
            trial.values[0],
            self.best_value
        );

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

fn ei(fmin: f64, mean: f64, stddev: f64) -> f64 {
    let u = (fmin - mean) / stddev;
    stddev * (u * StandardNormal.cdf(&u) + StandardNormal.pdf(&u))
}
