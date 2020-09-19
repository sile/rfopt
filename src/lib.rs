use kurobako_core::problem::ProblemSpec;
use kurobako_core::rng::ArcRng;
use kurobako_core::solver::Solver;
use kurobako_core::trial::{EvaluatedTrial, IdGen, NextTrial, Params, TrialId};
use ordered_float::OrderedFloat;
use rand::distributions::Distribution as _;
use randomforest::criterion::Mse;
use randomforest::table::{ColumnType, TableBuilder};
use randomforest::{RandomForestRegressor, RandomForestRegressorOptions};
use rustats::distributions::{Cdf, Pdf};
use std::collections::HashMap;
use std::num::NonZeroUsize;
use structopt::StructOpt;

#[derive(Debug, Clone, StructOpt)]
#[structopt(rename_all = "kebab-case")]
pub struct Options {
    #[structopt(long)]
    pub rank: bool,

    #[structopt(long)]
    pub debug: bool,

    #[structopt(long)]
    pub predicted_best_mean: bool,

    #[structopt(long, default_value = "200")]
    pub trees: NonZeroUsize,

    #[structopt(long, default_value = "10")]
    pub warmup: NonZeroUsize,

    #[structopt(long, default_value = "5000")]
    pub candidates: NonZeroUsize,

    #[structopt(long, default_value = "1")]
    pub batch_size: NonZeroUsize,

    #[structopt(long, default_value = "1.0")]
    pub cap: f64,
}

#[derive(Debug)]
pub struct RfOpt {
    problem: ProblemSpec,
    trials: Vec<Trial>,
    evaluating: HashMap<TrialId, Params>,
    rng: ArcRng,
    best_value: f64,
    best_params: Params,
    options: Options,
    ask_queue: Vec<Params>,
}

impl RfOpt {
    pub fn new(seed: u64, problem: ProblemSpec, options: Options) -> Self {
        Self {
            problem,
            trials: Vec::new(),
            evaluating: HashMap::new(),
            rng: ArcRng::new(seed),
            best_value: std::f64::INFINITY,
            best_params: Params::new(Vec::new()),
            options,
            ask_queue: Vec::new(),
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
        if self.options.rank {
            let n = (self.trials.len() as f64 * self.options.cap) as usize;
            for (rank, t) in self.trials.iter().enumerate() {
                table.add_row(&t.params, std::cmp::min(n, rank) as f64)?;
            }
        } else {
            let n = (self.trials.len() as f64 * self.options.cap) as usize;
            for t in &self.trials[..n] {
                table.add_row(&t.params, t.value)?;
            }
            for t in &self.trials[n..] {
                table.add_row(&t.params, self.trials[n].value)?;
            }
        }

        // TODO: set seed
        Ok(RandomForestRegressorOptions::new()
            .trees(self.options.trees)
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
        -ei
    }
}

fn cdf(x: f64) -> f64 {
    rustats::distributions::StandardNormal.cdf(&x)
}

fn pdf(x: f64) -> f64 {
    rustats::distributions::StandardNormal.pdf(&x)
}

impl Solver for RfOpt {
    fn ask(&mut self, idg: &mut IdGen) -> kurobako_core::Result<NextTrial> {
        let id = idg.generate();

        if self.ask_queue.is_empty() {
            for _ in 0..self.options.batch_size.get() {
                let params = if self.trials.len() < self.options.warmup.get() {
                    self.ask_random()
                } else {
                    let rf = self.fit_rf().expect("TODO");
                    let mut best_params = Params::new(Vec::new());
                    let mut best_score = std::f64::INFINITY;
                    let best_mean = if self.options.predicted_best_mean {
                        rf.predict(self.best_params.get())
                    } else {
                        self.best_value
                    };
                    for _ in 0..self.options.candidates.get() {
                        let params = self.ask_random();
                        let (mean, stddev) = mean_and_stddev(rf.predict_individuals(params.get()));
                        let score = self.score(best_mean, mean, stddev);
                        if score < best_score {
                            best_params = params;
                            best_score = score;
                        }
                    }
                    best_params
                };
                self.ask_queue.push(params);
            }
            self.ask_queue.reverse();
        }

        let mut params = self.ask_queue.pop().expect("unreachable");
        if params.get().is_empty() {
            params = self.ask_random();
        }
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

        if trial.values[0] < self.best_value {
            self.best_value = trial.values[0];
            self.best_params = params;
        }

        if self.options.debug {
            eprintln!(
                "[{}] {}\t{}\t{}",
                self.trials.len(),
                if trial.values[0] == self.best_value {
                    "o"
                } else {
                    "x"
                },
                trial.values[0],
                self.best_value
            );
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
