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
use rustats::distributions::{Cdf, Pdf};
use std::collections::HashMap;
use std::num::NonZeroUsize;
use structopt::StructOpt;

#[derive(Debug, Clone, StructOpt)]
#[structopt(rename_all = "kebab-case")]
pub struct Options {
    #[structopt(long)]
    pub debug: bool,

    #[structopt(long)]
    pub seed: Option<u64>,

    #[structopt(long)]
    pub predicted_best_mean: bool,

    #[structopt(long, default_value = "100")]
    pub trees: NonZeroUsize,

    #[structopt(long, default_value = "8")]
    pub warmup: NonZeroUsize,

    #[structopt(long, default_value = "10000")]
    pub candidates: NonZeroUsize,

    #[structopt(long, default_value = "1")]
    pub batch_size: NonZeroUsize,

    #[structopt(long, default_value = "1.0")]
    pub rate: f64,
}

#[derive(Debug)]
pub struct RfOpt {
    problem: ProblemSpec,
    trusted: Vec<kurobako_core::domain::Variable>,
    trials: Vec<Trial>,
    evaluating: HashMap<TrialId, Params>,
    rng: ArcRng,
    best_value: f64,
    best_params: Params,
    options: Options,
    ask_queue: Vec<Params>,
    l: f64,
}

impl RfOpt {
    pub fn new(mut seed: u64, problem: ProblemSpec, options: Options) -> Self {
        if let Some(s) = options.seed {
            seed = s; // for debug
        }

        if options.debug {
            eprintln!("# SEED: {}", seed);
        }
        let trusted = problem.params_domain.variables().to_owned();
        Self {
            problem,
            trials: Vec::new(),
            evaluating: HashMap::new(),
            rng: ArcRng::new(seed),
            best_value: std::f64::INFINITY,
            best_params: Params::new(Vec::new()),
            options,
            ask_queue: Vec::new(),
            trusted,
            l: 1.0,
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
        for (rank, t) in self
            .trials
            .iter()
            .filter(|t| self.is_trusted(&t.params))
            .enumerate()
        {
            if self.is_trusted(&t.params) {
                table.add_row(&t.params, rank as f64)?;
            }
        }

        // TODO: set seed
        Ok(RandomForestRegressorOptions::new()
            .seed(self.rng.gen())
            .trees(self.options.trees)
            .fit(Mse, table.build()?))
    }

    fn ask_random(&mut self) -> Params {
        let mut params = Vec::new();
        for p in &self.trusted {
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

    fn shrink_tr(&mut self) {
        self.l = self.l * self.options.rate;
        self.update_tr();
    }

    fn update_tr(&mut self) {
        use kurobako_core::domain::{self, Range};

        let mut trusted = Vec::new();
        for (p, v) in self
            .problem
            .params_domain
            .variables()
            .iter()
            .zip(self.best_params.iter())
        {
            match p.range() {
                Range::Categorical { .. } => {
                    // TODO: handle categorical
                    trusted.push(p.clone());
                }
                Range::Continuous { low, high } => {
                    if p.distribution() == kurobako_core::domain::Distribution::Uniform {
                        let half_size = (high - low) * self.l / 2.0;
                        let center = *v;
                        let mut new_low = center - half_size;
                        let mut new_high = center + half_size;
                        if new_low < *low {
                            new_high += *low - new_low;
                            new_low = *low;
                        } else if new_high > *high {
                            new_low -= new_high - *high;
                            new_high = *high;
                        }
                        trusted.push(
                            domain::var(p.name())
                                .continuous(new_low.max(*low), new_high.min(*high))
                                .finish()
                                .expect("TOOD"),
                        );
                    } else {
                        let half_size = (high.exp2() - low.exp2()) * self.l / 2.0;
                        let center = v.exp2();
                        let mut new_low = center - half_size;
                        let mut new_high = center + half_size;
                        if new_low < low.exp2() {
                            new_high += low.exp2() - new_low;
                            new_low = low.exp2();
                        } else if new_high > high.exp2() {
                            new_low -= new_high - high.exp2();
                            new_high = high.exp2();
                        }
                        trusted.push(
                            domain::var(p.name())
                                .continuous(new_low.log2().max(*low), new_high.log2().min(*high))
                                .log_uniform()
                                .finish()
                                .expect("TOOD"),
                        );
                    }
                }
                Range::Discrete { low, high } => {
                    let half_size = ((high - low) as f64 * self.l / 2.0).round() as i64;
                    let center = *v as i64;
                    let mut new_low = center - half_size;
                    let mut new_high = center + half_size;
                    if new_low < *low {
                        new_high += *low - new_low;
                        new_low = *low;
                    } else if new_high > *high {
                        new_low -= new_high - *high;
                        new_high = *high;
                    }
                    if new_low == new_high {
                        new_high += 1;
                    }
                    trusted.push(
                        domain::var(p.name())
                            .discrete(std::cmp::max(new_low, *low), std::cmp::min(new_high, *high))
                            .finish()
                            .expect("TOOD"),
                    );
                }
            }
        }
        self.trusted = trusted;
    }

    fn is_trusted(&self, params: &[f64]) -> bool {
        self.trusted
            .iter()
            .zip(params.iter())
            .all(|(p, v)| p.range().contains(*v))
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
                let mut params = if self.trials.len() < self.options.warmup.get() {
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
                if params.get().is_empty() {
                    if self.options.debug {
                        eprintln!("# WARN");
                    }
                    params = self.ask_random();
                }
                self.ask_queue.push(params);
            }
            self.ask_queue.reverse();
        }

        let params = self.ask_queue.pop().expect("unreachable");
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

        if self.trials.len() % 16 == 0 {
            self.shrink_tr();
        }

        if self.options.debug {
            eprintln!(
                "[{}] {}\t{}\t{}: tr={}",
                self.trials.len(),
                if trial.values[0] == self.best_value {
                    "o"
                } else {
                    "x"
                },
                trial.values[0],
                self.best_value,
                self.l
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
