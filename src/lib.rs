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

    #[structopt(long, default_value = "1.0")]
    pub shrink_rate: f64,
}

pub struct Gp {
    inner: friedrich::gaussian_process::GaussianProcess<
        //friedrich::kernel::Gaussian,
        friedrich::kernel::Matern2,
        friedrich::prior::ConstantPrior,
    >,
    trusted: Vec<kurobako_core::domain::Variable>,
}

impl Gp {
    pub fn new<F: Fn(&[f64]) -> bool>(
        trusted: Vec<kurobako_core::domain::Variable>,
        trials: &[Trial],
        f: F,
    ) -> Self {
        let mut xss = Vec::new();
        let mut ys = Vec::new();

        for (rank, trial) in trials.iter().filter(|t| f(&t.params)).enumerate() {
            let params = Self::normalize(&trusted, &trial.params);
            xss.push(params);
            ys.push(rank as f64);
        }

        let inner = friedrich::gaussian_process::GaussianProcess::builder(xss, ys)
            .set_kernel(friedrich::kernel::Matern2::default())
            .fit_kernel()
            .fit_prior()
            .train();
        //inner.fit_parameters(true, true, 100, 0.05);
        Self { trusted, inner }
    }

    pub fn predict(&self, params: &[f64]) -> (f64, f64) {
        let params = Self::normalize(&self.trusted, params);
        let mean = self.inner.predict(&params);
        let var = self.inner.predict_variance(&params);
        (mean, var.sqrt())
    }

    fn normalize(trusted: &[kurobako_core::domain::Variable], params: &[f64]) -> Vec<f64> {
        trusted
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
    sames: usize,
    prev: f64,
    l: f64,
    trusted: Vec<kurobako_core::domain::Variable>,
    best_params: Params,
}

impl Rfopt {
    pub fn new(seed: u64, problem: ProblemSpec, options: Options) -> Self {
        let trusted = problem.params_domain.variables().to_owned();

        Self {
            problem,
            trials: Vec::new(),
            evaluating: HashMap::new(),
            rng: ArcRng::new(seed),
            best_value: std::f64::INFINITY,
            options,
            sames: 0,
            prev: std::f64::NAN,
            l: 1.0,
            trusted,
            best_params: Params::new(Vec::new()),
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
        for (rank, trial) in self
            .trials
            .iter()
            .filter(|t| self.is_trusted(&t.params))
            .enumerate()
        {
            table.add_row(&trial.params, rank as f64)?;
        }

        Ok(RandomForestRegressorOptions::new()
            .trees(self.options.trees)
            .seed(self.rng.gen())
            .fit(Mse, table.build()?))
    }

    fn fit_gp(&mut self) -> anyhow::Result<Gp> {
        use anyhow::anyhow;

        let gp = crossbeam::scope(|scope| {
            let h = scope.spawn(|_| {
                self.trials.sort_by_key(|t| OrderedFloat(t.value));
                let gp = Gp::new(self.trusted.clone(), &self.trials, |params| {
                    self.is_trusted(params)
                });
                gp
            });
            h.join()
        })
        .map_err(|e| anyhow!("{:?}", e))?
        .map_err(|e| anyhow!("{:?}", e))?;
        Ok(gp)
    }

    fn ask_random(&mut self) -> Params {
        let mut params = Vec::new();
        for p in &self.trusted {
            let param = p.sample(&mut self.rng);
            params.push(param);
        }
        Params::new(params)
    }

    fn shrink_tr(&mut self) {
        self.l = self.l * self.options.shrink_rate;
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

    fn ask_tpe(&mut self) -> Params {
        let mut optimizers = Vec::new();
        for p in &self.trusted {
            let r = Self::warp_range(p);
            optimizers.push(tpe::TpeOptimizer::new(tpe::parzen_estimator(), r));
        }

        for trial in self.trials.iter().filter(|t| self.is_trusted(&t.params)) {
            for ((o, p), v) in optimizers
                .iter_mut()
                .zip(trial.params.iter().copied())
                .zip(self.trusted.iter())
            {
                let p = Self::warp_param(p, v);
                o.tell(p, trial.value).expect("unreachable");
            }
        }

        let rng = &mut self.rng;
        let params = optimizers
            .iter_mut()
            .zip(self.trusted.iter())
            .map(|(o, p)| Self::unwarp_param(o.ask(rng).expect("unreachable"), p))
            .collect();
        Params::new(params)
    }

    fn warp_range(p: &kurobako_core::domain::Variable) -> tpe::range::Range {
        match p.range() {
            kurobako_core::domain::Range::Continuous { low, high } => {
                if p.distribution() == kurobako_core::domain::Distribution::Uniform {
                    tpe::range(*low, *high).expect("unreachable")
                } else {
                    tpe::range(low.ln(), high.ln()).expect("unreachable")
                }
            }
            kurobako_core::domain::Range::Discrete { low, high } => {
                tpe::range(*low as f64, *high as f64).expect("unreachable")
            }
            kurobako_core::domain::Range::Categorical { choices } => {
                // TODO: For a categorical parameter, we should use `HistogramEstimator` instead `ParzenEstimator`.
                tpe::range(0.0, choices.len() as f64).expect("unreachable")
            }
        }
    }

    fn warp_param(p: f64, v: &kurobako_core::domain::Variable) -> f64 {
        match v.range() {
            kurobako_core::domain::Range::Continuous { .. } => {
                if v.distribution() == kurobako_core::domain::Distribution::Uniform {
                    p
                } else {
                    p.ln()
                }
            }
            kurobako_core::domain::Range::Discrete { .. } => p + 0.5,
            kurobako_core::domain::Range::Categorical { .. } => p,
        }
    }

    fn unwarp_param(p: f64, v: &kurobako_core::domain::Variable) -> f64 {
        match v.range() {
            kurobako_core::domain::Range::Continuous { .. } => {
                if v.distribution() == kurobako_core::domain::Distribution::Uniform {
                    p
                } else {
                    p.exp()
                }
            }
            kurobako_core::domain::Range::Discrete { .. } => p.floor(),
            kurobako_core::domain::Range::Categorical { .. } => p.round(),
        }
    }
}

impl Solver for Rfopt {
    fn ask(&mut self, idg: &mut IdGen) -> kurobako_core::Result<NextTrial> {
        let id = idg.generate();
        let mut next_params = Params::new(Vec::new());

        if self.trials.len() >= self.options.warmup.get() {
            if self.trials.len() % 3 == 0 {
                match self.fit_gp() {
                    Err(e) => {
                        eprintln!("Cannot fit GP model: {}", e);
                    }
                    Ok(gp) => {
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
                    }
                }
            }
            if next_params.get().is_empty() || self.trials.len() % 3 == 1 {
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
            if next_params.get().is_empty() {
                next_params = self.ask_tpe();
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

        // eprintln!(
        //     "[{}]\t{}\t{} ({}, {})",
        //     self.trials.len(),
        //     trial.values[0],
        //     self.best_value,
        //     self.sames,
        //     self.trials
        //         .iter()
        //         .filter(|t| !self.is_trusted(&t.params))
        //         .count()
        // );

        if (trial.values[0] - self.prev).abs() < std::f64::EPSILON {
            self.sames += 1;
        } else {
            self.sames = 0;
        }
        self.prev = trial.values[0];
        if self.sames >= 5 {
            self.trials.clear();
            self.sames = 0;
            self.prev = std::f64::NAN;
            self.trusted = self.problem.params_domain.variables().to_owned();
            self.l = 1.0;
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

fn ei(fmin: f64, mean: f64, stddev: f64) -> f64 {
    let u = (fmin - mean) / stddev;
    stddev * (u * StandardNormal.cdf(&u) + StandardNormal.pdf(&u))
}
