// pub mod error;
// pub use error::{Error, Result};
use color_eyre::eyre::{Result, eyre};

use core::{f64::consts::PI, ops::Range};
use ndarray::{Array1, Array2, ArrayView1, parallel::prelude::*, s};
use ndarray_npy::{read_npy, write_npy};

use rayon::iter::{IntoParallelIterator, ParallelExtend, ParallelIterator};
use ruviz::{plots::LineConfig, prelude::*};
use scirs2::{
    prelude::Zero,
    sparse::{AsLinearOperator, linalg::expm_multiply, sparse_diags},
};
use std::{iter::repeat_n, num::NonZeroUsize, sync::Mutex, thread, time::Instant};
use strum::IntoEnumIterator;
use strum_macros::EnumIter; // Allows us to use .iter()

/// Non-negative number, alpha square indicating the average number of pump photons at time = 0.
/// Also the mean and variance of the distribution
#[derive(Debug, EnumIter, Clone, Copy, PartialEq, Eq)]
pub enum AlphaSquare {
    //  with pb_th = 1e-16
    U1e1 = 10, // relevant_poisson_idxs_len =       46:      (0..     45), hilbert_space_dim =       1_127 * 5001 t_step: 0.043 GB
    U1e2 = 100, // relevant_poisson_idxs_len =     163:     (30..    192), hilbert_space_dim =      18_149 * 501  t_step: 0.070 GB
    U1e3 = 1_000, // relevant_poisson_idxs_len =   509:    (756..  1_264), hilbert_space_dim =     515_708 * 161  t_step: 0.633 GB
    U2e3 = 2_000, // relevant_poisson_idxs_len =   717:  (1_652..  2_368), hilbert_space_dim =   1_442_604 * 220  t_step: 2.421 GB
                  // U1e4 = 10_000, // relevant_poisson_idxs_len = 1_583:  (9_219.. 10_801), hilbert_space_dim =  15_847_413 * 400 t_step: 50.71 GB
                  // U1e5 = 100_000, // relevant_poisson_idxs_len = 4_912: (97_554..102_465), hilbert_space_dim = 491_251_576* 160 t_step: 628.8 GB
}
// const _E_ALPHA_SQUARE: AlphaSquare = AlphaSquare::U1e3;
// const _ALPHA_SQUARE: u16 = _E_ALPHA_SQUARE as u16;

/// non-negative real number. It is the probability threshold that is
/// used for truncating the Hilbert space. It cuts off the basis
/// elements whose associated probability is smaller than `PB_TH` at
/// initial time.
/// Threshold that determines the range of number states that are included
/// in the input coherent state: `|psi_in_pump> = \sum_{n=n1}^{n2} c_{n} |n>`
/// where n values are chosen such that `|c_{n}|^{2} > PB_TH`
/// Use `PB_TH >= 1` for fock state with pump-mode photon number = `alpha_square` as initial condition
const PB_TH: f64 = 1e-16;

const MAX_DELTA_T: f64 = 0.000_1;
const MID_DELTA_T: f64 = MAX_DELTA_T / 2.; //   0.000_05
const MIN_DELTA_T: f64 = 0.000_01;
/// Returns the `delta_t` for a given `e_alpha_square`
const fn delta_t_of_alpha_square(e_alpha_square: AlphaSquare) -> f64 {
    match e_alpha_square as u16 {
        ..=1_000 => MAX_DELTA_T,
        5_000.. => MIN_DELTA_T,
        _ => MID_DELTA_T,
    }
}
/// Precision used on the rounding of `time_t`s
const ROUNDING_DELTA_T: f64 = 1.0 / (MAX_DELTA_T / 100.); // 1. / 0.000_001 == 1_000_000.0

///  Allows to apply `round_t()` directly on `f64` such as `time_t`s
trait RoundingT {
    fn round_t(self) -> Self;
}
impl RoundingT for f64 {
    fn round_t(self) -> Self {
        (self * ROUNDING_DELTA_T).round() / ROUNDING_DELTA_T
    }
}

// Time step for the evolution
// Optimized parameters:
// - `DELTA_T = 0.000_1`  for `ALPHA_SQUARE <= 1_000`,
// - `DELTA_T = 0.000_05` for `ALPHA_SQUARE = 2_000`,
// - `DELTA_T = 0.000_01` for `ALPHA_SQUARE >= 5_000`

// const _DELTA_T: f64 = delta_t_of_alpha_square(_E_ALPHA_SQUARE);

/// Return the total time of the evolution
///
/// `end_t` is chosen to observe the dynamics until the first local maximum in signal-mode population.
/// Optimized parameters:
/// - `end_t = 0.5` for `ALPHA_SQUARE = 10`,
/// - `end_t = 0.05` for `ALPHA_SQUARE = 100`,
/// - `end_t = 0.016` for `ALPHA_SQUARE = 1_000`,
/// - `end_t = 0.011` for `ALPHA_SQUARE = 2_000`,
/// - `end_t = 0.005` for `ALPHA_SQUARE = 10_000`,
/// - `end_t = 0.001_6` for `ALPHA_SQUARE = 100_000`,
const fn end_t_of_alpha_square(e_alpha_square: AlphaSquare) -> f64 {
    match e_alpha_square {
        AlphaSquare::U1e1 => 0.5,
        AlphaSquare::U1e2 => 0.05,
        AlphaSquare::U1e3 => 0.016,
        AlphaSquare::U2e3 => 0.011,
        // AlphaSquare::U1e4 => 0.005,
        // AlphaSquare::U1e5 => 0.001_6,
    }
}
// const END_T: f64 = end_t_of_alpha_square(E_ALPHA_SQUARE);

/// Returns the `step_cnt` value for a given `e_alpha_square`.
/// `step_cnt` is the dimension used to generate the `time_ts` vector with steps of `delta_t`.
const fn step_cnt_of_alpha_square(e_alpha_square: AlphaSquare) -> u16 {
    //  Considering the values defined above there will be neither sign_loss nor possible_truncation
    #[allow(clippy::cast_sign_loss)]
    #[allow(clippy::cast_possible_truncation)]
    let step_cnt = (end_t_of_alpha_square(e_alpha_square) / delta_t_of_alpha_square(e_alpha_square)) as u16;
    step_cnt
}

// const _STEP_CNT: u16 = step_cnt_of_alpha_square(_E_ALPHA_SQUARE);

/// Generates the photon population states evolution for 3rd harmonic generation
///
/// ### Parameters
///
/// - `e_alpha_square` - mean and variance of the distribution; one of {`U1e1`, `U1e2`, `U1e3` and `U2e3`}
/// - `verbose` - verbosity of the run
///
/// ### Errors
///
///  - May return `Err` from the `ruviz` crate `subplots()` function and `SubplotFigure::subplot()`
///    and `SubplotFigure::save()` methods
pub fn run(e_alpha_square: AlphaSquare, verbose: bool) -> Result<()> {
    if (false, verbose).0 {
        for e in AlphaSquare::iter() {
            println!(
                "alpha_square::{e:?}={}, delta_t = {}, end_t={}, step_cnt={}",
                e as u32,
                delta_t_of_alpha_square(e),
                end_t_of_alpha_square(e),
                step_cnt_of_alpha_square(e)
            );
        }
    }

    let alpha_square = e_alpha_square as u16;
    let delta_t = delta_t_of_alpha_square(e_alpha_square);
    let step_cnt = step_cnt_of_alpha_square(e_alpha_square);

    let start = Instant::now();
    let evolution = StateEvolution::new(alpha_square, delta_t, step_cnt, PB_TH, verbose)?;
    println!(
        "StateEvolution::new(alpha_square={alpha_square}) finished on {} cores in {:?}",
        available_parallelism(),
        start.elapsed()
    );
    interpret_and_plot(&evolution, e_alpha_square, delta_t, verbose)?;

    if (false, verbose).0 {
        println!("3rd harmonic generation is done.");
    }
    Ok(())
}

/// Returns the Poisson probability mass function (pmf) distribution over (11 x `alpha_square`) points
///
/// The distribution is built over `mean + 10 x variance` points.
/// Note that "optimised" `scirs2_stats::distributions::poisson::Poisson::pmf()` version `0.4.2`
/// [is broken](assets/weird_poisson_results.md). So, we do it by hand for now.
///
/// ### Parameters
///
/// - `alpha_square` - non-negative mean and variance of the distribution; one of {`10`, `100`, `1_000` and `2_000`}
/// - `verbose` - verbosity
#[must_use]
fn poisson_distribution(alpha_square: u16, verbose: bool) -> Array1<f64> {
    // `k` goes from 0 to mean + (10 x variance). Since mean = variance = alpha_square:
    let k_range = 0..(11 * alpha_square); //  so: 0..110, 0..1_100, 0..11_000, 0..22_000
    if verbose {
        print!("poisson_distribution : k_range {k_range:?}, ");
    }

    // `scirs2_stats::distributions::poisson::Poisson::pmf()` version 0.4.2 was broken so we do it by hand :
    let mu = f64::from(alpha_square);
    Array1::<f64>::from_vec(
        k_range
            .map(|k| {
                // factorial(170) is the maximum that fits into f64::MAX (1.7976931348623157e308),
                // but so is 2_000.0_f64.powi(93), for alternative max value of alpha_square 2_000
                // but so is 100_000.0_f64.powi(61), for max value of alpha_square 100_000
                if (0..=93_u16).contains(&k) {
                    // k won't wrap an i32 as it's contained in 0..=93_u32.
                    #[allow(clippy::cast_possible_wrap)]
                    let k = i32::from(k);
                    // straight factorial method
                    mu.powi(k) * (-mu).exp() / (1..=k).map(f64::from).product::<f64>()
                } else {
                    // e**ln_factorial method
                    let k = f64::from(k);
                    (k.mul_add(mu.ln(), -mu) -
                        //  that's ln_factorial(k)
                        if k <= 1.0 {
                            0.0
                        } else {
                            0.5_f64.mul_add((2.0 * PI * k).ln(), k.mul_add(k.ln(), -k))
                        })
                    .exp()
                }
            })
            .collect(),
    )
}

/// Returns the values of `n` whose Poisson `prob(n, mean=alpha_square) > pb_th`.
///     If `pb_th >= 1.0`, it returns only the value `n` = `alpha_square`
///
/// ### Parameters
///
/// - `dist` - an Array1 of `scirs2::stats::distributions::Poisson<f64>::pmf()`
/// - `alpha_square` - non-negative mean _and_ variance of the distribution
/// - `pb_th` - non-negative  threshold for identifying the indices whose probability is greater
///   than `pb_th`. Use `pb_th` >= 1 for Fock states with photon number `n` = `alpha_square`
///
#[must_use]
fn relevant_poisson_distribution_indices(dist: &[f64], alpha_square: u16, pb_th: Option<f64>) -> Vec<u16> {
    let pb_th = pb_th.unwrap_or_else(|| 10.0_f64.powi(-16));

    if pb_th >= 1. {
        vec![alpha_square]
    } else {
        // the indices that satisfy the threshold.
        dist.par_iter()
            .enumerate()
            .filter_map(|(i, v)|
                    //  We skip index 0 because scirs2 expm_multiply() isn't able to deal with a [0] hamiltonian.
                    if *v > pb_th  &&  i > 0 {
                        // i < 22_000 < u16::MAX, bound by max AlphaSquare == 2_000
                        #[allow(clippy::cast_possible_truncation)]
                        let i = i as u16;
                        Some(i)
                    } else {
                        None
                    }
                )
            .collect()
    }
}

/// Returns lower diagonal elements of the Hamiltonian in the subspace defined by `num_pump_max`
/// corresponding to `ind_nu` of the matrix with size `num_pump_max + 1`
///
/// ### Parameters
///
/// - `num_pump_max` -  the maximum number of pump photons allowed in the subspace, whose Hamiltonian
///   elements are outputted from this function
#[must_use]
pub fn lower_diagonal_of_hamiltonian_in_subspace_of(num_pump_max: u16) -> Array1<f64> {
    // enum `AlphaSquare` enforces that `alpha_square` be at max 100_000, but 2_000 really.
    // Therefore poisson_distribution(alpha_square).len() is at max 11 x 2_000 = 22_000
    // Therefore, the num_pump_max indices iterated from its derived
    // relevant_poisson_distribution_indices(&dist) are < 22_000 too
    // So, num_pump_max can be cast into f64 without loss.
    let num_pump_max_f = f64::from(num_pump_max);
    // - `idx_nu` - indices ranging from `0` to `num_pump_max-1`, corresponding to lower-diagonal elements
    Array1::<f64>::range(0., num_pump_max_f, 1.).mapv_into(|idx_f| {
        ((num_pump_max_f - idx_f)
            * 3.0_f64.mul_add(idx_f, 1.)
            * 3.0_f64.mul_add(idx_f, 2.)
            * 3.0_f64.mul_add(idx_f, 3.))
        .sqrt()
    })
}

/// Returns `thread::available_parallelism()` Ok value, or 1 if it fails.
fn available_parallelism() -> usize {
    const NO_AVAILABLE_PARALLELILSM: NonZeroUsize = NonZeroUsize::new(1).expect("1 is not 0");
    thread::available_parallelism()
        .unwrap_or(NO_AVAILABLE_PARALLELILSM)
        .get()
}

/// Returns the optimal `chunk_size` for a table of size `dim`, considering the `available_parallelism()`.
fn chunk_size(dim: usize) -> usize {
    let available_parallelism = available_parallelism();
    dim / available_parallelism + usize::from(!dim.is_multiple_of(available_parallelism))
}

/// Wrapper struct around a `(Range<usize>, Array2<T>)` used to `par_expend()` an `Array2ColAssemblyInstruction`.
struct Array2ColAssemblyInstruction<T>((Range<usize>, Array2<T>));

/// Wrapper around an `Array2<T>` meant to be initialized with zeros by `new()` and `par_expend()`-ed from
/// `Array2ColAssemblyInstruction` yielded by multiple threads.
///
/// It implements `rayon::iter::ParallelExtend` to fill this inner `Array2<T>` with an `Array2<T>` which
/// rows `dim()` matches that of this inner `Array2<T>`, and which column `dim()` matches a column range
/// passed along with it, so this inner `Array2<T>` can be parallely assigned over all rows and a given
/// colums range. A `Mutex` wraps the inner `Array2<T>` to allow parrallel assembly from multiple thread.
struct ColAssembledArray2<T>(Mutex<Array2<T>>);

impl<T: std::clone::Clone + scirs2::prelude::Zero> ColAssembledArray2<T> {
    /// Creates a `ColAssembledArray2<T>` witha zeroed inner `Array2<T>` of dimension `row_dim` by `col_dim`.
    fn new(row_dim: usize, col_dim: usize) -> Self {
        Self(Mutex::new(Array2::<T>::zeros((row_dim, col_dim))))
    }

    /// Consumes self to return the inner `Array2<T>`.
    fn into_array(self) -> Array2<T> {
        self.0.into_inner().expect("Available Array2")
    }
}

impl<T: Send + Sync + Clone> ParallelExtend<Array2ColAssemblyInstruction<T>> for ColAssembledArray2<T> {
    fn par_extend<I>(&mut self, par_iter: I)
    where
        I: IntoParallelIterator<Item = Array2ColAssemblyInstruction<T>>,
    {
        par_iter
            .into_par_iter()
            .for_each(|Array2ColAssemblyInstruction::<T>((col_range, array_to_add))| {
                if let Ok(mut col_assembled_array) = self.0.lock() {
                    col_assembled_array
                        .slice_mut(s![.., col_range])
                        .assign(&array_to_add);
                }
            });
    }
}

/// Extra length used for shape of the `hamiltonian`s and `intial_state`s widht and `idx_dim` increment because,
/// the offsets of the `lower_diag` and `upper_diag` in the `hamiltonian` are `[-1, 1]`, and the dimension of
/// the `hamiltonian` is the max of `diag.len() + diag_offset.abs()` for all diags, so : `idx + 1`.
const DIAGO_OFFSET_IN_HAMIL: u16 = 1;
const HAMIL_XTRA: u16 = DIAGO_OFFSET_IN_HAMIL;
const TWO_HAMIL_XTRA_MINUS_ONE: u16 = 2 * HAMIL_XTRA - 1;
const HAMIL_XTRA_LEN: usize = HAMIL_XTRA as usize;

/// Helper function used in `IdxItem::new()`.
fn state_offset(idx0: u16, idx: u16) -> usize {
    usize::from(idx - idx0) * usize::from(idx + idx0 + TWO_HAMIL_XTRA_MINUS_ONE) / 2
}

/// Helper struct used in `StateEvolution::new()`.
struct IdxItem {
    idx: usize,
    idx_dim: usize,
    idx_range: Range<usize>,
}

impl IdxItem {
    fn new(idx0: u16, idx: u16) -> Self {
        Self {
            idx: usize::from(idx),
            idx_dim: usize::from(idx + HAMIL_XTRA),
            idx_range: state_offset(idx0, idx)..state_offset(idx0, idx + 1),
        }
    }
}

/// struct resulting from the population evolution for 3rd harmonic generation.
pub struct StateEvolution {
    pub time_ts: Array1<f64>,
    pub states_coeff: Array1<u16>,
    pub states: Array2<f64>,
}

impl StateEvolution {
    /// Returns the Poisson distribution, the squre root of that Poisson distribution, and the
    /// relevant index of the Poisson distribution (which values are above `pb_th`)
    ///
    /// ### Parameters
    ///
    /// - `dist` - an Array1 of `scirs2::stats::distributions::Poisson<f64>::pmf()`
    /// - `alpha_square` - non-negative mean _and_ variance of the distribution
    /// - `pb_th` - non-negative  threshold for identifying the indices whose probability is greater
    ///   than `pb_th`. Use `pb_th` >= 1 for Fock states with photon number `n` = `alpha_square`
    ///
    fn poisson(alpha_square: u16, pb_th: f64, verbose: bool) -> (Array1<f64>, Vec<f64>, Vec<u16>) {
        let dist = poisson_distribution(alpha_square, verbose);

        let sqrt_dist = dist.iter().map(|v| v.sqrt()).collect();

        // Identifying "appropriate indices" in the Poisson distribution Array1.
        let relevant_poisson_idxs = relevant_poisson_distribution_indices(
            dist.as_slice().expect("poisson_distribution is contiguous"),
            alpha_square,
            Some(pb_th),
        );

        (dist, sqrt_dist, relevant_poisson_idxs)
    }

    /// Returns a `StateEvolution` struct
    ///    
    /// ### Parameters
    ///
    /// - `alpha_square` - non-negative mean and variance of the distribution
    /// - `delta_t` -  time step of the evolution.
    /// - `step_cnt` -  number of steps in the evolution.
    /// - `pb_th` - non-negative threshold for identifying the indices whose probability is greater
    ///   than `pb_th`. Use `pb_th` >= 1 for Fock states with photon number `n` = `alpha_square`
    /// - `verbose` - verbosity of the states evolution generation
    ///
    /// ### Errors
    ///
    /// May return :
    ///  - `Err()` if no point of the poisson distribution is above `pb_th` for, for µ = `alpha_square`
    ///
    /// ### Panics
    ///
    /// May panic :
    /// - if there are unexpected errors in the dimensioning of the Arrays used through `state_evolution()`
    pub fn new(alpha_square: u16, delta_t: f64, step_cnt: u16, pb_th: f64, verbose: bool) -> Result<Self> {
        // Total time of the evolution, dervied from `step_cnt`, chosen to observe
        // the dynamics until the first local maximum in signal-mode population.
        let end_t = (delta_t * f64::from(step_cnt)).round_t();
        if (false, verbose).1 {
            println!(
                "alpha_square {alpha_square}, delta_t {delta_t
                    }, step_cnt {step_cnt}, end_t {end_t}, pb_th {pb_th:e}"
            );
        }

        let (dist, sqrt_dist, relevant_poisson_idxs) = Self::poisson(alpha_square, pb_th, verbose);
        if (false, verbose).0 {
            println!("dist {dist:?}");
        }

        let relevant_poisson_idxs_len = relevant_poisson_idxs.len(); // Again, max 22_000 < u16::MAX
        if relevant_poisson_idxs_len.is_zero() {
            return Err(eyre!(
                "No point of the poisson distribution is above pb_th = {pb_th:e}, for µ = alpha_square = {alpha_square}"
            ));
        }

        // total number states |n-k>_{p} |3k>_{s} over all relevant_poisson_idxs
        let hilbert_space_dim = relevant_poisson_idxs_len * HAMIL_XTRA_LEN
            + relevant_poisson_idxs
                .iter()
                .map(|&idx| usize::from(idx))
                .sum::<usize>();

        let idx0 = relevant_poisson_idxs[0];
        let largest_relevant_poisson_idx = relevant_poisson_idxs[relevant_poisson_idxs_len - 1];
        let chunk_size = chunk_size(relevant_poisson_idxs_len);
        if (false, verbose).1 {
            println!(
                "{relevant_poisson_idxs_len} relevant_poisson_idxs: ({idx0
                    }..{largest_relevant_poisson_idx }), hilbert_space_dim {hilbert_space_dim}"
            );
        }

        let step_cnt = usize::from(step_cnt);
        let time_ts = Array1::<f64>::linspace(0., end_t, step_cnt + 1).mapv_into(f64::round_t);

        if (false, verbose).0 {
            println!("{} time_ts {time_ts}", time_ts.len());
        }

        let states_filename = format!("assets/3rdHarmGen_a_sqr{alpha_square}.npy");

        Ok(Self {
            states_coeff: Array1::from_shape_vec(
                hilbert_space_dim,
                relevant_poisson_idxs
                    .iter()
                    .flat_map(|&idx| repeat_n(idx, usize::from(idx + HAMIL_XTRA)))
                    .collect(),
            ).expect("states_coeff dimension is hilbert_space_dim"),

            states: read_npy(&states_filename).unwrap_or_else(|_| {
                // Same thing: allocate it ONCE, reuse it relevant_poisson_idxs_len times, with increasingly larger views.
                // Each "Fock state" with a significant probability in the initial state
                // passed as `v` in  `expm_multiply(a, v)` bellow, with result vector y = exp(t*A) * v,
                // so, `v.len()` must be the same as hamiltonian square dim passed in `a`: idx_dim == idx + 1
                let mut largest_initial_state = Array1::<f64>::zeros(
                    usize::from(largest_relevant_poisson_idx + HAMIL_XTRA)
                );
                largest_initial_state[0] = 1.0; // each "Fock state" with a significant probability in the initial state

                // Simulate the time evolution for each "Fock state"
                // with a significant probability in the initial state
                let mut col_assembled_states  = ColAssembledArray2::<f64>::new(time_ts.dim(), hilbert_space_dim);
                col_assembled_states.par_extend(
                    relevant_poisson_idxs
                        .into_par_iter()
                        .chunks(chunk_size)
                        .flat_map_iter( |idxs| idxs.into_iter().map(|idx| {

                    let lower_diag = lower_diagonal_of_hamiltonian_in_subspace_of(idx + 1);
                    let upper_diag = - &lower_diag;
                    let lower_diag = lower_diag.as_slice().expect("lower_diag is contiguous");
                    let upper_diag = upper_diag.as_slice().expect("upper_diag is contiguous");

                    let IdxItem {idx, idx_dim, idx_range, ..} = IdxItem::new(idx0, idx);

                    // Hamiltonian in the subspace, passed as `a` in expm_multiply(a, v) below, with result vector y = exp(t*A) * v
                    let hamiltonian = sparse_diags(&[lower_diag, upper_diag], &[-1, 1],(idx_dim, idx_dim)
                        ).expect("hamiltonian sparse_diags dimensions match");
                    // let upper_diag = [-1.0, -2.0, -3.0];
                    // let lower_diag = [ 1.0,  2.0,  3.0];
                    // let dim = 4; 
                    // assert_eq!(sparse_diags(&[&lower_diag, &upper_diag], &[-1, 1],  (dim, dim)), 
                    //            array![[0.0, -1.0,  0,0,  0,0], 
                    //                   [1.0,  0.0, -2.0,  0.0],
                    //                   [0.0,  2.0,  0.0, -3.0],
                    //                   [0.0,  0.0,  3.0,  0.0]]);
                    // https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.diags.html 
                    // https://docs.rs/scirs2-sparse/0.4.1/scirs2_sparse/sparse_functions/fn.sparse_diags.html

                    // Fock state with probability 1 for n=alpha_square initially
                    let weight_sqrt = if pb_th >= 1. { 1.0 } else { sqrt_dist[idx] };
                    if (false, verbose && 10 == alpha_square).0  &&  23 == idx {
                        println!("idx_dim {idx_dim}, idx_range() {idx_range:?}, weight_sqrt {weight_sqrt:?}, lower_diag {lower_diag:?}");
                    }

                    // each "Fock state" with a significant probability in the initial state
                    // passed as `v` in  `expm_multiply(a, v)` bellow, with result vector y = exp(t*A) * v, 
                    // so, `v.len()` must be the same as hamiltonian square dim passed in `a`: idx_dim == idx + 2
                    let initial_state= largest_initial_state.slice(s![0..idx_dim]);
                    let mut local_states = Array2::<f64>::zeros((time_ts.dim(), idx_dim));

                    let mut prev_state = initial_state.as_slice().expect("initial_state is contiguous");
                    let mut expm: Array1::<f64>;    //  declared here so that prev_state borrowed value lives long enough.
                    for (i, &time_t) in time_ts.iter().enumerate() {

                        // For more details on expm_multiply, see both:
                        // - https://docs.rs/scirs2-sparse/0.4.1/scirs2_sparse/linalg/fn.expm_multiply.html, and
                        // - https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.expm_multiply.html

                        expm = Array1::from_vec(expm_multiply(
                                hamiltonian.as_linear_operator().as_ref(),
                                prev_state, delta_t, None, None)
                            .expect("hamiltonian is square and of same dimension as initial_state"));

                        local_states.slice_mut(s![i, ..]).assign(&expm);

                        prev_state = expm.as_slice().expect("expm is contiguous");

                        #[allow(clippy::float_cmp)]
                        if (false, verbose && 10 == alpha_square).0  &&  23 == idx  &&  0.25 == time_t {
                            println!("time_t {time_t}, expm {expm:?},");
                        }
                    }
                    local_states *= weight_sqrt;
                    Array2ColAssemblyInstruction((idx_range, local_states))
                })));

                let states = col_assembled_states.into_array();
                // Write the array to the file in .npy format
                if let Err(e)  = write_npy(&states_filename, &states) {
                    eprint!("Error writing srates to file: {e:#?}");
                }
                states
            }),
            time_ts,
        })
    }
}

/// Helper function for `moments()`
fn coeff(
    pump_photon_cnt: u16,
    signal_photon_cnt_third: u16,
    t_i_state: &ArrayView1<f64>,
    start: u16,
    end: u16,
) -> f64 {
    //  always fit in u16 for max alpha_square=2_000, k=2_368 in caller moments()
    let excitations = pump_photon_cnt + signal_photon_cnt_third;
    if start <= excitations && excitations <= end {
        // accounts for state vector truncation n1<= n<=n2
        // exc_ind is the number of elements in the state vector before "excitations"
        // excitations (pump+(signal/3) photons) given the way the state coefficients are arranged:
        let exc_ind = usize::from(excitations - start)
            * usize::from(start + excitations + TWO_HAMIL_XTRA_MINUS_ONE)
            / 2;
        t_i_state[exc_ind + usize::from(signal_photon_cnt_third)]
    } else {
        0.0 // to account for other unphysical cases  
    }
}

/// Helper function for `moments()`
/// returns the value of P(num, k) = num! / (num - k)! = num * (num - 1) * ... * (num - k + 1)
fn perm(num: u16, k: u16) -> f64 {
    // fn perm(num: i32, k: i32) -> f64 {
    if k > num + 1 {
        panic!("perm(num: {num}, k : {k}) :  k > num + 1");
    } else {
        ((num + 1 - k)..=num).map(f64::from).product::<f64>()
    }
}

/// Returns the expectation value of the following quantity out of the `StateEvolution` struct data:
/// <(a^dag)^{pi2}*(a)^{pi1}*(b^dag)^{sig2}*(b)^{sig1}>
///
/// #Parameters
///
/// - `pi1` - the number that appears as `(ap)^{pi1}` in the moments
/// - `pi2` - the number that appears as `(ap^dag)^{pi2}` in the moments
/// - `sig1` - the number that appears as `(as)^{sig1}` in the moments
/// - `sig2` - the number that appears as `(as^dag)^{sig2}` in the moments
/// - `t_istate` - 1d array with the state of the system at time `t_i`
/// - `start` - start part of the range used to slice the Hibert space as it appears in
///   `psi(t)=sum_{n=n_start}^{n_end}` beta{n-k,3k}|n-k>_{p} |3k>_{s})
/// - `end` - end part of the range used to slice the Hibert space as it appears in
///   `psi(t)=sum_{n=n_start}^{n_end} beta{n-k,3k}|n-k>_{p} |3k>_{s})`
fn moments(
    pi1: u16,
    pi2: u16,
    sig1: u16,
    sig2: u16,
    t_i_state: &ArrayView1<f64>,
    start: u16,
    end: u16,
) -> f64 {
    // let u16_max = u32::from(u16::MAX);
    // let sig1_u32 = u32::from(sig1);
    // let sig2_u32 = u32::from(sig2);

    // if 0 == (i32::from(sig2) - i32::from(sig1)) % 3 {
    assert!(sig1 <= sig2, "sig1 {sig1} > sig2 {sig2}");
    if (sig2 - sig1).is_multiple_of(3) {
        // sig2 - sig1 is always 0, with the known calls below.
        (start..=end)
            .flat_map(|idx| {
                (0..=idx).map(move |k| {
                    let idx_m_k = idx - k;
                    if pi1 > idx_m_k || sig1 > k {
                        0.0
                    }
                    // pass
                    else {
                        // pi1 <= idx_m_k  &&  sig1 <= k
                        // if pi1 <= idx_m_k then pi1 <= idx_m_k + pi2, with all u16; and 0 <= idx_m_k + pi2 - pi1
                        assert!(
                            pi1 <= idx_m_k + pi2,
                            "pi1 {pi1}, pi2 {pi2}, idx {idx}, k {k},  pi1 <= idx_m_k + pi2"
                        );
                        let idx_m_k_p_pi2_m_pi1 = idx_m_k + pi2 - pi1;

                        // if sig1 <= k then sig1 <= k + sig2, with all u16; and 0 <= 3k + sig2 - sig1
                        assert!(
                            sig1 <= k + sig2,
                            "sig1 {sig1}, sig2 {sig2}, k {k},  sig1 <= k + sig2"
                        );
                        // Always true with max alpha_square=2_000, k=2_368; even for alpha_square=10_000 & k=10_801
                        // assert!(3*u32::from(k) + sig2_u32 - sig1_u32 <= u16_max,
                        //         "sig1 {sig1}, sig2 {sig2}, k {k},  3k + sig2 - sig1 <= u16::MAX");
                        let sig2_m_sig1_p_3k = sig2 - sig1 + 3 * k;

                        coeff(idx_m_k, k, t_i_state, start, end)
                            * coeff(idx_m_k_p_pi2_m_pi1, sig2_m_sig1_p_3k / 3, t_i_state, start, end)
                            * perm(idx_m_k_p_pi2_m_pi1, pi2).sqrt()
                            * perm(idx_m_k, pi1).sqrt()
                            * perm(sig2_m_sig1_p_3k, sig2).sqrt()
                            * perm(3 * k, sig1).sqrt()
                    }
                })
            })
            .sum::<f64>()
    } else {
        0.0
    }
}

/// Helper macro to extract either the min or the max value from an `Array1<f64>`
#[macro_export]
macro_rules! f64_best {
    ($array1_f64:path, $iter_min_or_max_by:path) => {
        $iter_min_or_max_by($array1_f64.iter(), |best, nxt| {
            best.partial_cmp(nxt)
                .expect("best.partial_cmp(nxt) are two f64::is_normal()")
        })
        .ok_or_else(|| eyre!("Empty {} Array1::<f64>", $array1_f64))
    };
}

/// Extracts the values from `StateEvolution`, computes the different `moments()`, and plot the results
///
/// #Parameters
///
/// - `evolution` - a `StateEvolution` computed or cached from the disk
/// - `e_alpha_square` - the `AlphaSquare` used to compute the `evolution`
/// - `delta_t` - the time step used to compute the `evolution`
/// - `verbose` - verbosity of the plot generation
///
/// ### Errors
///
/// May return :
///  - May return `Err` from the `ruviz` crate `subplots()` function and `SubplotFigure::subplot()`
///    and `SubplotFigure::save()` methods
pub fn interpret_and_plot(
    evolution: &StateEvolution,
    e_alpha_square: AlphaSquare,
    delta_t: f64,
    verbose: bool,
) -> Result<()> {
    let alpha_square = e_alpha_square as u16;
    let f_alpha_square = f64::from(alpha_square);
    let StateEvolution {
        time_ts,
        states,
        states_coeff,
    } = evolution;
    let start = states_coeff[0];
    let end = states_coeff[states_coeff.len() - 1];

    let begin = Instant::now();

    let offset = usize::from(0.0 == moments(1, 1, 0, 0, &states.slice(s![0, ..]), start, end));
    let dim = time_ts.len() - offset;
    if (false, verbose && 10 == alpha_square).0 {
        println!(
            "{start}..{end}, offset {offset}, dim {dim}, states[:10, :10] :\n{},\nstates[1250, 299:324] :\n{}\n",
            states.slice(s![0..10, 0..10]),
            states.slice(s![2500, 299..324])
        );
    }

    // println!("mmt2200 {}/ (mmt1100 {})²",moments(2,2,0,0, &states.slice(s![0, ..]), start, end),moments(1,1,0,0, &states.slice(s![0, ..]), start, end));
    // println!("mmt0022 {}/ (mmt0011 {})²",moments(0,0,2,2, &states.slice(s![0, ..]), start, end),moments(0,0,1,1, &states.slice(s![0, ..]), start, end));
    // _test(start, end);

    let mut pump_population = Array1::<f64>::zeros(dim);
    let mut signal_population = Array1::<f64>::zeros(dim);
    let mut pump_variance_momentum = Array1::<f64>::zeros(dim);
    let mut pump_g2 = Array1::<f64>::zeros(dim);
    let mut signal_g2 = Array1::<f64>::zeros(dim);
    let mut ap_vals = Array1::<f64>::zeros(dim);

    for t_i in 0..dim {
        let t_i_state = &states.slice(s![t_i + offset, ..]);
        // let mmt1000 = moments(1, 0, 0, 0, phi, start, end);
        let mmt1100 = moments(1, 1, 0, 0, t_i_state, start, end);
        let mmt0011 = moments(0, 0, 1, 1, t_i_state, start, end);
        let mmt2000 = moments(2, 0, 0, 0, t_i_state, start, end);

        pump_population[t_i] = mmt1100;
        signal_population[t_i] = mmt0011;
        pump_variance_momentum[t_i] = mmt1100 - mmt2000 + 0.5;
        pump_g2[t_i] = moments(2, 2, 0, 0, t_i_state, start, end) / (mmt1100 * mmt1100);
        signal_g2[t_i] = moments(0, 0, 2, 2, t_i_state, start, end) / (mmt0011 * mmt0011);
        ap_vals[t_i] = moments(1, 0, 0, 0, t_i_state, start, end);
    }

    let conserved_quant = 3.0 * &pump_population + &signal_population;
    let last_t = time_ts[dim - 1];
    let time_ts = &time_ts.slice(s![offset..]);

    let subplot0 = subplot0(
        time_ts,
        &pump_population,
        &signal_population,
        &conserved_quant,
        last_t,
        f_alpha_square,
    );
    let subplot1 = subplot1(time_ts, &pump_variance_momentum, last_t, e_alpha_square);
    let subplot2 = subplot2(time_ts, &pump_g2, &signal_g2, last_t, delta_t);

    subplots(1, 3, 2_000, 600)?
        .subplot(0, 0, subplot0.into())?
        .subplot(0, 1, subplot1.into())?
        .subplot(0, 2, subplot2.into())?
        .suptitle("")
        .save(format!("3rdHarmonicGenerationASqr{alpha_square}.png"))?;

    if (false, verbose).0 {
        println!(
            "pump_population ({}..{}),\nsignal_population ({}..{}),\nconserved_quantity ({}..{
                }),\npump_variance_momentum ({}..{}),\npump g² ({}..{}),\nsignal g² ({}..{})",
            f64_best!(pump_population, Iterator::min_by)?,
            f64_best!(pump_population, Iterator::max_by)?,
            f64_best!(signal_population, Iterator::min_by)?,
            f64_best!(signal_population, Iterator::max_by)?,
            f64_best!(conserved_quant, Iterator::min_by)?,
            f64_best!(conserved_quant, Iterator::max_by)?,
            f64_best!(pump_variance_momentum, Iterator::min_by)?,
            f64_best!(pump_variance_momentum, Iterator::max_by)?,
            f64_best!(pump_g2, Iterator::min_by)?,
            f64_best!(pump_g2, Iterator::max_by)?,
            f64_best!(signal_g2, Iterator::min_by)?,
            f64_best!(signal_g2, Iterator::max_by)?,
        );
    }
    println!("Moments extraction and plot finished in {:?}", begin.elapsed());
    Ok(())
}

/// Returns subplot 0, of `pump_population`, `signal_population`, `conserved_quantity`
fn subplot0(
    time_ts: &ArrayView1<f64>,
    pump_population: &Array1<f64>,
    signal_population: &Array1<f64>,
    conserved_quant: &Array1<f64>,
    last_t: f64,
    f_alpha_square: f64,
) -> PlotBuilder<LineConfig> {
    Plot::new()
        .line(time_ts, pump_population)
        .line_style(LineStyle::Solid)
        .color(Color::new(31, 119, 180))
        .label(r"$angle.l N_p angle.r$")
        .line(time_ts, signal_population)
        .line_style(LineStyle::Solid)
        .color(Color::BLACK)
        .label(r"$angle.l N_s angle.r$")
        .line(time_ts, conserved_quant)
        .line_style(LineStyle::Dashed)
        .color(Color::BLACK)
        .label(r"$3 angle.l N_p angle.r + angle.l N_s angle.r$")
        .legend(Position::Custom { x: 0.03, y: 0.85 })
        .grid(true)
        .xlabel(r"time")
        .ylabel(r"Signal population, $angle.l N_s angle.r$")
        .xlim(0.0, last_t)
        .ylim(-f_alpha_square / 20., 3.2 * f_alpha_square)
        .typst(true)
}

/// Returns subplot 1, of `pump_variance_momentum`
fn subplot1(
    time_ts: &ArrayView1<f64>,
    pump_variance_momentum: &Array1<f64>,
    last_t: f64,
    e_alpha_square: AlphaSquare,
) -> PlotBuilder<LineConfig> {
    let alpha_square = e_alpha_square as u16;
    let (min, max) = match e_alpha_square {
        AlphaSquare::U1e1 => (0.4, 0.8),
        AlphaSquare::U1e2 => (0.45, 0.55),
        AlphaSquare::U1e3 => (-50.0, 925.0),
        AlphaSquare::U2e3 => (-100.0, 2000.0),
    };

    Plot::new()
        .line(time_ts, pump_variance_momentum)
        .line_style(LineStyle::Solid)
        .color(Color::BLACK)
        .line(&vec![0.0, last_t], &vec![0.5, 0.5])
        .line_style(LineStyle::Dashed)
        .color(Color::RED)
        .title(format!("$| alpha |^2={alpha_square}$"))
        .legend_best()
        .grid(true)
        .xlabel(r"time")
        .ylabel(r"Variance, $(Delta_(P_p))^2$")
        .xlim(0.0, last_t)
        .ylim(min, max)
        .typst(true)
}

/// Returns subplot 2, of `pump_g2` and `signal_g2`
fn subplot2(
    time_ts: &ArrayView1<f64>,
    pump_g2: &Array1<f64>,
    signal_g2: &Array1<f64>,
    last_t: f64,
    delta_t: f64,
) -> PlotBuilder<LineConfig> {
    Plot::new()
        .line(time_ts, pump_g2)
        .line_style(LineStyle::Solid)
        .color(Color::new(31, 119, 180))
        .label("Pump")
        .line(time_ts, signal_g2)
        .line_style(LineStyle::Solid)
        .color(Color::new(214, 39, 40))
        .label("Signal")
        .legend_best()
        .grid(true)
        .ylabel(r"$g^2$")
        .xlabel(r"time")
        .xlim(-delta_t, last_t)
        .yscale(AxisScale::Log)
        .typst(true)
}

#[allow(clippy::too_many_lines)]
fn _test(start: u16, end: u16) {
    for (pi1, pi2, sig1, sig2) in [
        (1, 1, 0, 0),
        (0, 0, 1, 1),
        (2, 0, 0, 0),
        (2, 2, 0, 0),
        (0, 0, 2, 2),
        (1, 0, 0, 0),
    ] {
        let d_pi = i32::from(pi2) - i32::from(pi1);
        if d_pi < 0 {
            println!("pi1 {pi1} > pi2 {pi2}, d_pi {d_pi}");
        }
        assert!(sig1 <= sig2, "sig1 {sig1} > sig2 {sig2}");
        let d_sig = sig2 - sig1;
        let min_pi2_or_1 = pi2.min(1);

        if let Some(min_0) = (start..=end)
            .flat_map(|idx| {
                (0..=idx).filter_map(move |k| {
                    let idx_m_k = idx - k;
                    if pi1 > idx_m_k + min_pi2_or_1 || sig1 > k {
                        None
                    } else {
                        assert!(
                            pi1 <= idx_m_k + min_pi2_or_1,
                            "pi1 {pi1}, pi2 {pi2}, idx {idx}, k {k},  pi1 <= idx_m_k + min_pi2_or_1"
                        );
                        let idx_m_k_p_pi2_m_pi1 = idx_m_k + pi2 - pi1;
                        Some((idx_m_k_p_pi2_m_pi1, pi2, perm(idx_m_k_p_pi2_m_pi1, pi2).sqrt()))
                        // Some((idx_m_k, pi1, perm(idx_m_k, pi1).sqrt()))
                        // * perm(3*k + d_sig, sig2).sqrt()
                        // * perm(3*k, sig1).sqrt()
                    }
                })
            })
            .min_by(|min, nxt| {
                min.2
                    .partial_cmp(&nxt.2)
                    .expect("min.2.partial_cmp(&nxt.2) are two f64::is_normal()")
            })
        {
            println!(
                "{pi1}{pi2}{sig1}{sig2} min (idx_m_k_p_pi2_m_pi1 ={}, pi2={
                    }, perm(idx_m_k + d_pi, pi2).sqrt()={})",
                min_0.0, min_0.1, min_0.2
            );
        }

        if let Some(max_0) = (start..=end)
            .flat_map(|idx| {
                (0..=idx).filter_map(move |k| {
                    let idx_m_k = idx - k;
                    if pi1 > idx_m_k + min_pi2_or_1 || sig1 > k {
                        None
                    } else {
                        //  Already validated in min_0 test above.
                        let idx_m_k_p_pi2_m_pi1 = idx_m_k + pi2 - pi1;
                        Some((idx_m_k_p_pi2_m_pi1, pi2, perm(idx_m_k_p_pi2_m_pi1, pi2).sqrt()))
                        // * perm(idx_m_k, pi1).sqrt()
                        // * perm(3*k + d_sig, sig2).sqrt()
                        // * perm(3*k, sig1).sqrt()
                    }
                })
            })
            .max_by(|max, nxt| {
                max.2
                    .partial_cmp(&nxt.2)
                    .expect("max.2.partial_cmp(&nxt.2) are two f64::is_normal()")
            })
        {
            println!(
                "{pi1}{pi2}{sig1}{sig2} max (idx_m_k_p_pi2_m_pi1 ={}, pi2={}, perm(idx_m_k + d_pi, pi2).sqrt()={})",
                max_0.0, max_0.1, max_0.2
            );
        }

        if let Some(min_1) = (start..=end)
            .flat_map(|idx| {
                (0..=idx).filter_map(move |k| {
                    let idx_m_k = idx - k;
                    if pi1 > idx_m_k + min_pi2_or_1 || sig1 > k {
                        None
                    } else {
                        // Some((p1, pi2, perm(p1, pi2).sqrt()))
                        Some((idx_m_k, pi1, perm(idx_m_k, pi1).sqrt()))
                        // * perm(3*k + d_sig, sig2).sqrt()
                        // * perm(3*k, sig1).sqrt()
                    }
                })
            })
            .min_by(|min, nxt| {
                min.2
                    .partial_cmp(&nxt.2)
                    .expect("min.2.partial_cmp(&nxt.2) are two f64::is_normal()")
            })
        {
            println!(
                "{pi1}{pi2}{sig1}{sig2} min (idx_m_k ={}, pi1={}, perm(idx_m_k, pi1).sqrt()={})",
                min_1.0, min_1.1, min_1.2
            );
        }

        if let Some(max_1) = (start..=end)
            .flat_map(|idx| {
                (0..=idx).filter_map(move |k| {
                    let idx_m_k = idx - k;
                    if pi1 > idx_m_k + min_pi2_or_1 || sig1 > k {
                        None
                    } else {
                        // Some((p1, pi2, perm(p1, pi2).sqrt()))
                        Some((idx_m_k, pi1, perm(idx_m_k, pi1).sqrt()))
                        // * perm(3*k + d_sig, sig2).sqrt()
                        // * perm(3*k, sig1).sqrt()
                    }
                })
            })
            .max_by(|max, nxt| {
                max.2
                    .partial_cmp(&nxt.2)
                    .expect("max.2.partial_cmp(&nxt.2) are two f64::is_normal()")
            })
        {
            println!(
                "{pi1}{pi2}{sig1}{sig2} max (idx_m_k ={}, pi1={}, perm(idx_m_k, pi1).sqrt()={})",
                max_1.0, max_1.1, max_1.2
            );
        }

        if let Some(min_2) = (start..=end)
            .flat_map(|idx| {
                (0..=idx).filter_map(move |k| {
                    let idx_m_k = idx - k;
                    if pi1 > idx_m_k + min_pi2_or_1 || sig1 > k {
                        None
                    } else {
                        // Some((p1, pi2, perm(p1, pi2).sqrt()))
                        // Some((idx_m_k, pi1, perm(idx_m_k.into(), pi1.into()).sqrt()))
                        Some((3 * k + d_sig, sig2, perm(3 * k + d_sig, sig2).sqrt()))
                        // * perm(3*k, sig1).sqrt()
                    }
                })
            })
            .min_by(|min, nxt| {
                min.2
                    .partial_cmp(&nxt.2)
                    .expect("min.2.partial_cmp(&nxt.2) are two f64::is_normal()")
            })
        {
            println!(
                "{pi1}{pi2}{sig1}{sig2} min (3*k + d_sig ={}, sig2={}, perm(3*k + d_sig, sig2).sqrt()={})",
                min_2.0, min_2.1, min_2.2
            );
        }

        if let Some(max_2) = (start..=end)
            .flat_map(|idx| {
                (0..=idx).filter_map(move |k| {
                    let idx_m_k = idx - k;
                    if pi1 > idx_m_k + min_pi2_or_1 || sig1 > k {
                        None
                    } else {
                        // Some((p1, pi2, perm(p1, pi2).sqrt()))
                        // Some((idx_m_k, pi1, perm(idx_m_k.into(), pi1.into()).sqrt()))
                        Some((3 * k + d_sig, sig2, perm(3 * k + d_sig, sig2).sqrt()))
                        // * perm(3*k, sig1).sqrt()
                    }
                })
            })
            .max_by(|max, nxt| {
                max.2
                    .partial_cmp(&nxt.2)
                    .expect("max.2.partial_cmp(&nxt.2) are two f64::is_normal()")
            })
        {
            println!(
                "{pi1}{pi2}{sig1}{sig2} max (3*k + d_sig ={}, sig2={}, perm(3*k + d_sig, sig2).sqrt()={})",
                max_2.0, max_2.1, max_2.2
            );
        }

        if let Some(min_3) = (start..=end)
            .flat_map(|idx| {
                (0..=idx).filter_map(move |k| {
                    let idx_m_k = idx - k;
                    if pi1 > idx_m_k + min_pi2_or_1 || sig1 > k {
                        None
                    } else {
                        // Some((p1, pi2, perm(p1, pi2).sqrt()))
                        // Some((idx_m_k, pi1, perm(idx_m_k.into(), pi1.into()).sqrt()))
                        // Some((3*k + d_sig, sig2, perm((3*k + d_sig).into(), sig2.into()).sqrt()))
                        Some((3 * k, sig1, perm(3 * k, sig1).sqrt()))
                    }
                })
            })
            .min_by(|min, nxt| {
                min.2
                    .partial_cmp(&nxt.2)
                    .expect("min.2.partial_cmp(&nxt.2) are two f64::is_normal()")
            })
        {
            println!(
                "{pi1}{pi2}{sig1}{sig2} min (3*k ={}, sig1={}, perm(3*k, sig1).sqrt()={})",
                min_3.0, min_3.1, min_3.2
            );
        }

        if let Some(max_3) = (start..=end)
            .flat_map(|idx| {
                (0..=idx).filter_map(move |k| {
                    let idx_m_k = idx - k;
                    if pi1 > idx_m_k + min_pi2_or_1 || sig1 > k {
                        None
                    } else {
                        // Some((p1, pi2, perm(p1, pi2).sqrt()))
                        // Some((idx_m_k, pi1, perm(idx_m_k.into(), pi1.into()).sqrt()))
                        // Some((3*k + d_sig, sig2, perm((3*k + d_sig).into(), sig2.into()).sqrt()))
                        Some((3 * k, sig1, perm(3 * k, sig1).sqrt()))
                    }
                })
            })
            .max_by(|max, nxt| {
                max.2
                    .partial_cmp(&nxt.2)
                    .expect("max.2.partial_cmp(&nxt.2) are two f64::is_normal()")
            })
        {
            println!(
                "{pi1}{pi2}{sig1}{sig2} max (3*k ={}, sig1={}, perm(3*k, sig1).sqrt()={})",
                max_3.0, max_3.1, max_3.2
            );
        }
    }
}
