// pub mod error;
// pub use error::{Error, Result};
use color_eyre::eyre::{eyre, Result};

use core::{
    f64::consts::PI,
    // ops::Range,
};
// use ndarray::parallel::prelude::*;
use ndarray::{
    Array1,
    ArrayView1,
    Array2,
    Zip,
    parallel::prelude::*,
    s,
};
use ndarray_npy::{
    read_npy,
    write_npy,
};
// use num_integer::Integer;

use rayon::iter::ParallelIterator;
// use rand::prelude::*;
use ruviz::prelude::*;
use scirs2::{
    prelude::Zero,
    sparse::{AsLinearOperator, linalg::expm_multiply, sparse_diags},
    // stats::distributions::Poisson
};
use std::{iter::repeat_n, num::NonZeroUsize, thread, time::Instant};
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
const _E_ALPHA_SQUARE: AlphaSquare = AlphaSquare::U1e3;
const _ALPHA_SQUARE: u16 = _E_ALPHA_SQUARE as u16;

// non-negative real number. It is the probability threshold that is
// used for truncating the Hilbert space. It cuts off the basis
// elements whose associated probability is smaller than pb_th at
// initial time.
// use pb_th >= 1 for Fock states with photon number n = alpha_square
// Threshold that determines the range of number states that are included
// in the input coherent state: |psi_in_pump> = \sum_{n=n1}^{n2} c_{n} |n>
// where n values are chosen such that |c_{n}|^{2} > PB_TH
// Use PB_TH >= 1 for fock state with pump-mode photon number = ALP_SQ as initial condition
const PB_TH: f64 = 1e-16;

const MAX_DELTA_T: f64 = 0.000_1;
const MID_DELTA_T: f64 = MAX_DELTA_T / 2.; //   0.000_05
const MIN_DELTA_T: f64 = 0.000_01;
const fn delta_t_of_alpha_square(e_alpha_square: AlphaSquare) -> f64 {
    match e_alpha_square as u16 {
        ..=1_000 => MAX_DELTA_T,
        5_000.. => MIN_DELTA_T,
        _ => MID_DELTA_T,
    }
}
const ROUNDING_DELTA_T: f64 = 1.0 / (MAX_DELTA_T / 100.); // 1. / 0.000_001 == 1_000_000.0
trait RoundingT {
    fn round_t(self) -> Self;
}
impl RoundingT for f64 {
    fn round_t(self) -> Self {
        (self * ROUNDING_DELTA_T).round() / ROUNDING_DELTA_T
    }
}

/// Time step for the evolution
/// Optimized parameters:
/// - `DELTA_T = 0.000_1`  for `ALPHA_SQUARE <= 1_000`,
/// - `DELTA_T = 0.000_05` for `ALPHA_SQUARE = 2_000`,
/// - `DELTA_T = 0.000_01` for `ALPHA_SQUARE >= 5_000`
const _DELTA_T: f64 = delta_t_of_alpha_square(_E_ALPHA_SQUARE);

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

const fn step_cnt_of_alpha_square(e_alpha_square: AlphaSquare) -> u16 {
    //  Considering the values defined above there will be neither sign_loss nor possible_truncation
    #[allow(clippy::cast_sign_loss)]
    #[allow(clippy::cast_possible_truncation)]
    let step_cnt = (end_t_of_alpha_square(e_alpha_square) / delta_t_of_alpha_square(e_alpha_square)) as u16;
    step_cnt
}

const _STEP_CNT: u16 = step_cnt_of_alpha_square(_E_ALPHA_SQUARE);

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
    interpret_and_plot(&evolution, alpha_square, delta_t, verbose)?;

    if verbose {
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
pub fn poisson_distribution(alpha_square: u16, verbose: bool) -> Array1<f64> {
    // let poisson = Poisson::new(f64::from(alpha_square), 0.0)?;

    // `n` goes from 0 to mean + (10 x variance). Since mean = variance = alpha_square:
    let k_range = 0..(11 * alpha_square);   //  so: 0..110, 0..1_100, 0..11_000, 0..22_000
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
pub fn relevant_poisson_distribution_indices(
    dist: &[f64],
    alpha_square: u16,
    pb_th: Option<f64>,
) -> Vec<u16> {
    let pb_th = pb_th.unwrap_or_else(|| 10.0_f64.powi(-16));

    if pb_th >= 1. {
        vec![alpha_square]
    } else {
        // the indices that satisfy the threshold.
        dist.par_iter()
            .enumerate()
            .filter_map(|(i, v)| 
                    if *v > pb_th {
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

fn available_parallelism() -> usize {
    const NO_AVAILABLE_PARALLELILSM: NonZeroUsize = NonZeroUsize::new(1).expect("1 is not 0");
    thread::available_parallelism()
        .unwrap_or(NO_AVAILABLE_PARALLELILSM)
        .get()
}

fn chunk_size(step_cnt: usize) -> usize {
    let available_parallelism = available_parallelism();
    step_cnt / available_parallelism + usize::from(!step_cnt.is_multiple_of(available_parallelism))
}

pub struct StateEvolution {
    pub time_ts: Array1<f64>,
    pub states_coeff: Array1<u16>,
    pub states: Array2<f64>,
}

/// Extra length used for shape of the `hamiltonian`s and `intial_state`s widht and `idx_dim` increment,
/// because the `hamiltonian` `lower_diag` and `upper_diag` both has len = idx + 1 (to include idx) and
/// the offsets of the `lower_diag` and `upper_diag` in the `hamiltonian` are `[-1, 1]`, and the dimension
/// of the `hamiltonian` is the max of `diag.len() + diag_offset.abs()` for all diags, so : `idx + 2`.
const XTRA_LEN: u16 = 2;
const XTRA_LEN_P: usize = XTRA_LEN as usize;

impl StateEvolution {

    fn poisson(alpha_square: u16, pb_th: f64, verbose: bool) -> (Array1<f64>, Vec<u16>) {
        let dist = poisson_distribution(alpha_square, verbose);

        // Identifying "appropriate indices" in the Poisson distribution Array1.
        let relevant_poisson_idxs = relevant_poisson_distribution_indices(
            dist.as_slice().expect("poisson_distribution is contiguous"),
            alpha_square,
            Some(pb_th),
        );

        (dist, relevant_poisson_idxs)
    }

    /// ### Parameters
    ///
    /// - `alpha_square` - non-negative mean and variance of the distribution
    /// - `delta_t` -  time step of the evolution.
    /// - `step_cnt` -  number of steps in the evolution.
    /// - `pb_th` - non-negative threshold for identifying the indices whose probability is greater
    ///   than `pb_th`. Use `pb_th` >= 1 for Fock states with photon number `n` = `alpha_square`
    ///
    /// ### Errors
    ///
    /// May return :
    ///  - `scirs2::stats::error::StatsError` from `scirs2::stats::distributions::Poisson::new()`,
    ///  - `scirs2::sparse::error::SparseError` from `scirs2::sparse::sparse_diags()`
    ///
    /// ### Panics
    ///
    /// May panic :
    /// - if there are errors in the dimensioning of the Arrays used through `state_evolution()`
    pub fn new(alpha_square: u16, delta_t: f64, step_cnt: u16, pb_th: f64, verbose: bool) -> Result<Self> {
        // Total time of the evolution, dervied from `step_cnt`, chosen to observe
        // the dynamics until the first local maximum in signal-mode population.
        let end_t = (delta_t * f64::from(step_cnt)).round_t();
        if verbose {
            println!(
                "alpha_square {alpha_square}, delta_t {delta_t}, step_cnt {step_cnt
                    }, end_t {end_t}, pb_th {pb_th:e}"
            );
        }

        let (dist, relevant_poisson_idxs) = Self::poisson(alpha_square, pb_th, verbose);
        if (false, verbose).0 {
            println!("dist {dist:?}");
        }

        let relevant_poisson_idxs_len = relevant_poisson_idxs.len(); // Again, max 22_000 < u16::MAX
        if relevant_poisson_idxs_len.is_zero() {
            return Err(eyre!("No point of the poisson distribution is above pb_th = {pb_th:e}, for µ = alpha_square = {alpha_square}"));
        }

        // total number states |n-k>_{p} |3k>_{s} over all relevant_poisson_idxs
        let hilbert_space_dim = relevant_poisson_idxs_len * XTRA_LEN_P +
            relevant_poisson_idxs.iter().map(|&idx| usize::from(idx)).sum::<usize>();

        let largest_relevant_poisson_idx = relevant_poisson_idxs[relevant_poisson_idxs_len - 1];
        if verbose {
            println!(
                "{relevant_poisson_idxs_len} relevant_poisson_idxs: ({}..{}), hilbert_space_dim {}, ",
                relevant_poisson_idxs[0], largest_relevant_poisson_idx, hilbert_space_dim
            );
        }

        let step_cnt = usize::from(step_cnt);
        let time_ts = Array1::<f64>::linspace(0., end_t, step_cnt + 1).mapv_into(f64::round_t);
        // Chunking the simulations reduces the overhead of creating many arrays
        // time_ts.axis_chunks_iter(Axis(0), chunk_size)

        if verbose {
            println!("(unused) chunk_size {}, {} time_ts {time_ts}", chunk_size(step_cnt), time_ts.len(),);
        }

        let states_filename = format!("assets/3rdHarmGen_a_sqr{alpha_square}_v0.npy");

        Ok(Self {
            states_coeff: Array1::from_shape_vec(
                hilbert_space_dim,
                relevant_poisson_idxs
                    .iter()
                    .flat_map(|&idx| repeat_n(idx, usize::from(idx + XTRA_LEN)))
                    .collect(),
            )?,

            states: read_npy(&states_filename).unwrap_or_else(|_| {
                // Compute it ONCE, reuse it relevant_poisson_idxs_len times, with increasingly larger views.
                // largest_relevant_poisson_idx + 1, because we want the largest_relevant_poisson_idx to be included
                let largest_lower_diag =
                    lower_diagonal_of_hamiltonian_in_subspace_of(largest_relevant_poisson_idx + 1);
                let largest_upper_diag = -&largest_lower_diag;

                // Same thing: allocate it ONCE, reuse it relevant_poisson_idxs_len times, with increasingly larger views.
                // Each "Fock state" with a significant probability in the initial state
                // passed as `v` in  `expm_multiply(a, v)` bellow, with result vector y = exp(t*A) * v,
                // so, `v.len()` must be the same as hamiltonian square dim passed in `a`: idx_dim == idx + 2
                let mut largest_initial_state = Array1::<f64>::zeros(usize::from(largest_relevant_poisson_idx + XTRA_LEN));
                largest_initial_state[0] = 1.0; // each "Fock state" with a significant probability in the initial state

                let mut states = Array2::<f64>::zeros((time_ts.dim(), hilbert_space_dim));

                Zip::from(states.rows_mut())
                .and(&time_ts)
                .par_for_each( |mut state_for_time_t, &time_t| {

                    // let mut range_for_idx = 0_usize..0_usize;
                    let mut idx_range_start: usize;
                    let mut idx_range_end = 0_usize;

                    // Simulate the time volution for each "Fock state"
                    // with a significant probability in the initial state
                    for &idx in &relevant_poisson_idxs {
                        let idx = usize::from(idx);
                        // dimension used as shape of the hamiltonian, len of intial_state, and range_for_idx width
                        // because the hamiltonian lower_diag and upper_diag both has len = idx + 1 (to include idx)
                        // and the offsets of the lower_diag and upper_diag in the hamiltonian are [-1, 1], and the
                        // dimension of the hamiltonian == diag.len() + diag_offset.abs() for all diags, so idx + 2
                        let idx_dim = idx + XTRA_LEN_P;

                        // range_for_idx determines where |psi_{idx}> is stored in the second dimension array of stats
                        // range_for_idx.start = range_for_idx.end;
                        // range_for_idx.end += idx_dim;
                        idx_range_start = idx_range_end;
                        idx_range_end += idx_dim;
                        let range_for_idx = || idx_range_start..idx_range_end;

                        // let lower_diag = lower_diagonal_elements_of_hamiltonian_in_subspace_of(idx );
                        let lower_diag = largest_lower_diag.slice(s![0..=idx]);
                        let upper_diag = largest_upper_diag.slice(s![0..=idx]);
                        let lower_diag = lower_diag.as_slice().expect("lower_diag is contiguous");
                        let upper_diag = upper_diag.as_slice().expect("upper_diag is contiguous");

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

                        // each "Fock state" with a significant probability in the initial state
                        // passed as `v` in  `expm_multiply(a, v)` bellow, with result vector y = exp(t*A) * v, 
                        // so, `v.len()` must be the same as hamiltonian square dim passed in `a`: idx_dim == idx + 2
                        let initial_state= largest_initial_state.slice(s![0..idx_dim]);
                        let initial_state= initial_state.as_slice().expect("initial_state is contiguous");

                        if (false, verbose).0 { println!("time_t {time_t}, idx_dim {idx_dim}, range_for_idx() {:?
                        }, initial_state {initial_state:?}, lower_diag {lower_diag:?}, upper_diag {upper_diag:?
                        }, hamiltonian {hamiltonian:?}", range_for_idx()); }

                        // For more details on expm_multiply, see both:
                        // - https://docs.rs/scirs2-sparse/0.4.1/scirs2_sparse/linalg/fn.expm_multiply.html, and
                        // - https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.expm_multiply.html

                        // The returned SparseResult<Vec<F>> has len == idx_dim.
                        let expm = expm_multiply(
                                hamiltonian.as_linear_operator().as_ref(),
                                initial_state, time_t, None, None)
                            .expect("hamiltonian is square and of same dimension as initial_state");
                        if (false, verbose).0 { println!("expm = {expm:?}"); }

                        let weight_sqrt = if pb_th >= 1. {
                            1.0     // Fock state with probability 1 for n=alp_sq initially
                        } else {
                            dist[idx].sqrt()
                        };

                        state_for_time_t.slice_mut(s![range_for_idx()])
                            .assign(&Array1::from_shape_vec(
                                idx_dim,
                                expm.into_iter()
                                    .map(|v_elem|  v_elem * weight_sqrt)
                                    .collect()
                                ).expect("idx_dim is the right Array1 dimensions")
                            );
                        if (false, verbose && 3 == idx_range_end).1 { println!("state_for_time_t {state_for_time_t}"); }
                    }

                });
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

fn coeff(pump_photon_cnt: u16, signal_photon_cnt_third: u16, psi: &ArrayView1<f64>, start: u16, end: u16) -> f64 {
    let excitations = pump_photon_cnt + signal_photon_cnt_third;
    if start <= excitations && excitations <= end {
        // #[allow(clippy::cast_sign_loss)]
        // #[allow(clippy::cast_possible_truncation)]
        // let excitations_u = excitations as u16;

        // accounts for state vector truncation n1<= n<=n2
        // exc_ind is the number of elements in the state vector before "excitations"
        // excitations (pump+(signal/3) photons) given the way the state coefficients are arranged:
        let exc_ind = usize::from(excitations - start) * usize::from(start + excitations + 1) / 2;
        psi[exc_ind + usize::from(signal_photon_cnt_third)]
        // let exc_ind = (excitations_u - start) * (start + excitations_u + 2) / 2;
        // let i = i32::from(exc_ind) + signal_photon_cnt_third;

        // if i < 0 {
        //     println!("exc_ind + signal_photon_cnt_third = {i} < 0");
        //     0.0
        // }
        // else {
        //     #[allow(clippy::cast_sign_loss)]
        //     #[allow(clippy::cast_possible_truncation)]
        //     psi[i as usize]
        // }
    }
    else {
        0.0 // to account for other unphysical cases  
    }
}

///  returns the value of P(num, k) = num! / (num - k)! = num * (num - 1) * ... * (num - k + 1)
/// 
fn perm(num: u16, k: u16) -> f64 {
// fn perm(num: i32, k: i32) -> f64 {
    if k > num + 1 {
        panic!("perm(num: {num}, k : {k}) :  k > num + 1");
    }
    else {
        ((num + 1 - k)..=num).map(f64::from).product::<f64>()
    }
}

fn moments(pi1: u16, pi2: u16, sig1: u16, sig2: u16, phi: &ArrayView1<f64>, start: u16, end: u16) -> f64 {
// fn moments(pi1: i32, pi2: i32, sig1: i32, sig2: i32, phi: &ArrayView1<f64>, start: u16, end: u16) -> f64 {
// fn moments(pi1: u16, pi2: u16, sig1: u16, sig2: u16, phi: &ArrayView1<f64>, start: u16, end: u16) -> f64 {
    let min_pi2_or_1 = pi2.min(1);
    assert!(sig1 <= sig2, "sig1 {sig1} > sig2 {sig2}");

    let d_sig = sig2 - sig1;
    if d_sig.is_multiple_of(3) {
    // if 0 == d_sig %3 {
        // (i32::from(start)..=i32::from(end)).flat_map(|idx| 
        (start..=end).flat_map(|idx| 
            (0..=idx).map(move |k| {
                let idx_m_k = idx - k;
                if pi1 > idx_m_k + min_pi2_or_1  ||  sig1 > k { 0.0 }
                else{
                    assert!(pi1 <= idx_m_k + min_pi2_or_1, "pi1 {pi1}, pi2 {pi2}, idx {idx}, k {k},  pi1 <= idx_m_k + pi2" );
                    let idx_m_k_p_pi2_m_pi1 = idx_m_k + pi2 - pi1;
                    coeff(idx_m_k, k, phi, start, end)
                        * coeff(idx_m_k_p_pi2_m_pi1, k + d_sig/3, phi, start, end)
                        * perm(idx_m_k_p_pi2_m_pi1, pi2).sqrt()
                        * perm(idx_m_k, pi1).sqrt()
                        * perm(3*k + d_sig, sig2).sqrt()
                        * perm(3*k, sig1).sqrt()
                }
            }))
            .sum::<f64>()
    } else {
        0.0
    }

}

#[macro_export]
macro_rules! f64_best {
    ($array1_f64:path, $iter_min_or_max_by:path) => (
        $iter_min_or_max_by($array1_f64.iter(), |best, nxt| 
            best.partial_cmp(nxt)
                .expect("best.partial_cmp(nxt) are two f64::is_normal()"))
        .ok_or_else(||eyre!("Empty {} Array1::<f64>", $array1_f64))
    )
}

fn interpret_and_plot(evolution: &StateEvolution, alpha_square: u16, delta_t: f64, verbose: bool) -> Result<()> {
    let StateEvolution { time_ts, states, states_coeff } = evolution;
    let start = states_coeff[0];
    let end = states_coeff[states_coeff.len()-1];
    let offset = usize::from(0.0 == moments(1,1,0,0, &states.slice(s![0, ..]), start, end));
    let dim = time_ts.len() - offset;

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
        let phi = &states.slice(s![t_i + offset, ..]);
        // let mmt1000 = moments(1, 0, 0, 0, phi, start, end);
        let mmt1100 = moments(1, 1, 0, 0, phi, start, end);
        let mmt0011 = moments(0, 0, 1, 1, phi, start, end);
        let mmt2000 = moments(2, 0, 0, 0, phi, start, end);

        pump_population[t_i] = mmt1100;
        signal_population[t_i] = mmt0011;
        pump_variance_momentum[t_i] = mmt1100 - mmt2000 + 0.5;
        pump_g2[t_i] = moments(2, 2, 0, 0, phi, start, end) / (mmt1100 * mmt1100);
        signal_g2[t_i] = moments(0, 0, 2, 2, phi, start, end) / (mmt0011 * mmt0011);
        ap_vals[t_i] = moments(1, 0, 0, 0, phi, start, end);
    }

    let conserved_quantity = 3.0*&pump_population + &signal_population;
    let last_t = time_ts[dim - 1];
    let time_ts = &time_ts.slice(s![offset..]);

    let start = Instant::now();

    let plot00 = Plot::new()
        .line(time_ts, &pump_population)
            .line_style(LineStyle::Solid)
            .color(Color::new(31, 119, 180))
            .label(r"$\langle N_{p}\rangle$")
        .line(time_ts, &signal_population)
            .line_style(LineStyle::Solid)
            .color(Color::BLACK)
            .label(r"$\langle N_{s}\rangle$")
        .line(time_ts, &conserved_quantity)
            .line_style(LineStyle::Dashed)
            .color(Color::BLACK)
            .label(r"$3\langle N_{p}\rangle+\langle N_{s}\rangle$")
        .legend_best()
        .grid(true)
        .xlabel(r"time")
        .ylabel(r"Signal population, $\langle N_{s}\rangle$")
        .xlim(0.0, last_t)
        .ylim(0.0, 3.3*f64::from(alpha_square))
        .typst(true);

    let plot01 = Plot::new()
        .line(time_ts, &pump_variance_momentum)
            .line_style(LineStyle::Solid)
            .color(Color::BLACK)
        .line(&vec![0.0, last_t], &vec![0.5, 0.5])
            .line_style(LineStyle::Dashed)
            .color(Color::RED)
        .title(format!("$| \\a |^2={alpha_square}$"))    
        .legend_best()
        .grid(true)
        .xlabel(r"time")
        .ylabel(r"Variance, $( \D {P_{p}})^2$")
        .xlim(0.0, last_t)
        .ylim(0.35, 0.75)
        .typst(true);

    let plot02 = Plot::new()
        .line(time_ts, &pump_g2)
            .line_style(LineStyle::Solid)
            .color(Color::new(31, 119, 180))
            .label("Pump")
        .line(time_ts, &signal_g2)
            .line_style(LineStyle::Solid)
            .color(Color::new(214, 39, 40))
            .label("Signal")
        .legend_best()
        .grid(true)
        .ylabel(r"$g^2$")
        .xlabel(r"time")
        .xlim(-delta_t, last_t)
        .yscale(AxisScale::Log)
        // .ylim(1.0, 2.0)
        .typst(true);

    subplots(1, 3, 2_000, 600)?
        .subplot(0, 0, plot00.into())?
        .subplot(0, 1, plot01.into())?
        .subplot(0, 2, plot02.into())?
        .suptitle("")
        .save(format!("3rdHarmonicGenerationASqr{alpha_square}.png"))?;

    if verbose {
        println!("pump_population ({}..{}),\nsignal_population ({}..{}),\nconserved_quantity ({}..{
                }),\npump_variance_momentum ({}..{}),\npump g² ({}..{}),\nsignal g² ({}..{})",
            f64_best!(pump_population, Iterator::min_by)?, f64_best!(pump_population, Iterator::max_by)?,
            f64_best!(signal_population, Iterator::min_by)?, f64_best!(signal_population, Iterator::max_by)?,
            f64_best!(conserved_quantity, Iterator::min_by)?,
            f64_best!(conserved_quantity, Iterator::max_by)?,
            f64_best!(pump_variance_momentum, Iterator::min_by)?,
            f64_best!(pump_variance_momentum, Iterator::max_by)?,
            f64_best!(pump_g2, Iterator::min_by)?, f64_best!(pump_g2, Iterator::max_by)?,
            f64_best!(signal_g2, Iterator::min_by)?, f64_best!(signal_g2, Iterator::max_by)?,
        );
        println!("Plot finished in {:?}", start.elapsed());
    }
    Ok(())
}


#[allow(clippy::too_many_lines)]
fn _test(start: u16, end: u16) {
    for (pi1, pi2, sig1, sig2) in [(1,1,0,0),(0,0,1,1),(2,0,0,0),(2,2,0,0),(0,0,2,2),(1,0,0,0)] {
        let d_pi = i32::from(pi2) - i32::from(pi1);
        if d_pi < 0 {
            println!("pi1 {pi1} > pi2 {pi2}, d_pi {d_pi}");
        }
        assert!(sig1 <= sig2, "sig1 {sig1} > sig2 {sig2}");
        let d_sig = sig2 - sig1;
        let min_pi2_or_1 = pi2.min(1);

        if let Some(min_0) = (start..=end).flat_map(|idx| 
                (0..=idx).filter_map(move |k| {
                        let idx_m_k = idx - k;
                        if pi1 > idx_m_k + min_pi2_or_1 ||  sig1 > k { None }
                        else {
                            assert!(pi1 <= idx_m_k + min_pi2_or_1, "pi1 {pi1}, pi2 {pi2}, idx {idx}, k {k},  pi1 <= idx_m_k + min_pi2_or_1" );
                            let idx_m_k_p_pi2_m_pi1 = idx_m_k + pi2 - pi1;
                            Some((idx_m_k_p_pi2_m_pi1, pi2, perm(idx_m_k_p_pi2_m_pi1, pi2).sqrt()))
                            // Some((idx_m_k, pi1, perm(idx_m_k, pi1).sqrt()))
                            // * perm(3*k + d_sig, sig2).sqrt()
                            // * perm(3*k, sig1).sqrt()
                        }
                    }))
                .min_by(|min, nxt| 
                    min.2.partial_cmp(&nxt.2)
                        .expect("min.2.partial_cmp(&nxt.2) are two f64::is_normal()")) {
            println!("{pi1}{pi2}{sig1}{sig2} min (idx_m_k_p_pi2_m_pi1 ={}, pi2={
                    }, perm(idx_m_k + d_pi, pi2).sqrt()={})", min_0.0, min_0.1, min_0.2);
        }

        if let Some(max_0) = (start..=end).flat_map(|idx| 
                (0..=idx).filter_map(move |k| {
                    let idx_m_k = idx - k;
                    if pi1 > idx_m_k + min_pi2_or_1  ||  sig1 > k { None }
                    else {
                        //  Already validated in min_0 test above.
                        let idx_m_k_p_pi2_m_pi1 = idx_m_k + pi2 - pi1;
                        Some((idx_m_k_p_pi2_m_pi1, pi2, perm(idx_m_k_p_pi2_m_pi1, pi2).sqrt()))
                        // * perm(idx_m_k, pi1).sqrt()
                        // * perm(3*k + d_sig, sig2).sqrt()
                        // * perm(3*k, sig1).sqrt()
                    }
                }))
                .max_by(|max, nxt| max.2.partial_cmp(&nxt.2).expect("max.2.partial_cmp(&nxt.2) are two f64::is_normal()")) {
            println!("{pi1}{pi2}{sig1}{sig2} max (idx_m_k_p_pi2_m_pi1 ={}, pi2={}, perm(idx_m_k + d_pi, pi2).sqrt()={})", max_0.0, max_0.1, max_0.2);
        }

        if let Some(min_1) = (start..=end).flat_map(|idx| 
                (0..=idx).filter_map(move |k| {
                    let idx_m_k = idx - k;
                    if pi1 > idx_m_k + min_pi2_or_1  ||  sig1 > k { None }
                    else {
                        // Some((p1, pi2, perm(p1, pi2).sqrt()))
                        Some((idx_m_k, pi1, perm(idx_m_k, pi1).sqrt()))
                        // * perm(3*k + d_sig, sig2).sqrt()
                        // * perm(3*k, sig1).sqrt()
                    }
                }))
                .min_by(|min, nxt| min.2.partial_cmp(&nxt.2).expect("min.2.partial_cmp(&nxt.2) are two f64::is_normal()")) {
            println!("{pi1}{pi2}{sig1}{sig2} min (idx_m_k ={}, pi1={}, perm(idx_m_k, pi1).sqrt()={})", min_1.0, min_1.1, min_1.2);
        }

        if let Some(max_1) = (start..=end).flat_map(|idx| 
                (0..=idx).filter_map(move |k| {
                    let idx_m_k = idx - k;
                    if pi1 > idx_m_k + min_pi2_or_1  ||  sig1 > k { None }
                    else {
                        // Some((p1, pi2, perm(p1, pi2).sqrt()))
                        Some((idx_m_k, pi1, perm(idx_m_k, pi1).sqrt()))
                        // * perm(3*k + d_sig, sig2).sqrt()
                        // * perm(3*k, sig1).sqrt()
                    }
                }))
                .max_by(|max, nxt| max.2.partial_cmp(&nxt.2).expect("max.2.partial_cmp(&nxt.2) are two f64::is_normal()")) {
            println!("{pi1}{pi2}{sig1}{sig2} max (idx_m_k ={}, pi1={}, perm(idx_m_k, pi1).sqrt()={})", max_1.0, max_1.1, max_1.2);
        }

        if let Some(min_2) = (start..=end).flat_map(|idx| 
                (0..=idx).filter_map(move |k| {
                    let idx_m_k = idx - k;
                    if pi1 > idx_m_k + min_pi2_or_1  ||  sig1 > k { None }
                    else {
                        // Some((p1, pi2, perm(p1, pi2).sqrt()))
                        // Some((idx_m_k, pi1, perm(idx_m_k.into(), pi1.into()).sqrt()))
                        Some((3*k + d_sig, sig2, perm(3*k + d_sig, sig2).sqrt()))
                        // * perm(3*k, sig1).sqrt()
                    }
                }))
                .min_by(|min, nxt| min.2.partial_cmp(&nxt.2).expect("min.2.partial_cmp(&nxt.2) are two f64::is_normal()")) {
            println!("{pi1}{pi2}{sig1}{sig2} min (3*k + d_sig ={}, sig2={}, perm(3*k + d_sig, sig2).sqrt()={})", min_2.0, min_2.1, min_2.2);
        }

        if let Some(max_2) = (start..=end).flat_map(|idx| 
                (0..=idx).filter_map(move |k| {
                    let idx_m_k = idx - k;
                    if pi1 > idx_m_k + min_pi2_or_1  ||  sig1 > k { None }
                    else {
                        // Some((p1, pi2, perm(p1, pi2).sqrt()))
                        // Some((idx_m_k, pi1, perm(idx_m_k.into(), pi1.into()).sqrt()))
                        Some((3*k + d_sig, sig2, perm(3*k + d_sig, sig2).sqrt()))
                        // * perm(3*k, sig1).sqrt()
                    }
                }))
                .max_by(|max, nxt| max.2.partial_cmp(&nxt.2).expect("max.2.partial_cmp(&nxt.2) are two f64::is_normal()")) {
            println!("{pi1}{pi2}{sig1}{sig2} max (3*k + d_sig ={}, sig2={}, perm(3*k + d_sig, sig2).sqrt()={})", max_2.0, max_2.1, max_2.2);
        }

        if let Some(min_3) = (start..=end).flat_map(|idx| 
                (0..=idx).filter_map(move |k| {
                    let idx_m_k = idx - k;
                    if pi1 > idx_m_k + min_pi2_or_1  ||  sig1 > k { None }
                    else {
                        // Some((p1, pi2, perm(p1, pi2).sqrt()))
                        // Some((idx_m_k, pi1, perm(idx_m_k.into(), pi1.into()).sqrt()))
                        // Some((3*k + d_sig, sig2, perm((3*k + d_sig).into(), sig2.into()).sqrt()))
                        Some((3*k, sig1, perm(3*k, sig1).sqrt()))
                    }
                }))
                .min_by(|min, nxt| min.2.partial_cmp(&nxt.2).expect("min.2.partial_cmp(&nxt.2) are two f64::is_normal()")) {
            println!("{pi1}{pi2}{sig1}{sig2} min (3*k ={}, sig1={}, perm(3*k, sig1).sqrt()={})", min_3.0, min_3.1, min_3.2);
        }

        if let Some(max_3) = (start..=end).flat_map(|idx| 
                (0..=idx).filter_map(move |k| {
                    let idx_m_k = idx - k;
                    if pi1 > idx_m_k + min_pi2_or_1  ||  sig1 > k { None }
                    else {
                        // Some((p1, pi2, perm(p1, pi2).sqrt()))
                        // Some((idx_m_k, pi1, perm(idx_m_k.into(), pi1.into()).sqrt()))
                        // Some((3*k + d_sig, sig2, perm((3*k + d_sig).into(), sig2.into()).sqrt()))
                        Some((3*k, sig1, perm(3*k, sig1).sqrt()))
                    }
                }))
                .max_by(|max, nxt| max.2.partial_cmp(&nxt.2).expect("max.2.partial_cmp(&nxt.2) are two f64::is_normal()")) {
            println!("{pi1}{pi2}{sig1}{sig2} max (3*k ={}, sig1={}, perm(3*k, sig1).sqrt()={})", max_3.0, max_3.1, max_3.2);
        }
    }

}

