pub mod error;
pub use error::{Error, Result};

// use ndarray::parallel::prelude::*;
use ndarray::{
    Array1, 
    Array2, 
    // Axis, 
    parallel::prelude::*, s
};
use rayon::{
    iter::ParallelIterator,
};
// use rand::prelude::*;
use ruviz::prelude::*;
use scirs2::{
    sparse::{
        AsLinearOperator, linalg::expm_multiply, sparse_diags
    }, spatial::SpatialPoint, stats::distributions::Poisson
};
use std::{ iter::repeat_n, num::NonZeroUsize, thread, time::Instant };
use strum_macros::EnumIter;
use strum::IntoEnumIterator; // Allows us to use .iter()


/// Non-negative number, alpha square indicating the average number of pump photons at time = 0.
/// Also the mean and variance of the distribution
#[derive(Debug, EnumIter, Clone, Copy, PartialEq, Eq)]
pub enum AlphaSquare {
    U1e1=10,
    U1e2=100,
    U1e3=1_000,
    U2e3=2_000,
    U1e4=10_000,
    U1e5=100_000,
}
const E_ALPHA_SQUARE: AlphaSquare = AlphaSquare::U1e1; // 10
const ALPHA_SQUARE: u32 =  E_ALPHA_SQUARE as u32;

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

const fn delta_t_of_alpha_square(e_alpha_square: AlphaSquare) -> f64 {
    match e_alpha_square as u32 {
        ..=1_000 => 0.000_1,
        5_000.. => 0.000_01,
        _ => 0.000_05
    }
}

/// Time step for the evolution
/// Optimized parameters: 
/// - `DELTA_T = 0.000_1`  for `ALPHA_SQUARE <= 1_000`, 
/// - `DELTA_T = 0.000_05` for `ALPHA_SQUARE = 2_000`, 
/// - `DELTA_T = 0.000_01` for `ALPHA_SQUARE >= 5_000`
const DELTA_T: f64 = delta_t_of_alpha_square(E_ALPHA_SQUARE);

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
        AlphaSquare::U1e4 => 0.005,
        AlphaSquare::U1e5 => 0.001_6,
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

const STEP_CNT: u16 = step_cnt_of_alpha_square(E_ALPHA_SQUARE);

pub fn run( verbose: bool) -> Result<()> {
    if verbose {
        for e in AlphaSquare::iter() {
            println!("alpha_square::{e:?}={}, delta_t = {}, end_t={}, step_cnt={}", 
                e as u32, 
                delta_t_of_alpha_square(e), 
                end_t_of_alpha_square(e),
                step_cnt_of_alpha_square(e)
            );
        }
    }

    let poisson = Poisson::new(f64::from(ALPHA_SQUARE), 1.0)?;
    let dist = poisson_distribution(&poisson, ALPHA_SQUARE);
    if verbose { dbg!(&dist); }        

    // Identifying "appropriate indices" in the Poisson distribution Array1.
    let relevant_idxs = relevant_poisson_distribution_indices(&dist, ALPHA_SQUARE, Some(PB_TH));
    if verbose { dbg!(&relevant_idxs); }

    let start = Instant::now();
    let evolution = StateEvolution::new(ALPHA_SQUARE, DELTA_T, STEP_CNT, PB_TH, verbose)?;
    println!(
        "StateEvolution::new(alpha_square={ALPHA_SQUARE}) finished on {} cores in {:?}",
        evolution.available_parallelism(),
        start.elapsed()
    );

    plot(verbose)?;

    if verbose {
        println!("3rd harmonic generation is done.");
    }
    Ok(())
}

pub fn plot( verbose: bool) -> Result<()> {

    let start = Instant::now();

    Plot::new()
        // .xlim(0., x_axis_scaling)
        .xlabel("Position in x(mm)")
        // .ylim(z_axis_scaling, 0.)
        .ylabel("Depth (mm)")
        .title(format!("3rd harmonic generation: {verbose}"))
        .save("3rdHarmonicGeneration.png")?;

    if verbose {
        println!("Plot finished in {:?}", start.elapsed());
    }

    Ok(())
}

/// Returns the Poisson probability mass function (pmf) distribution over (11 x `alpha_square`) points
/// 
/// The distribution is built over `mean + 10 x variance` points
/// 
/// ### Parameters
/// 
/// - `alpha_square` - non-negative mean and variance of the distribution
#[must_use]
pub fn poisson_distribution(poisson: &Poisson<f64>, alpha_square: u32) -> Array1<f64> {
    // `n` goes from 0 to mean + (10 x variance). Since mean = variance = alpha_square:
    Array1::<f64>::range(0., 11. * f64::from(alpha_square), 1.)
        .mapv_into(|v| poisson.pmf(v))
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
pub fn relevant_poisson_distribution_indices(dist: &Array1<f64>, alpha_square: u32, pb_th: Option<f64>) -> Vec<usize> {
    let pb_th = pb_th.unwrap_or_else(|| 10.0_f64.powi(-16));
    
    if pb_th >= 1. {
        vec![alpha_square as usize]
    }
    else {
        // the indices that satisfy the threshold.
        dist.iter()
            .enumerate()
            .filter_map(|(i, v)| if *v > pb_th { Some(i) } else { None })
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
pub fn lower_diagonal_elements_of_hamiltonian_in_subspace_of(num_pump_max: usize) -> Array1<f64> {
    // enum `AlphaSquare` enforces that `alpha_square` be at max 100_000, 
    // Therefore poisson_distribution(alpha_square).len() is at max 11 x 100_000 = 1_100_000
    // Therefore, the num_pump_max indices iterated from its derived 
    // relevant_poisson_distribution_indices(&dist) are < 1_100_000 too 
    // So, num_pump_max can be cast into f64 without loss.
    #[allow(clippy::cast_precision_loss)]
    let num_pump_max_f = num_pump_max as f64;
    // - `idx_nu` - indices ranging from `0` to `num_pump_max-1`, corresponding to lower-diagonal elements
    Array1::<f64>::range(0., num_pump_max_f, 1.)
        .mapv_into(|idx_f| ((num_pump_max_f - idx_f) * 3.0_f64.mul_add(idx_f, 1.) * 
                                 3.0_f64.mul_add(idx_f, 2.) * 3.0_f64.mul_add(idx_f, 3.)).sqrt()
        )

}

pub struct StateEvolution {
    available_parallelism: usize,
    time_ts: Array1::<f64>,
    states_coeff : Array1::<usize>,
    states : Array2::<f64>,
}

impl StateEvolution {
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
    pub fn new(alpha_square: u32, delta_t: f64, step_cnt: u16, pb_th: f64, verbose: bool) -> Result<Self> {
        // Total time of the evolution, dervied from `step_cnt`, chosen to observe
        // the dynamics until the first local maximum in signal-mode population.
        let end_t = delta_t * f64::from(step_cnt);
        
        let poisson = Poisson::new(f64::from(alpha_square), 1.0)?;
        let dist = poisson_distribution(&poisson, alpha_square);
        if verbose { dbg!(&dist); }        

        // Identifying "appropriate indices" in the Poisson distribution Array1.
        let relevant_poisson_idxs = relevant_poisson_distribution_indices(&dist, alpha_square, Some(pb_th));
        // total number states |n-k>_{p} |3k>_{s} over all relevant_poisson_idxs
        let hilbert_space_dim = relevant_poisson_idxs.iter().sum::<usize>() + relevant_poisson_idxs.len();
        
        

        let (available_parallelism,chunk_size, time_ts) = {
            let step_cnt = usize::from(step_cnt);
            let available_parallelism = {
                    const NO_AVAILABLE_PARALLELILSM: NonZeroUsize = NonZeroUsize::new(1).expect("1 is not 0");
                    thread::available_parallelism()
                    .unwrap_or(NO_AVAILABLE_PARALLELILSM)
                    .get()
                };
            (available_parallelism, 
            step_cnt/available_parallelism + usize::from(! step_cnt.is_multiple_of(available_parallelism)),
            Array1::<f64>::linspace(0., end_t, step_cnt))
        };

        // time_ts
        //     // Chunking the simulations reduces the overhead of creating many arrays
        //     .axis_chunks_iter(Axis(0), chunk_size)
        //     .into_par_iter()
        //     .map(|time_ts_chunk| {

        Ok( Self {
            available_parallelism,
            time_ts,
            states_coeff : Array1::from_shape_vec(hilbert_space_dim, 
                relevant_poisson_idxs.iter()
                .flat_map(|&idx| repeat_n(idx, idx+1))
                .collect()
            )?,

            // Time values of interest for this chuck of state evolution. end_t is inclusive like in  0..=end_t, 
            // so, we chunk across 0..=step_cnt, but use non-inclusive range() instead of inclusive linspace()
            states: (0..=step_cnt)
                .into_par_iter()
                // Chunking the simulations reduces the overhead of creating many arrays
                .chunks(chunk_size)
                .map(|steps_chunk| {

                    let step_start = *steps_chunk.first().expect("always at least one step per chunck");
                    let step_end = *steps_chunk.last().expect("always at least one step per chunck");
                    let time_ts_chunk = Array1::<f64>::range(delta_t * f64::from(step_start), delta_t * f64::from(step_end), delta_t);
                    let time_ts_chunk_len = time_ts_chunk.len();
            

                    //  the `states` array is used to store the state for the local time_ts_chunk
                    let mut states_chunk = Array2::<f64>::zeros((time_ts_chunk_len, hilbert_space_dim));
                    
                    // let mut range_for_idx = 0_usize..0_usize;
                    let mut idx_range_start: usize;
                    let mut idx_range_end = 0_usize;

                    // Simulate the time volution for each "Fock state"
                    // with a significant probability in the initial state
                    for &idx in &relevant_poisson_idxs {
                        // dimension used as shape of the hamiltonian, len of intial_state, and range_for_idx width
                        // idx + 1, here, because the hamiltonian lower_diag and upper_diag both has len = idx
                        // and the offsets of the lower_diag and upper_diag in the hamiltonian are [-1, 1], 
                        // and the dimension of the hamiltonian == diag.len() + diag_offset.abs() for all diags
                        let idx_dim = idx + 1;

                        // range_for_idx determines where |psi_{idx}> is stored in the second dimension array of stats
                        // range_for_idx.start = range_for_idx.end;
                        // range_for_idx.end += idx_dim;
                        idx_range_start = idx_range_end;
                        idx_range_end += idx_dim;
                        let range_for_idx = || idx_range_start..idx_range_end;

                        let lower_diag = lower_diagonal_elements_of_hamiltonian_in_subspace_of(idx );
                        let upper_diag = - &lower_diag;
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
                        // so, `v.len()` must be the same as hamiltonian square dim passed in `a`: idx_dim == idx + 1
                        let mut initial_state = Array1::<f64>::zeros(idx_dim);
                        initial_state[0] = 1.; // each "Fock state" with a significant probability in the initial state
                        let initial_state= initial_state.as_slice().expect("initial_state is contiguous");

                        let weight_sqrt = if pb_th >= 1. { 
                            1.0     // Fock state with probability 1 for n=alp_sq initially
                        } else { 
                            dist[idx].sqrt()
                        };

                        // For more details on expm_multiply, see both:
                        // - https://docs.rs/scirs2-sparse/0.4.1/scirs2_sparse/linalg/fn.expm_multiply.html, and
                        // - https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.expm_multiply.html

                        // ndarray stores data in a contiguous block, so, instead of `collect()`-ing  an iterator of vectors into an Array2 
                        // the vectos must first be flattened, then `reshape`d as an Array2

                        states_chunk.slice_mut(s![.., range_for_idx()])
                            .assign(
                                &Array2::from_shape_vec((time_ts_chunk_len, idx_dim), {
                                    // complex `flat_map()` have had difficulties guessing the right `Iterator::size_hint()` to
                                    // pass to `::collect()` to minimize realloc. So, it's safer to pre-alloc `::with_capacity()`
                                    let mut v = Vec::with_capacity(time_ts_chunk_len * idx_dim);
                                    v.extend(time_ts_chunk
                                        .iter()
                                        .flat_map(|t| 
                                            // The returned SparseResult<Vec<F>> has len == idx_dim.
                                            expm_multiply(hamiltonian.as_linear_operator().as_ref(), initial_state, *t, None, None)
                                                .expect("hamiltonian is square and of same dimension as initial_state")
                                                .into_iter()
                                                .map(|v_elem|  v_elem * weight_sqrt)
                                        )
                                    );
                                    v
                                }).expect("(time_ts_chunk_len, idx_dim) are the right Array2 dimensions")
                            );

                    }
                    (usize::from(step_start)..usize::from(step_end), states_chunk)
                })
                .reduce(|| (0..0, Array2::<f64>::zeros((usize::from(step_cnt + 1), hilbert_space_dim))),
                    | (unused, mut states), (step_range, states_chunk)| {

                            states.slice_mut(s![step_range, ..]).assign(&states_chunk);
                            (unused, states)
                }).1,
            })
    }

    #[must_use]
    pub const fn available_parallelism(&self) -> usize {
        self.available_parallelism
    }
    #[must_use]
    pub const fn time_ts(&self) -> &Array1::<f64> {
        &self.time_ts
    }
    #[must_use]
    pub const fn states_coeff(&self) -> &Array1::<usize> {
        &self.states_coeff
    }
    #[must_use]
    pub const fn states(&self) -> &Array2::<f64> {
        &self.states
    }
}

