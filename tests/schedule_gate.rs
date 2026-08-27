//! The hard gate's training/inference asymmetry, and the refresh schedule.

use anaphora::schedule::{
    NoiseLevel, Phase, RefreshSchedule, RetrievalBand, retrieve_now, trajectory,
};

fn t(v: f32) -> NoiseLevel {
    NoiseLevel::new(v).expect("in range")
}

#[test]
fn low_noise_band_is_training_only() {
    // The asymmetry the design sketch turns on. Near t=0 the training query
    // is nearly clean, so a neighbour's continuation can hold the few
    // remaining answers and the low-t loss goes trivial. At inference there
    // is nothing to leak and this is where retrieval helps most.
    assert!(!retrieve_now(t(0.05), Phase::Training));
    assert!(retrieve_now(t(0.05), Phase::Inference));
}

#[test]
fn high_noise_band_is_closed_in_both_phases() {
    // Nearly all [MASK]: the query embedding is noise in either phase.
    assert!(!retrieve_now(t(0.95), Phase::Training));
    assert!(!retrieve_now(t(0.95), Phase::Inference));
}

#[test]
fn mid_band_is_open_in_both_phases() {
    for v in [0.2, 0.5, 0.8] {
        assert!(retrieve_now(t(v), Phase::Training), "training at t={v}");
        assert!(retrieve_now(t(v), Phase::Inference), "inference at t={v}");
    }
}

#[test]
fn band_edges_are_exclusive() {
    let band = RetrievalBand::DEFAULT_TRAINING;
    assert!(!band.admits(t(0.15)));
    assert!(!band.admits(t(0.85)));
    assert!(band.admits(t(0.150001)));
}

#[test]
fn noise_level_rejects_out_of_range() {
    assert!(NoiseLevel::new(-0.01).is_none());
    assert!(NoiseLevel::new(1.01).is_none());
    assert!(NoiseLevel::new(f32::NAN).is_none());
    // A broken schedule should skip retrieval, not retrieve on garbage.
    assert_eq!(NoiseLevel::saturating(f32::NAN), NoiseLevel::MASKED);
}

#[test]
fn first_step_always_refreshes() {
    // Nothing is cached at the start, so a trajectory whose first t sits
    // below every threshold must still retrieve once rather than run the
    // whole way with no neighbours.
    let mut schedule = RefreshSchedule::new(&[0.8]);
    assert!(schedule.advance(t(0.2)));
}

#[test]
fn thresholds_fire_once_each_on_descent() {
    let mut schedule = RefreshSchedule::new(&[0.8, 0.5, 0.25]);
    let fired: Vec<bool> = [1.0, 0.9, 0.8, 0.7, 0.5, 0.4, 0.25, 0.1]
        .iter()
        .map(|&v| schedule.advance(t(v)))
        .collect();
    // t=1.0 is the first step; 0.8, 0.5 and 0.25 are the crossings.
    assert_eq!(
        fired,
        vec![true, false, true, false, true, false, true, false]
    );
    assert!(schedule.remaining().is_empty());
}

#[test]
fn coarse_steps_collapse_skipped_thresholds() {
    // A refresh recomputes from the current t, so crossing three thresholds
    // in one step is one refresh, not three.
    let mut schedule = RefreshSchedule::new(&[0.8, 0.5, 0.25]);
    assert!(schedule.advance(t(1.0)));
    assert!(schedule.advance(t(0.1)));
    assert!(!schedule.advance(t(0.05)));
}

#[test]
fn reset_rewinds_for_the_next_sample() {
    let mut schedule = RefreshSchedule::new(&[0.5]);
    schedule.advance(t(1.0));
    schedule.advance(t(0.4));
    assert!(schedule.remaining().is_empty());
    schedule.reset();
    assert_eq!(schedule.remaining().len(), 1);
}

#[test]
fn trajectory_descends_from_masked_to_clean() {
    let steps = trajectory(5);
    assert_eq!(steps.len(), 5);
    assert_eq!(steps[0], NoiseLevel::MASKED);
    assert_eq!(steps[4], NoiseLevel::CLEAN);
    for pair in steps.windows(2) {
        assert!(pair[0].get() > pair[1].get());
    }
    assert!(trajectory(0).is_empty());
    assert_eq!(trajectory(1), vec![NoiseLevel::MASKED]);
}
