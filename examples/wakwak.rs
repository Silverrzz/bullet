use bullet_lib::{
    game::formats::{bulletformat::ChessBoard, wakformat::bullet::DUCK_EXTRA_INDEX},
    game::inputs::{ChessBucketsMirrored, SparseInputType},
    nn::optimiser::{AdamW, AdamWParams},
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader::WakFormatLoader},
};

const L1_SIZE: usize = 768;
const L2_SIZE: usize = 16;
const L3_SIZE: usize = 32;
const SCALE: f32 = 300.0;
const Q0: i16 = 255;
const Q1: i16 = 128;
const Q: i16 = 64;
const FT_SHIFT: usize = 8;
const FT_SHIFT_SCALE: f32 = Q0 as f32 / ((1 << FT_SHIFT) as f32);
const I8_RANGE: f32 = i8::MAX as f32 / Q1 as f32;
const L1_RANGE: f32 = I8_RANGE * FT_SHIFT_SCALE * FT_SHIFT_SCALE;

#[derive(Clone, Copy, Default)]
struct DuckInputs;

impl SparseInputType for DuckInputs {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        832
    }

    fn max_active(&self) -> usize {
        33
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &ChessBoard, mut f: F) {
        ChessBucketsMirrored::default().map_features(pos, &mut f);
        let square = usize::from(pos.extra()[DUCK_EXTRA_INDEX]);
        if square < 64 {
            let stm_flip = if pos.our_ksq() % 8 > 3 { 7 } else { 0 };
            let ntm_flip = if pos.opp_ksq() % 8 > 3 { 7 } else { 0 };
            f(768 + (square ^ stm_flip), 768 + (square ^ 56 ^ ntm_flip));
        }
    }

    fn shorthand(&self) -> String {
        "832hm".to_string()
    }

    fn description(&self) -> String {
        "Horizontally mirrored chess and duck inputs".to_string()
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut paths = Vec::new();
    let mut net_id = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--name" => {
                net_id = Some(
                    args.next()
                        .filter(|name| !name.is_empty() && !name.starts_with("--"))
                        .expect("--name requires a network name"),
                )
            }
            _ if arg.starts_with("--") => panic!("Unknown option: {arg}"),
            _ => paths.push(arg),
        }
    }
    assert!(!paths.is_empty(), "Usage: wakwak [--name NAME] <data.wf> [more-data.wf ...]");
    let paths: Vec<&str> = paths.iter().map(String::as_str).collect();
    let inputs = ChessBucketsMirrored::default();
    let loader = WakFormatLoader::new_concat_multiple(&paths, 1024, 4, |_, mv, _, _| !mv.flag().is_noisy());
    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(inputs)
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(Q0),
            SavedFormat::id("l0b").round().quantise::<i16>(Q0),
            SavedFormat::id("l1w")
                .transform(|_, weights| {
                    let mut permuted = vec![0.0; weights.len()];
                    for i1 in (0..L1_SIZE).step_by(4) {
                        for o in 0..L2_SIZE {
                            for i2 in 0..4 {
                                permuted[i1 * L2_SIZE + o * 4 + i2] = weights[(i1 + i2) * L2_SIZE + o];
                            }
                        }
                    }
                    for weight in &mut permuted {
                        *weight /= FT_SHIFT_SCALE * FT_SHIFT_SCALE;
                    }
                    permuted
                })
                .round()
                .quantise::<i8>(Q1),
            SavedFormat::id("l1b").round().quantise::<i32>(i32::from(Q) * 256),
            SavedFormat::id("l2w").round().quantise::<i32>(i32::from(Q)),
            SavedFormat::id("l2b").round().quantise::<i32>(i32::from(Q).pow(3)),
            SavedFormat::id("l3w").round().quantise::<i32>(i32::from(Q)),
            SavedFormat::id("l3b").round().quantise::<i32>(i32::from(Q).pow(4)),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs| {
            let l0 = builder.new_affine("l0", inputs.num_inputs(), L1_SIZE);
            let l1 = builder.new_affine("l1", L1_SIZE, L2_SIZE);
            let l2 = builder.new_affine("l2", L2_SIZE * 2, L3_SIZE);
            let l3 = builder.new_affine("l3", L3_SIZE, 1);

            let ft = |input, start, end| l0.slice(start, end).forward(input).crelu();
            let stm_hidden = ft(stm_inputs, 0, L1_SIZE / 2) * ft(stm_inputs, L1_SIZE / 2, L1_SIZE);
            let ntm_hidden = ft(ntm_inputs, 0, L1_SIZE / 2) * ft(ntm_inputs, L1_SIZE / 2, L1_SIZE);

            let l1_out = l1.forward(stm_hidden.concat(ntm_hidden));
            let l1_out = l1_out.concat(l1_out.abs_pow(2.0)).crelu();

            let l2_out = l2.forward(l1_out);
            let l2_out = l2_out.crelu();

            let l3_out = l3.forward(l2_out);

            l3_out
        });

    let l1_clip = AdamWParams { max_weight: L1_RANGE, min_weight: -L1_RANGE, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l1w", l1_clip);

    let sbs = 240;

    let schedule = TrainingSchedule {
        net_id: net_id.unwrap_or_else(|| "shokupan".to_string()),
        eval_scale: SCALE,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: sbs,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.3 },
        lr_scheduler: lr::Sequence {
            first: lr::ConstantLR { value: 0.001 },
            second: lr::LinearDecayLR { initial_lr: 0.001, final_lr: 0.000025, final_superbatch: sbs - 1 },
            first_scheduler_final_superbatch: 1,
        },
        save_rate: 40,
    };
    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 64 };
    trainer.run(&schedule, &settings, &loader);
}
