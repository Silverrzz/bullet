use bullet_lib::{
    game::inputs::{ChessBucketsMirrored, SparseInputType},
    nn::optimiser::AdamW,
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader::WakFormatLoader},
};

const HIDDEN_SIZE: usize = 512;
const SCALE: f32 = 400.0;
const QA: i16 = 255;
const QB: i16 = 64;

fn main() {
    let mut args = std::env::args().skip(1);
    let mut paths = Vec::new();
    let mut net_id = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--name" => net_id = Some(args.next().filter(|name| !name.is_empty() && !name.starts_with("--"))
                .expect("--name requires a network name")),
            _ if arg.starts_with("--") => panic!("Unknown option: {arg}"),
            _ => paths.push(arg),
        }
    }
    assert!(!paths.is_empty(), "Usage: wakwak [--name NAME] <data.wf> [more-data.wf ...]");
    let paths: Vec<&str> = paths.iter().map(String::as_str).collect();
    let inputs = ChessBucketsMirrored::default();
    let loader = WakFormatLoader::new_concat_multiple(&paths, 1024, 4, |_, _, _, _| true);
    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(inputs)
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            SavedFormat::id("l1w").round().quantise::<i16>(QB),
            SavedFormat::id("l1b").round().quantise::<i16>(QA * QB),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs| {
            let l0 = builder.new_affine("l0", inputs.num_inputs(), HIDDEN_SIZE);
            let l1 = builder.new_affine("l1", 2 * HIDDEN_SIZE, 1);
            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            l1.forward(stm_hidden.concat(ntm_hidden))
        });

    let schedule = TrainingSchedule {
        net_id: net_id.unwrap_or_else(|| format!("chess768hm-{HIDDEN_SIZE}")),
        eval_scale: SCALE,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: 240,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.3 },
        lr_scheduler: lr::Sequence {
            first: lr::ConstantLR { value: 0.001 },
            second: lr::LinearDecayLR { initial_lr: 0.001, final_lr: 0.000025, final_superbatch: 239 },
            first_scheduler_final_superbatch: 1,
        },
        save_rate: 40,
    };
    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 64 };
    trainer.run(&schedule, &settings, &loader);
}
