use bullet_lib::{
    game::{
        formats::{bulletformat::ChessBoard, wakformat::bullet::DUCK_EXTRA_INDEX},
        inputs::{Chess768, SparseInputType},
    },
    nn::optimiser::AdamW,
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader::WakFormatLoader},
};

const HIDDEN_SIZE: usize = 16;
const SCALE: f32 = 400.0;
const QA: i16 = 255;
const QB: i16 = 64;

fn main() {
    let mut paths: Vec<String> = std::env::args().skip(1).collect();
    let no_duck = paths.first().is_some_and(|arg| arg == "--no-duck");
    if no_duck {
        paths.remove(0);
    }
    assert!(!paths.is_empty(), "Usage: wakwak [--no-duck] <data.wf> [more-data.wf ...]");
    let paths: Vec<&str> = paths.iter().map(String::as_str).collect();
    let inputs = WakwakInputs { duck: !no_duck };
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
        net_id: if no_duck { format!("chess768-{HIDDEN_SIZE}") } else { format!("duckchess-832x{HIDDEN_SIZE}") },
        eval_scale: SCALE,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: 30,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.0 },
        lr_scheduler: lr::Sequence {
            first: lr::ConstantLR { value: 0.001 },
            second: lr::LinearDecayLR { initial_lr: 0.001, final_lr: 0.000025, final_superbatch: 29 },
            first_scheduler_final_superbatch: 1,
        },
        save_rate: 10,
    };
    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 64 };
    trainer.run(&schedule, &settings, &loader);
}

#[derive(Clone, Copy)]
struct WakwakInputs {
    duck: bool,
}

impl SparseInputType for WakwakInputs {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        768 + 64 * usize::from(self.duck)
    }

    fn max_active(&self) -> usize {
        32 + usize::from(self.duck)
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &ChessBoard, mut f: F) {
        Chess768.map_features(pos, &mut f);
        if self.duck {
            let square = usize::from(pos.extra()[DUCK_EXTRA_INDEX]);
            if square < 64 {
                f(768 + square, 768 + (square ^ 56));
            }
        }
    }

    fn shorthand(&self) -> String {
        self.num_inputs().to_string()
    }

    fn description(&self) -> String {
        if self.duck { "Chess piece-square and duck-square inputs".to_string() } else { Chess768.description() }
    }
}
