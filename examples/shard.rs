use std::{
    env, fs, io,
    path::{Path, PathBuf},
    str::FromStr,
};

use bullet_lib::{
    game::{
        inputs::{ChessBucketsMirrored, get_num_buckets},
        outputs::MaterialCount,
    },
    nn::{
        InitSettings, Shape,
        optimiser::{AdamW, AdamWParams},
    },
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{
        ValueTrainerBuilder,
        loader::viribinpack::{Filter, ViriBinpackLoader, ViriFilter},
    },
};

const HIDDEN_SIZE: usize = 768;
const NUM_OUTPUT_BUCKETS: usize = 8;
const QA: i16 = 255;
const QB: i16 = 64;
const EVAL_SCALE: i32 = 400;

#[rustfmt::skip]
const BUCKET_LAYOUT: [usize; 32] = [
     0,  1,  2,  3,
     0,  1,  2,  3,
     4,  5,  6,  7,
     4,  5,  6,  7,
     8,  9, 10, 11,
     8,  9, 10, 11,
    12, 13, 14, 15,
    12, 13, 14, 15,
];

const NUM_INPUT_BUCKETS: usize = get_num_buckets(&BUCKET_LAYOUT);
const NUM_INPUTS: usize = 768 * NUM_INPUT_BUCKETS;

fn main() {
    let data_paths = collect_data_paths();
    let data_path_refs = data_paths.iter().map(String::as_str).collect::<Vec<_>>();

    let initial_lr = env_value("SHARD_INITIAL_LR", 0.001_f32);
    let final_lr = env_value("SHARD_FINAL_LR", 0.000025_f32);
    let superbatches = env_value("SHARD_SUPERBATCHES", 25_usize);
    let threads = env_value("SHARD_THREADS", 4_usize);
    let buffer_size_mb = env_value("SHARD_BUFFER_SIZE_MB", 1024_usize);
    let output_directory = env::var("SHARD_OUTPUT_DIR").unwrap_or_else(|_| "checkpoints".to_string());

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>)
        .save_format(&[
            // Merge the training-only PSQT factoriser into the feature weights.
            SavedFormat::id("l0w")
                .transform(|store, weights| {
                    let factoriser = store.get("l0f").values.f32().repeat(NUM_INPUT_BUCKETS);
                    weights.into_iter().zip(factoriser).map(|(bucket, shared)| bucket + shared).collect()
                })
                .round()
                .quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            // Output-bucket weights are transposed for efficient CPU inference.
            SavedFormat::id("l1w").transpose().round().quantise::<i16>(QB),
            SavedFormat::id("l1b").round().quantise::<i16>(QA * QB),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0f = builder.new_weights("l0f", Shape::new(HIDDEN_SIZE, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(NUM_INPUT_BUCKETS);

            let mut l0 = builder.new_affine("l0", NUM_INPUTS, HIDDEN_SIZE);
            l0.weights = l0.weights + expanded_factoriser;

            let l1 = builder.new_affine("l1", 2 * HIDDEN_SIZE, NUM_OUTPUT_BUCKETS);

            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            l1.forward(stm_hidden.concat(ntm_hidden)).select(output_buckets)
        });

    // Each first-layer value saved for inference is the sum of l0w and l0f.
    // These limits keep the merged i16 accumulators within Sable's accepted range.
    let first_layer_clipping = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l0w", first_layer_clipping);
    trainer.optimiser.set_params_for_weight("l0f", first_layer_clipping);

    let schedule = TrainingSchedule {
        net_id: "shard".to_string(),
        eval_scale: EVAL_SCALE as f32,
        steps: TrainingSteps {
            batch_size: env_value("SHARD_BATCH_SIZE", 16_384_usize),
            batches_per_superbatch: env_value("SHARD_BATCHES_PER_SUPERBATCH", 6104_usize),
            start_superbatch: 1,
            end_superbatch: superbatches,
        },
        wdl_scheduler: wdl::LinearWDL {
            start: env_value("SHARD_WDL_START", 0.1_f32),
            end: env_value("SHARD_WDL_END", 0.7_f32),
        },
        lr_scheduler: lr::CosineDecayLR { initial_lr, final_lr, final_superbatch: superbatches },
        save_rate: env_value("SHARD_SAVE_RATE", 10_usize),
    };

    let settings = LocalSettings {
        threads,
        test_set: None,
        output_directory: &output_directory,
        batch_queue_size: env_value("SHARD_BATCH_QUEUE_SIZE", 32_usize),
    };

    let dataloader = ViriBinpackLoader::new_concat_multiple(
        &data_path_refs,
        buffer_size_mb,
        threads,
        ViriFilter::Builtin(Filter::default()),
    );

    trainer.run(&schedule, &settings, &dataloader);
}

fn collect_data_paths() -> Vec<String> {
    let roots = env::args_os().skip(1).map(PathBuf::from).collect::<Vec<_>>();
    assert!(!roots.is_empty(), "pass one or more Viriformat .vf/.binpack files or directories after `--`");

    let mut paths = Vec::new();
    for root in roots {
        collect_data_path(&root, true, &mut paths)
            .unwrap_or_else(|error| panic!("failed to inspect '{}': {error}", root.display()));
    }

    paths.sort();
    paths.dedup();
    assert!(!paths.is_empty(), "no Viriformat .vf/.binpack files were found");

    println!("Using {} Viriformat file(s):", paths.len());
    for path in &paths {
        println!("  {}", path.display());
    }

    paths
        .into_iter()
        .map(|path| {
            path.into_os_string()
                .into_string()
                .unwrap_or_else(|path| panic!("dataset path is not valid UTF-8: {path:?}"))
        })
        .collect()
}

fn collect_data_path(path: &Path, explicit: bool, paths: &mut Vec<PathBuf>) -> io::Result<()> {
    let metadata = fs::metadata(path)?;
    if metadata.is_file() {
        if explicit || is_viriformat(path) {
            paths.push(path.to_path_buf());
        }
        return Ok(());
    }

    if metadata.is_dir() {
        let mut entries = fs::read_dir(path)?.collect::<Result<Vec<_>, _>>()?;
        entries.sort_by_key(|entry| entry.path());
        for entry in entries {
            collect_data_path(&entry.path(), false, paths)?;
        }
    }

    Ok(())
}

fn is_viriformat(path: &Path) -> bool {
    path.extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("vf") || extension.eq_ignore_ascii_case("binpack"))
}

fn env_value<T>(name: &str, default: T) -> T
where
    T: FromStr,
    T::Err: std::fmt::Display,
{
    match env::var(name) {
        Ok(value) => value.parse().unwrap_or_else(|error| panic!("invalid {name}={value:?}: {error}")),
        Err(env::VarError::NotPresent) => default,
        Err(error) => panic!("failed to read {name}: {error}"),
    }
}
