use std::{
    fs::File,
    io::BufReader,
    sync::mpsc::{self, Receiver, SyncSender},
};

use crate::game::formats::bulletformat::ChessBoard;

use super::rng::seeded_rng;

use bullet_trainer::reader::DataReader;
use wakformat::{board::Board, bullet::splat_to_bulletformat, common::Move, format};

#[derive(Clone)]
pub struct WakFormatLoader {
    file_paths: Vec<String>,
    buffer_size: usize,
    threads: usize,
    filter: fn(&Board, Move, i16, f32) -> bool,
}

impl WakFormatLoader {
    pub fn new(path: &str, buffer_size_mb: usize, threads: usize, filter: fn(&Board, Move, i16, f32) -> bool) -> Self {
        Self::new_concat_multiple(&[path], buffer_size_mb, threads, filter)
    }

    pub fn new_concat_multiple(
        paths: &[&str],
        buffer_size_mb: usize,
        threads: usize,
        filter: fn(&Board, Move, i16, f32) -> bool,
    ) -> Self {
        Self {
            file_paths: paths.iter().map(|x| x.to_string()).collect(),
            buffer_size: buffer_size_mb * 1024 * 1024 / std::mem::size_of::<ChessBoard>() / 2,
            threads,
            filter,
        }
    }
}

impl DataReader<ChessBoard> for WakFormatLoader {
    fn read_chunks<F: FnMut(&[ChessBoard]) -> bool>(&self, _: usize, mut f: F) {
        let mut shuffle_buffer = Vec::new();
        shuffle_buffer.reserve_exact(self.buffer_size);

        let file_paths = self.file_paths.clone();
        let buffer_size = self.buffer_size;
        let threads = self.threads;
        let filter = self.filter;

        let (sender, receiver) = mpsc::sync_channel::<Vec<Vec<u8>>>(4);
        let (msg_sender, msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            read_concat(&file_paths, threads, sender, msg_receiver);
        });

        let (game_sender, game_receiver) = mpsc::sync_channel::<Vec<ChessBoard>>(4 * self.threads);
        let (game_msg_sender, game_msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            'dataloading: while let Ok(games) = receiver.recv() {
                if game_msg_receiver.try_recv().unwrap_or(false) {
                    msg_sender.send(true).unwrap();
                    break 'dataloading;
                }

                convert_buffer(threads, &game_sender, &games, &filter);
            }
        });

        let (buffer_sender, buffer_receiver) = mpsc::sync_channel::<Vec<ChessBoard>>(0);
        let (buffer_msg_sender, buffer_msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            'dataloading: while let Ok(game) = game_receiver.recv() {
                if buffer_msg_receiver.try_recv().unwrap_or(false) {
                    game_msg_sender.send(true).unwrap();
                    break 'dataloading;
                }

                if shuffle_buffer.len() + game.len() < shuffle_buffer.capacity() {
                    shuffle_buffer.extend_from_slice(&game);
                } else {
                    let diff = shuffle_buffer.capacity() - shuffle_buffer.len();
                    if diff > 0 {
                        shuffle_buffer.extend_from_slice(&game[..diff]);
                    }

                    shuffle(&mut shuffle_buffer);

                    if buffer_msg_receiver.try_recv().unwrap_or(false) || buffer_sender.send(shuffle_buffer).is_err() {
                        game_msg_sender.send(true).unwrap();
                        break 'dataloading;
                    }

                    shuffle_buffer = Vec::new();
                    shuffle_buffer.reserve_exact(buffer_size);
                    shuffle_buffer.extend_from_slice(&game[diff..]);
                }
            }
        });

        'dataloading: while let Ok(shuffle_buffer) = buffer_receiver.recv() {
            if f(&shuffle_buffer) {
                buffer_msg_sender.send(true).unwrap();
                break 'dataloading;
            }
        }

        drop(buffer_receiver);
    }
}

fn read_concat(file_paths: &[String], threads: usize, sender: SyncSender<Vec<Vec<u8>>>, msg_receiver: Receiver<bool>) {
    let mut games = Vec::new();
    loop {
        let mut count = 0;
        for file_path in file_paths {
            let mut reader = BufReader::new(File::open(file_path).unwrap());
            format::read_file_header(&mut reader).unwrap();

            loop {
                let mut buf = Vec::new();
                if !format::read_game(&mut reader, &mut buf).unwrap() {
                    break;
                }
                count += 1;
                games.push(buf);

                if games.len().is_multiple_of(8192 * threads) {
                    if msg_receiver.try_recv().unwrap_or(false) || sender.send(games).is_err() {
                        return;
                    }
                    games = Vec::new();
                }
            }
        }
        if count == 0 {
            return;
        }
    }
}

fn convert_buffer(
    threads: usize,
    sender: &SyncSender<Vec<ChessBoard>>,
    games: &[Vec<u8>],
    filter: &fn(&Board, Move, i16, f32) -> bool,
) {
    let chunk_size = games.len().div_ceil(threads);

    std::thread::scope(|s| {
        for chunk in games.chunks(chunk_size) {
            let this_sender = sender.clone();
            s.spawn(move || {
                let mut buffer = Vec::new();

                for game_bytes in chunk {
                    parse_into_buffer(game_bytes, &mut buffer, filter);
                }

                this_sender.send(buffer)
            });
        }
    });
}

fn parse_into_buffer(game: &[u8], buffer: &mut Vec<ChessBoard>, filter: &fn(&Board, Move, i16, f32) -> bool) {
    splat_to_bulletformat(
        game,
        |board| {
            buffer.push(board);
            Ok(())
        },
        filter,
    )
    .unwrap();
}

fn shuffle(data: &mut [ChessBoard]) {
    let mut rng = seeded_rng();

    for i in (0..data.len()).rev() {
        let idx = rng.rand_range(0..i as u64 + 1) as usize;
        data.swap(idx, i);
    }
}
