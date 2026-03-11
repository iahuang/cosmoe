use std::sync::Arc;

use candle_core::{Device, quantized::QTensor};

use crate::pread_loader::PreadTensorLoader;
use flume::{Receiver, Sender, TryRecvError};

pub trait Primitive: Copy + Send + Sync + 'static {}

pub struct PrefetchRequest<T: Primitive> {
    pub tensors: Vec<String>,
    pub id: T,
}

pub struct PrefetchResponse<T: Primitive> {
    pub id: T,
    pub tensors: Result<Vec<QTensor>, candle_core::Error>,
}

pub struct Prefetcher<T: Primitive> {
    tx: Sender<PrefetchRequest<T>>,
    rx: Receiver<PrefetchResponse<T>>,
}

impl<T: Primitive> Prefetcher<T> {
    pub fn new(loader: Arc<PreadTensorLoader>, device: Device) -> Self {
        let (req_tx, req_rx) = flume::unbounded::<PrefetchRequest<T>>();
        let (res_tx, res_rx) = flume::unbounded::<PrefetchResponse<T>>();

        std::thread::spawn(move || {
            while let Ok(request) = req_rx.recv() {
                let tensors = request
                    .tensors
                    .iter()
                    .map(|tensor| loader.load_qtensor(tensor, &device))
                    .collect::<Result<Vec<QTensor>, candle_core::Error>>();
                res_tx
                    .send(PrefetchResponse {
                        id: request.id,
                        tensors,
                    })
                    .unwrap();
            }
        });

        Self {
            tx: req_tx,
            rx: res_rx,
        }
    }

    pub fn enqueue(&mut self, id: T, tensors: Vec<String>) {
        self.tx.send(PrefetchRequest { tensors, id }).unwrap();
    }

    pub fn try_recv(&self) -> Option<PrefetchResponse<T>> {
        match self.rx.try_recv() {
            Ok(response) => Some(response),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => panic!("Prefetcher disconnected"),
        }
    }

    pub fn recv_blocking(&self) -> PrefetchResponse<T> {
        match self.rx.recv() {
            Ok(response) => response,
            Err(flume::RecvError::Disconnected) => panic!("Prefetcher disconnected"),
        }
    }
}
