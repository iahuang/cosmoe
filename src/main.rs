mod expert_cache;
mod mixtral_offloading;
mod mixtral_original;
mod pread_loader;
mod prefetcher;

use anyhow::{Error as E, Result};

use mixtral_offloading::{Config, Model};

use candle_core::{DType, Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{Repo, RepoType, api::sync::Api};
use tokenizers::Tokenizer;

use pread_loader::PreadTensorLoader;

struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        use std::io::Write;
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        for &t in tokens.iter() {
            if let Some(t) = self.tokenizer.next_token(t)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("</s>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the </s> token"),
        };
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;

        self.model.stats.print_summary();
        self.model.loader.print_io_stats();
        Ok(())
    }
}

#[derive(Debug)]
struct Args {
    use_flash_attn: bool,
    prompt: String,
    temperature: Option<f64>,
    top_p: Option<f64>,
    seed: u64,
    sample_len: usize,
    model_id: String,
    gguf_filename: String,
    tokenizer_repo: String,
    tokenizer_file: Option<String>,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

fn main() -> Result<()> {
    let args = Args {
        use_flash_attn: false,
        prompt: "Hello, world!".to_string(),
        temperature: None,
        top_p: None,
        seed: 299792458,
        sample_len: 5,
        model_id: "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF".to_string(),
        gguf_filename: "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf".to_string(),
        tokenizer_repo: "mistralai/Mixtral-8x7B-Instruct-v0.1".to_string(),
        tokenizer_file: None,
        repeat_penalty: 1.1,
        repeat_last_n: 64,
    };

    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature.unwrap_or(0.),
        args.repeat_penalty,
        args.repeat_last_n
    );

    let start = std::time::Instant::now();
    let api = Api::new()?;

    let tokenizer_repo = api.repo(Repo::new(args.tokenizer_repo, RepoType::Model));
    let tokenizer_filename = match args.tokenizer_file {
        Some(file) => std::path::PathBuf::from(file),
        None => tokenizer_repo.get("tokenizer.json")?,
    };

    let gguf_repo = api.repo(Repo::new(args.model_id, RepoType::Model));
    let gguf_path = gguf_repo.get(&args.gguf_filename)?;
    println!("retrieved the files in {:?}", start.elapsed());

    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let config = Config::v0_1_8x7b(args.use_flash_attn);
    let device = candle_examples::device(false)?;

    let loader = std::sync::Arc::new(
        PreadTensorLoader::new(&gguf_path).map_err(|e| anyhow::anyhow!("{}", e))?,
    );

    let model = Model::new(&config, 16, loader, &device)?;
    println!("loaded the model in {:?}", start.elapsed());

    // Reset I/O stats so we only measure expert loading during inference.
    model.loader.reset_stats();

    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        args.seed,
        args.temperature,
        args.top_p,
        args.repeat_penalty,
        args.repeat_last_n,
        &device,
    );
    pipeline.run(&args.prompt, args.sample_len)?;
    Ok(())
}
