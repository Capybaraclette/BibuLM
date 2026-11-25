//! Speed benchmark, interactive API playground, `llama-bench` matrix runner, and memory placement inspector.
//! Captures TTFT, prompt eval (`pp`) tok/s, generation (`tg`) tok/s, reasoning vs answer token counts,
//! KV cache reuse, and exact GPU VRAM vs System RAM component placement.

pub mod llama_bench;
pub mod playground;
pub mod quick_test;

pub use llama_bench::{
    LlamaBenchRequest, LlamaBenchRow, build_llama_bench_args, parse_llama_bench_json,
    run_llama_bench,
};
pub use playground::{
    PlaygroundMessage, PlaygroundStreamChunk, PlaygroundTurnRequest, PlaygroundTurnResult,
    run_playground_turn,
};
pub use quick_test::{
    BenchError, QuickTestRequest, QuickTestResult, QuickTestStreamChunk,
    drain_complete_sse_lines, resolve_preset_prompt, run_quick_speed_test,
    run_quick_speed_test_against_url,
};
