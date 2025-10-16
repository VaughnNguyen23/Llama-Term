use anyhow::Result;
use chrono::Local;
use ollama_rs::{generation::completion::request::GenerationRequest, models::ModelOptions, Ollama};
use ratatui::widgets::ListState;
use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf, sync::Arc};
use sysinfo::System;
use tokio::sync::Mutex;
use tokio_stream::StreamExt;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AppMode {
    Chat,
    ModelSelection,
    ModelDownload,
    SystemMonitor,
    ChatHistory,
    ModelConfig,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConfigField {
    Temperature,
    TopP,
    TopK,
    RepeatPenalty,
    ContextWindow,
    SystemPrompt,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ChatSession {
    pub timestamp: String,
    pub model: String,
    pub messages: Vec<(String, String)>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ModelConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub repeat_penalty: f32,
    pub num_ctx: u64,
    pub system_prompt: String,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.1,
            num_ctx: 2048,
            system_prompt: String::from("You are a helpful AI assistant."),
        }
    }
}

pub struct App {
    pub mode: AppMode,
    pub input: String,
    pub messages: Vec<(String, String)>, // (role, content)
    pub current_model: String,
    pub available_models: Vec<String>,
    pub model_list_state: ListState,
    pub download_input: String,
    pub status_message: String,
    pub ollama: Ollama,
    pub scroll_offset: usize,
    pub is_thinking: bool,
    pub thinking_frame: usize,
    pub sys_info: System,
    pub cpu_usage: f32,
    pub memory_usage: u64,
    pub memory_total: u64,
    pub gpu_info: Option<String>,
    pub chat_history: Vec<ChatSession>,
    pub history_list_state: ListState,
    pub chat_dir: PathBuf,
    pub selected_text: Option<String>,
    pub process_scroll: usize,
    pub model_config: ModelConfig,
    pub config_field: ConfigField,
    pub config_input: String,
    pub config_dir: PathBuf,
    pub vim_mode: bool,
    pub vim_insert: bool,
    pub pending_g: bool,
}

impl App {
    pub fn new() -> Self {
        let ollama = Ollama::default();
        let mut sys_info = System::new_all();
        sys_info.refresh_all();

        // Create directories
        let base_dir = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".ollama_tui");
        let chat_dir = base_dir.join("chats");
        let config_dir = base_dir.clone();

        fs::create_dir_all(&chat_dir).ok();
        fs::create_dir_all(&config_dir).ok();

        // Load config or use default
        let config_path = config_dir.join("model_config.json");
        let model_config = if let Ok(content) = fs::read_to_string(&config_path) {
            serde_json::from_str(&content).unwrap_or_default()
        } else {
            ModelConfig::default()
        };

        Self {
            mode: AppMode::Chat,
            input: String::new(),
            messages: Vec::new(),
            current_model: String::from("llama2:latest"),
            available_models: Vec::new(),
            model_list_state: ListState::default(),
            download_input: String::new(),
            status_message: String::from("Ready. Press F1 for help"),
            ollama,
            scroll_offset: 0,
            is_thinking: false,
            thinking_frame: 0,
            sys_info,
            cpu_usage: 0.0,
            memory_usage: 0,
            memory_total: 0,
            gpu_info: None,
            chat_history: Vec::new(),
            history_list_state: ListState::default(),
            chat_dir,
            selected_text: None,
            process_scroll: 0,
            model_config,
            config_field: ConfigField::Temperature,
            config_input: String::new(),
            config_dir,
            vim_mode: true,
            vim_insert: true,
            pending_g: false,
        }
    }

    pub fn get_thinking_spinner(&self) -> &str {
        let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        frames[self.thinking_frame % frames.len()]
    }

    pub fn update_thinking_animation(&mut self) {
        if self.is_thinking {
            self.thinking_frame += 1;
        }
    }

    pub fn update_system_info(&mut self) {
        self.sys_info.refresh_all();

        // Calculate average CPU usage
        let cpus = self.sys_info.cpus();
        self.cpu_usage = if !cpus.is_empty() {
            cpus.iter().map(|cpu| cpu.cpu_usage()).sum::<f32>() / cpus.len() as f32
        } else {
            0.0
        };

        self.memory_usage = self.sys_info.used_memory();
        self.memory_total = self.sys_info.total_memory();

        // Try to get GPU info using nvidia-smi
        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .args(&[
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ])
            .output()
        {
            if output.status.success() {
                self.gpu_info = Some(String::from_utf8_lossy(&output.stdout).to_string());
            }
        }
    }

    pub fn save_current_chat(&mut self) -> Result<()> {
        if self.messages.is_empty() {
            return Ok(());
        }

        let session = ChatSession {
            timestamp: Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
            model: self.current_model.clone(),
            messages: self.messages.clone(),
        };

        let filename = format!("chat_{}.json", Local::now().format("%Y%m%d_%H%M%S"));
        let path = self.chat_dir.join(filename);
        let json = serde_json::to_string_pretty(&session)?;
        fs::write(path, json)?;

        self.status_message = "Chat saved successfully".to_string();
        Ok(())
    }

    pub fn load_chat_history(&mut self) -> Result<()> {
        self.chat_history.clear();

        if let Ok(entries) = fs::read_dir(&self.chat_dir) {
            for entry in entries.flatten() {
                if let Ok(content) = fs::read_to_string(entry.path()) {
                    if let Ok(session) = serde_json::from_str::<ChatSession>(&content) {
                        self.chat_history.push(session);
                    }
                }
            }
        }

        // Sort by timestamp (newest first)
        self.chat_history
            .sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        Ok(())
    }

    pub fn load_selected_chat(&mut self) -> Result<()> {
        if let Some(selected) = self.history_list_state.selected() {
            if let Some(session) = self.chat_history.get(selected) {
                self.messages = session.messages.clone();
                self.current_model = session.model.clone();
                self.status_message = format!("Loaded chat from {}", session.timestamp);
                self.switch_mode(AppMode::Chat);
            }
        }
        Ok(())
    }

    pub fn clear_chat(&mut self) {
        self.messages.clear();
        self.scroll_offset = 0;
        self.status_message = "Chat cleared".to_string();
    }

    pub fn copy_to_clipboard(&mut self) {
        if let Some(text) = &self.selected_text {
            if let Ok(mut clipboard) = arboard::Clipboard::new() {
                if clipboard.set_text(text.clone()).is_ok() {
                    self.status_message = "Copied to clipboard".to_string();
                } else {
                    self.status_message = "Failed to copy".to_string();
                }
            }
        }
    }

    pub fn select_last_message(&mut self) {
        if let Some((_, content)) = self.messages.last() {
            self.selected_text = Some(content.clone());
            self.status_message = "Message selected. Press Ctrl+Y to copy".to_string();
        }
    }

    pub fn save_config(&mut self) -> Result<()> {
        let config_path = self.config_dir.join("model_config.json");
        let json = serde_json::to_string_pretty(&self.model_config)?;
        fs::write(config_path, json)?;
        self.status_message = "Configuration saved".to_string();
        Ok(())
    }

    pub fn update_config_field(&mut self, value: String) {
        match self.config_field {
            ConfigField::Temperature => {
                if let Ok(val) = value.parse::<f32>() {
                    self.model_config.temperature = val.clamp(0.0, 2.0);
                }
            }
            ConfigField::TopP => {
                if let Ok(val) = value.parse::<f32>() {
                    self.model_config.top_p = val.clamp(0.0, 1.0);
                }
            }
            ConfigField::TopK => {
                if let Ok(val) = value.parse::<u32>() {
                    self.model_config.top_k = val.max(1);
                }
            }
            ConfigField::RepeatPenalty => {
                if let Ok(val) = value.parse::<f32>() {
                    self.model_config.repeat_penalty = val.clamp(0.0, 2.0);
                }
            }
            ConfigField::ContextWindow => {
                if let Ok(val) = value.parse::<u64>() {
                    self.model_config.num_ctx = val.clamp(512, 32768);
                }
            }
            ConfigField::SystemPrompt => {
                self.model_config.system_prompt = value;
            }
        }
    }

    pub fn next_config_field(&mut self) {
        self.config_field = match self.config_field {
            ConfigField::Temperature => ConfigField::TopP,
            ConfigField::TopP => ConfigField::TopK,
            ConfigField::TopK => ConfigField::RepeatPenalty,
            ConfigField::RepeatPenalty => ConfigField::ContextWindow,
            ConfigField::ContextWindow => ConfigField::SystemPrompt,
            ConfigField::SystemPrompt => ConfigField::Temperature,
        };
    }

    pub fn prev_config_field(&mut self) {
        self.config_field = match self.config_field {
            ConfigField::Temperature => ConfigField::SystemPrompt,
            ConfigField::TopP => ConfigField::Temperature,
            ConfigField::TopK => ConfigField::TopP,
            ConfigField::RepeatPenalty => ConfigField::TopK,
            ConfigField::ContextWindow => ConfigField::RepeatPenalty,
            ConfigField::SystemPrompt => ConfigField::ContextWindow,
        };
    }

    pub fn get_current_config_value(&self) -> String {
        match self.config_field {
            ConfigField::Temperature => self.model_config.temperature.to_string(),
            ConfigField::TopP => self.model_config.top_p.to_string(),
            ConfigField::TopK => self.model_config.top_k.to_string(),
            ConfigField::RepeatPenalty => self.model_config.repeat_penalty.to_string(),
            ConfigField::ContextWindow => self.model_config.num_ctx.to_string(),
            ConfigField::SystemPrompt => self.model_config.system_prompt.clone(),
        }
    }

    pub fn switch_mode(&mut self, mode: AppMode) {
        self.mode = mode;
        if mode == AppMode::ModelSelection {
            self.model_list_state.select(Some(0));
        }
    }

    pub async fn fetch_models(&mut self) -> Result<()> {
        let models = self.ollama.list_local_models().await?;
        self.available_models = models.iter().map(|m| m.name.clone()).collect();
        Ok(())
    }

    pub async fn download_model(&mut self, model_name: String) -> Result<()> {
        self.status_message = format!("Downloading model: {}", model_name);
        self.ollama.pull_model(model_name.clone(), false).await?;
        self.status_message = format!("Model {} downloaded successfully", model_name);
        self.fetch_models().await?;
        Ok(())
    }

    pub fn start_message_stream(&mut self, shared_app: Arc<Mutex<App>>) {
        if self.input.trim().is_empty() {
            return;
        }

        let user_message = self.input.clone();
        self.messages
            .push(("user".to_string(), user_message.clone()));
        self.input.clear();

        // Start thinking animation
        self.is_thinking = true;
        self.thinking_frame = 0;
        self.messages.push(("assistant".to_string(), String::new()));

        let model = self.current_model.clone();
        let ollama = self.ollama.clone();
        let config = self.model_config.clone();

        // Spawn the streaming task in the background
        tokio::spawn(async move {
            let message_index = {
                let app = shared_app.lock().await;
                app.messages.len() - 1
            };

            // Build request with config parameters using ModelOptions
            let options = ModelOptions::default()
                .temperature(config.temperature)
                .top_p(config.top_p)
                .top_k(config.top_k)
                .repeat_penalty(config.repeat_penalty)
                .num_ctx(config.num_ctx);

            let mut request = GenerationRequest::new(model, user_message).options(options);

            // Add system prompt if not empty
            if !config.system_prompt.is_empty() {
                request = request.system(config.system_prompt);
            }

            match ollama.generate_stream(request).await {
                Ok(mut stream) => {
                    while let Some(responses) = stream.next().await {
                        match responses {
                            Ok(response_chunks) => {
                                for response in response_chunks {
                                    // Append each token to the message as it arrives
                                    let mut app = shared_app.lock().await;
                                    if let Some((_, content)) = app.messages.get_mut(message_index)
                                    {
                                        content.push_str(&response.response);
                                    }
                                }
                            }
                            Err(e) => {
                                let mut app = shared_app.lock().await;
                                app.status_message = format!("Stream error: {}", e);
                                break;
                            }
                        }
                    }
                    let mut app = shared_app.lock().await;
                    app.status_message = "Ready".to_string();
                    app.is_thinking = false;
                }
                Err(e) => {
                    let mut app = shared_app.lock().await;
                    // Remove the empty thinking message on error
                    app.messages.pop();
                    app.status_message = format!("Error: {}", e);
                    app.is_thinking = false;
                }
            }
        });
    }

    pub fn scroll_up(&mut self) {
        if self.scroll_offset > 0 {
            self.scroll_offset -= 1;
        }
    }
    pub fn scroll_down(&mut self) {
        self.scroll_offset += 1;
    }
    pub fn scroll_top(&mut self) {
        self.scroll_offset = 0;
    }
    pub fn scroll_bottom(&mut self) {
        self.scroll_offset = u16::MAX as usize;
    }
}
