use anyhow::Result;
use chrono::Local;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ollama_rs::{generation::completion::request::GenerationRequest, models::ModelOptions, Ollama};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Gauge, List, ListItem, ListState, Paragraph, Row, Table, Wrap},
    Frame, Terminal,
};
use serde::{Deserialize, Serialize};
use std::{fs, io, path::PathBuf, sync::Arc, time::Duration};
use sysinfo::System;
use tokio::sync::Mutex;
use tokio_stream::StreamExt;

#[derive(Debug, Clone, Copy, PartialEq)]
enum AppMode {
    Chat,
    ModelSelection,
    ModelDownload,
    SystemMonitor,
    ChatHistory,
    ModelConfig,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ConfigField {
    Temperature,
    TopP,
    TopK,
    RepeatPenalty,
    ContextWindow,
    SystemPrompt,
}

#[derive(Serialize, Deserialize, Clone)]
struct ChatSession {
    timestamp: String,
    model: String,
    messages: Vec<(String, String)>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct ModelConfig {
    temperature: f32,
    top_p: f32,
    top_k: u32,
    repeat_penalty: f32,
    num_ctx: u64,
    system_prompt: String,
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

struct App {
    mode: AppMode,
    input: String,
    messages: Vec<(String, String)>, // (role, content)
    current_model: String,
    available_models: Vec<String>,
    model_list_state: ListState,
    download_input: String,
    status_message: String,
    ollama: Ollama,
    scroll_offset: usize,
    is_thinking: bool,
    thinking_frame: usize,
    sys_info: System,
    cpu_usage: f32,
    memory_usage: u64,
    memory_total: u64,
    gpu_info: Option<String>,
    chat_history: Vec<ChatSession>,
    history_list_state: ListState,
    chat_dir: PathBuf,
    selected_text: Option<String>,
    process_scroll: usize,
    model_config: ModelConfig,
    config_field: ConfigField,
    config_input: String,
    config_dir: PathBuf,
}

impl App {
    fn new() -> Self {
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
        }
    }

    fn get_thinking_spinner(&self) -> &str {
        let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        frames[self.thinking_frame % frames.len()]
    }

    fn update_thinking_animation(&mut self) {
        if self.is_thinking {
            self.thinking_frame += 1;
        }
    }

    fn update_system_info(&mut self) {
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

    fn save_current_chat(&mut self) -> Result<()> {
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

    fn load_chat_history(&mut self) -> Result<()> {
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

    fn load_selected_chat(&mut self) -> Result<()> {
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

    fn clear_chat(&mut self) {
        self.messages.clear();
        self.scroll_offset = 0;
        self.status_message = "Chat cleared".to_string();
    }

    fn copy_to_clipboard(&mut self) {
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

    fn select_last_message(&mut self) {
        if let Some((_, content)) = self.messages.last() {
            self.selected_text = Some(content.clone());
            self.status_message = "Message selected. Press Ctrl+Y to copy".to_string();
        }
    }

    fn save_config(&mut self) -> Result<()> {
        let config_path = self.config_dir.join("model_config.json");
        let json = serde_json::to_string_pretty(&self.model_config)?;
        fs::write(config_path, json)?;
        self.status_message = "Configuration saved".to_string();
        Ok(())
    }

    fn update_config_field(&mut self, value: String) {
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

    fn next_config_field(&mut self) {
        self.config_field = match self.config_field {
            ConfigField::Temperature => ConfigField::TopP,
            ConfigField::TopP => ConfigField::TopK,
            ConfigField::TopK => ConfigField::RepeatPenalty,
            ConfigField::RepeatPenalty => ConfigField::ContextWindow,
            ConfigField::ContextWindow => ConfigField::SystemPrompt,
            ConfigField::SystemPrompt => ConfigField::Temperature,
        };
    }

    fn prev_config_field(&mut self) {
        self.config_field = match self.config_field {
            ConfigField::Temperature => ConfigField::SystemPrompt,
            ConfigField::TopP => ConfigField::Temperature,
            ConfigField::TopK => ConfigField::TopP,
            ConfigField::RepeatPenalty => ConfigField::TopK,
            ConfigField::ContextWindow => ConfigField::RepeatPenalty,
            ConfigField::SystemPrompt => ConfigField::ContextWindow,
        };
    }

    fn get_current_config_value(&self) -> String {
        match self.config_field {
            ConfigField::Temperature => self.model_config.temperature.to_string(),
            ConfigField::TopP => self.model_config.top_p.to_string(),
            ConfigField::TopK => self.model_config.top_k.to_string(),
            ConfigField::RepeatPenalty => self.model_config.repeat_penalty.to_string(),
            ConfigField::ContextWindow => self.model_config.num_ctx.to_string(),
            ConfigField::SystemPrompt => self.model_config.system_prompt.clone(),
        }
    }

    fn switch_mode(&mut self, mode: AppMode) {
        self.mode = mode;
        if mode == AppMode::ModelSelection {
            self.model_list_state.select(Some(0));
        }
    }

    async fn fetch_models(&mut self) -> Result<()> {
        let models = self.ollama.list_local_models().await?;
        self.available_models = models.iter().map(|m| m.name.clone()).collect();
        Ok(())
    }

    async fn download_model(&mut self, model_name: String) -> Result<()> {
        self.status_message = format!("Downloading model: {}", model_name);
        self.ollama.pull_model(model_name.clone(), false).await?;
        self.status_message = format!("Model {} downloaded successfully", model_name);
        self.fetch_models().await?;
        Ok(())
    }

    fn start_message_stream(&mut self, shared_app: Arc<Mutex<App>>) {
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

    fn scroll_up(&mut self) {
        if self.scroll_offset > 0 {
            self.scroll_offset -= 1;
        }
    }

    fn scroll_down(&mut self) {
        self.scroll_offset += 1;
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new();

    // Try to fetch models at startup
    let _ = app.fetch_models().await;

    let app_arc = Arc::new(Mutex::new(app));
    let res = run_app(&mut terminal, app_arc).await;

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        println!("Error: {:?}", err);
    }

    Ok(())
}

async fn run_app<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    app_arc: Arc<Mutex<App>>,
) -> Result<()> {
    loop {
        {
            let app = app_arc.lock().await;
            terminal.draw(|f| ui(f, &app))?;
        }

        // Update thinking animation and system info
        {
            let mut app = app_arc.lock().await;
            app.update_thinking_animation();
            if app.mode == AppMode::SystemMonitor {
                app.update_system_info();
            }
        }

        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                let mut app = app_arc.lock().await;
                match app.mode {
                    AppMode::Chat => match key.code {
                        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            return Ok(());
                        }
                        KeyCode::F(1) => {
                            app.status_message = "F2: Models | F3: Download | F4: Monitor | F5: History | F6: Save | F7: Clear | F8: Config | Ctrl+S: Select | Ctrl+Y: Copy".to_string();
                        }
                        KeyCode::F(2) => {
                            let _ = app.fetch_models().await;
                            app.switch_mode(AppMode::ModelSelection);
                        }
                        KeyCode::F(3) => {
                            app.switch_mode(AppMode::ModelDownload);
                        }
                        KeyCode::F(4) => {
                            app.update_system_info();
                            app.switch_mode(AppMode::SystemMonitor);
                        }
                        KeyCode::F(5) => {
                            let _ = app.load_chat_history();
                            app.history_list_state.select(Some(0));
                            app.switch_mode(AppMode::ChatHistory);
                        }
                        KeyCode::F(6) => {
                            let _ = app.save_current_chat();
                        }
                        KeyCode::F(7) => {
                            app.clear_chat();
                        }
                        KeyCode::F(8) => {
                            app.config_input = app.get_current_config_value();
                            app.switch_mode(AppMode::ModelConfig);
                        }
                        KeyCode::Char('s') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            app.select_last_message();
                        }
                        KeyCode::Char('y') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            app.copy_to_clipboard();
                        }
                        KeyCode::Enter => {
                            app.start_message_stream(Arc::clone(&app_arc));
                        }
                        KeyCode::Char(c) => {
                            app.input.push(c);
                        }
                        KeyCode::Backspace => {
                            app.input.pop();
                        }
                        KeyCode::Up => {
                            app.scroll_up();
                        }
                        KeyCode::Down => {
                            app.scroll_down();
                        }
                        _ => {}
                    },
                    AppMode::ModelSelection => match key.code {
                        KeyCode::Esc => {
                            app.switch_mode(AppMode::Chat);
                        }
                        KeyCode::Up => {
                            if let Some(selected) = app.model_list_state.selected() {
                                if selected > 0 {
                                    app.model_list_state.select(Some(selected - 1));
                                }
                            }
                        }
                        KeyCode::Down => {
                            if let Some(selected) = app.model_list_state.selected() {
                                if selected < app.available_models.len().saturating_sub(1) {
                                    app.model_list_state.select(Some(selected + 1));
                                }
                            }
                        }
                        KeyCode::Enter => {
                            if let Some(selected) = app.model_list_state.selected() {
                                if let Some(model) = app.available_models.get(selected).cloned() {
                                    app.current_model = model.clone();
                                    app.status_message = format!("Model changed to: {}", model);
                                    app.switch_mode(AppMode::Chat);
                                }
                            }
                        }
                        _ => {}
                    },
                    AppMode::ModelDownload => match key.code {
                        KeyCode::Esc => {
                            app.download_input.clear();
                            app.switch_mode(AppMode::Chat);
                        }
                        KeyCode::Enter => {
                            let model_name = app.download_input.clone();
                            app.download_input.clear();
                            let _ = app.download_model(model_name).await;
                            app.switch_mode(AppMode::Chat);
                        }
                        KeyCode::Char(c) => {
                            app.download_input.push(c);
                        }
                        KeyCode::Backspace => {
                            app.download_input.pop();
                        }
                        _ => {}
                    },
                    AppMode::SystemMonitor => match key.code {
                        KeyCode::Esc => {
                            app.switch_mode(AppMode::Chat);
                        }
                        KeyCode::Up => {
                            if app.process_scroll > 0 {
                                app.process_scroll -= 1;
                            }
                        }
                        KeyCode::Down => {
                            app.process_scroll += 1;
                        }
                        _ => {}
                    },
                    AppMode::ChatHistory => match key.code {
                        KeyCode::Esc => {
                            app.switch_mode(AppMode::Chat);
                        }
                        KeyCode::Up => {
                            if let Some(selected) = app.history_list_state.selected() {
                                if selected > 0 {
                                    app.history_list_state.select(Some(selected - 1));
                                }
                            }
                        }
                        KeyCode::Down => {
                            if let Some(selected) = app.history_list_state.selected() {
                                if selected < app.chat_history.len().saturating_sub(1) {
                                    app.history_list_state.select(Some(selected + 1));
                                }
                            }
                        }
                        KeyCode::Enter => {
                            let _ = app.load_selected_chat();
                        }
                        _ => {}
                    },
                    AppMode::ModelConfig => match key.code {
                        KeyCode::Esc => {
                            app.switch_mode(AppMode::Chat);
                        }
                        KeyCode::Up => {
                            app.prev_config_field();
                            app.config_input = app.get_current_config_value();
                        }
                        KeyCode::Down | KeyCode::Tab => {
                            app.next_config_field();
                            app.config_input = app.get_current_config_value();
                        }
                        KeyCode::Enter => {
                            let value = app.config_input.clone();
                            app.update_config_field(value);
                            let _ = app.save_config();
                            app.config_input.clear();
                        }
                        KeyCode::Char(c) => {
                            app.config_input.push(c);
                        }
                        KeyCode::Backspace => {
                            app.config_input.pop();
                        }
                        _ => {}
                    },
                }
            }
        }
    }
}

fn ui(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(0),
            Constraint::Length(3),
            Constraint::Length(1),
        ])
        .split(f.area());

    // Title bar
    let title = Paragraph::new(format!(
        "Ollama TUI Chat - Model: {} | Mode: {:?}",
        app.current_model, app.mode
    ))
    .style(
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    )
    .block(Block::default().borders(Borders::ALL));
    f.render_widget(title, chunks[0]);

    // Main content area
    match app.mode {
        AppMode::Chat => {
            render_chat(f, app, chunks[1]);
            render_input(f, app, chunks[2]);
        }
        AppMode::ModelSelection => {
            render_model_selection(f, app, chunks[1]);
        }
        AppMode::ModelDownload => {
            render_model_download(f, app, chunks[1]);
        }
        AppMode::SystemMonitor => {
            render_system_monitor(f, app, chunks[1]);
        }
        AppMode::ChatHistory => {
            render_chat_history(f, app, chunks[1]);
        }
        AppMode::ModelConfig => {
            render_model_config(f, app, chunks[1]);
        }
    }

    // Status bar
    let status =
        Paragraph::new(app.status_message.as_str()).style(Style::default().fg(Color::Yellow));
    f.render_widget(status, chunks[3]);
}

fn render_chat(f: &mut Frame, app: &App, area: Rect) {
    let mut text = Vec::new();

    for (i, (role, content)) in app.messages.iter().enumerate() {
        let style = if role == "user" {
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default()
                .fg(Color::Blue)
                .add_modifier(Modifier::BOLD)
        };

        // Check if this is the last message and we're thinking
        let is_last = i == app.messages.len() - 1;
        let is_thinking_message = is_last && app.is_thinking && content.is_empty();

        if is_thinking_message {
            text.push(Line::from(vec![
                Span::styled(format!("{}: ", role), style),
                Span::styled(
                    format!("{} Thinking...", app.get_thinking_spinner()),
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::ITALIC),
                ),
            ]));
        } else {
            text.push(Line::from(vec![Span::styled(format!("{}: ", role), style)]));
            if !content.is_empty() {
                text.push(Line::from(content.clone()));
            }
        }
        text.push(Line::from(""));
    }

    let messages_widget = Paragraph::new(text)
        .block(Block::default().borders(Borders::ALL).title("Chat"))
        .wrap(Wrap { trim: true })
        .scroll((app.scroll_offset as u16, 0));

    f.render_widget(messages_widget, area);
}

fn render_input(f: &mut Frame, app: &App, area: Rect) {
    let input = Paragraph::new(app.input.as_str())
        .style(Style::default().fg(Color::White))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Input (Press Enter to send)"),
        );
    f.render_widget(input, area);
}

fn render_model_selection(f: &mut Frame, app: &App, area: Rect) {
    let items: Vec<ListItem> = app
        .available_models
        .iter()
        .map(|model| {
            let style = if model == &app.current_model {
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            ListItem::new(model.as_str()).style(style)
        })
        .collect();

    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Select Model (Enter to select, Esc to cancel)"),
        )
        .highlight_style(
            Style::default()
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol(">> ");

    let mut state = app.model_list_state.clone();
    f.render_stateful_widget(list, area, &mut state);
}

fn render_model_download(f: &mut Frame, app: &App, area: Rect) {
    let download = Paragraph::new(app.download_input.as_str())
        .style(Style::default().fg(Color::White))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Download Model (Enter model name, e.g., 'llama2:latest')"),
        );
    f.render_widget(download, area);
}

fn render_system_monitor(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),
            Constraint::Length(4),
            Constraint::Length(5),
            Constraint::Min(0),
        ])
        .split(area);

    // CPU Usage with sleek design
    let cpu_percent = app.cpu_usage.min(100.0);
    let cpu_color = if cpu_percent > 80.0 {
        Color::Red
    } else if cpu_percent > 50.0 {
        Color::Yellow
    } else {
        Color::Cyan
    };

    let cpu_gauge = Gauge::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(Span::styled(
                    "━━━ CPU ━━━",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ))
                .border_style(Style::default().fg(Color::Cyan)),
        )
        .gauge_style(
            Style::default()
                .fg(cpu_color)
                .bg(Color::Black)
                .add_modifier(Modifier::BOLD),
        )
        .percent(cpu_percent as u16)
        .label(Span::styled(
            format!("{:.1}%", cpu_percent),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ));
    f.render_widget(cpu_gauge, chunks[0]);

    // Memory Usage with sleek design
    let memory_percent = if app.memory_total > 0 {
        ((app.memory_usage as f64 / app.memory_total as f64) * 100.0) as u16
    } else {
        0
    };
    let memory_gb_used = app.memory_usage as f64 / 1024.0 / 1024.0 / 1024.0;
    let memory_gb_total = app.memory_total as f64 / 1024.0 / 1024.0 / 1024.0;

    let mem_color = if memory_percent > 80 {
        Color::Red
    } else if memory_percent > 50 {
        Color::Yellow
    } else {
        Color::Magenta
    };

    let memory_gauge = Gauge::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(Span::styled(
                    "━━━ MEMORY ━━━",
                    Style::default()
                        .fg(Color::Magenta)
                        .add_modifier(Modifier::BOLD),
                ))
                .border_style(Style::default().fg(Color::Magenta)),
        )
        .gauge_style(
            Style::default()
                .fg(mem_color)
                .bg(Color::Black)
                .add_modifier(Modifier::BOLD),
        )
        .percent(memory_percent)
        .label(Span::styled(
            format!("{:.1} GB / {:.1} GB", memory_gb_used, memory_gb_total),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ));
    f.render_widget(memory_gauge, chunks[1]);

    // GPU Info with sleek design
    let gpu_lines = if let Some(ref gpu_info) = app.gpu_info {
        let parts: Vec<&str> = gpu_info.trim().split(',').collect();
        if parts.len() >= 4 {
            let gpu_util = parts[0].trim();
            let mem_used = parts[1].trim();
            let mem_total = parts[2].trim();
            let temp = parts[3].trim();
            vec![
                Line::from(vec![
                    Span::styled("  Utilization: ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        format!("{}%", gpu_util),
                        Style::default()
                            .fg(Color::Green)
                            .add_modifier(Modifier::BOLD),
                    ),
                ]),
                Line::from(vec![
                    Span::styled("  VRAM: ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        format!("{} / {} MB", mem_used, mem_total),
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    ),
                ]),
                Line::from(vec![
                    Span::styled("  Temperature: ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        format!("{}°C", temp),
                        Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                    ),
                ]),
            ]
        } else {
            vec![Line::from("GPU detected")]
        }
    } else {
        vec![Line::from(Span::styled(
            "  No GPU detected",
            Style::default().fg(Color::DarkGray),
        ))]
    };

    let gpu_widget = Paragraph::new(gpu_lines).block(
        Block::default()
            .borders(Borders::ALL)
            .title(Span::styled(
                "━━━ GPU ━━━",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ))
            .border_style(Style::default().fg(Color::Green)),
    );
    f.render_widget(gpu_widget, chunks[2]);

    // Top Processes Table
    let mut processes: Vec<_> = app.sys_info.processes().values().collect();
    processes.sort_by(|a, b| b.cpu_usage().partial_cmp(&a.cpu_usage()).unwrap());

    let process_rows: Vec<Row> = processes
        .iter()
        .skip(app.process_scroll)
        .take(15)
        .map(|p| {
            let cpu = format!("{:.1}%", p.cpu_usage());
            let mem = format!("{:.0} MB", p.memory() as f64 / 1024.0 / 1024.0);
            let name = p.name().to_string_lossy();
            Row::new(vec![name.to_string(), cpu, mem]).style(Style::default().fg(Color::White))
        })
        .collect();

    let process_table = Table::new(
        process_rows,
        [
            Constraint::Percentage(60),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
        ],
    )
    .header(
        Row::new(vec!["Process", "CPU", "Memory"])
            .style(
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            )
            .bottom_margin(1),
    )
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title(Span::styled(
                "━━━ TOP PROCESSES ━━━",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ))
            .border_style(Style::default().fg(Color::Yellow)),
    )
    .column_spacing(2);

    f.render_widget(process_table, chunks[3]);
}

fn render_chat_history(f: &mut Frame, app: &App, area: Rect) {
    let items: Vec<ListItem> = app
        .chat_history
        .iter()
        .map(|session| {
            let msg_count = session.messages.len();
            let preview = if let Some((_, content)) = session.messages.first() {
                let preview_text = content.chars().take(50).collect::<String>();
                format!(
                    "{} - {} msgs - {}",
                    session.timestamp, msg_count, preview_text
                )
            } else {
                format!("{} - {} msgs", session.timestamp, msg_count)
            };
            ListItem::new(preview).style(Style::default().fg(Color::White))
        })
        .collect();

    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Chat History (Enter to load, Esc to cancel)"),
        )
        .highlight_style(
            Style::default()
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol(">> ");

    let mut state = app.history_list_state.clone();
    f.render_stateful_widget(list, area, &mut state);
}

fn render_model_config(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(3)])
        .split(area);

    // Configuration fields
    let config_items = vec![
        Line::from(vec![
            Span::styled(
                "  Temperature ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("[{}]", app.model_config.temperature),
                if matches!(app.config_field, ConfigField::Temperature) {
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::White)
                },
            ),
        ]),
        Line::from("    Controls randomness. Lower = more focused, Higher = more creative"),
        Line::from("    Range: 0.0 - 2.0, Default: 0.8"),
        Line::from(""),
        Line::from(vec![
            Span::styled(
                "  Top P ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("[{}]", app.model_config.top_p),
                if matches!(app.config_field, ConfigField::TopP) {
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::White)
                },
            ),
        ]),
        Line::from("    Nucleus sampling. Controls diversity of responses"),
        Line::from("    Range: 0.0 - 1.0, Default: 0.9"),
        Line::from(""),
        Line::from(vec![
            Span::styled(
                "  Top K ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("[{}]", app.model_config.top_k),
                if matches!(app.config_field, ConfigField::TopK) {
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::White)
                },
            ),
        ]),
        Line::from("    Limits token selection to top K options"),
        Line::from("    Range: 1+, Default: 40"),
        Line::from(""),
        Line::from(vec![
            Span::styled(
                "  Repeat Penalty ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("[{}]", app.model_config.repeat_penalty),
                if matches!(app.config_field, ConfigField::RepeatPenalty) {
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::White)
                },
            ),
        ]),
        Line::from("    Penalizes repetition. Higher = less repetition"),
        Line::from("    Range: 0.0 - 2.0, Default: 1.1"),
        Line::from(""),
        Line::from(vec![
            Span::styled(
                "  Context Window ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("[{}]", app.model_config.num_ctx),
                if matches!(app.config_field, ConfigField::ContextWindow) {
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::White)
                },
            ),
        ]),
        Line::from("    Number of tokens in context window"),
        Line::from("    Range: 512 - 32768, Default: 2048"),
        Line::from(""),
        Line::from(vec![
            Span::styled(
                "  System Prompt ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(
                    "[{}]",
                    if app.model_config.system_prompt.len() > 30 {
                        format!("{}...", &app.model_config.system_prompt[..30])
                    } else {
                        app.model_config.system_prompt.clone()
                    }
                ),
                if matches!(app.config_field, ConfigField::SystemPrompt) {
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::White)
                },
            ),
        ]),
        Line::from("    System instructions for the model"),
        Line::from(""),
        Line::from(""),
        Line::from(Span::styled(
            "Navigation: Up/Down or Tab | Edit: Type value & Enter | Save: Auto | Esc: Back",
            Style::default().fg(Color::Green),
        )),
    ];

    let config_widget = Paragraph::new(config_items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(Span::styled(
                    "━━━ MODEL CONFIGURATION ━━━",
                    Style::default()
                        .fg(Color::Magenta)
                        .add_modifier(Modifier::BOLD),
                ))
                .border_style(Style::default().fg(Color::Magenta)),
        )
        .wrap(Wrap { trim: false });

    f.render_widget(config_widget, chunks[0]);

    // Input field for current selection
    let field_name = match app.config_field {
        ConfigField::Temperature => "Temperature",
        ConfigField::TopP => "Top P",
        ConfigField::TopK => "Top K",
        ConfigField::RepeatPenalty => "Repeat Penalty",
        ConfigField::ContextWindow => "Context Window",
        ConfigField::SystemPrompt => "System Prompt",
    };

    let input = Paragraph::new(app.config_input.as_str())
        .style(Style::default().fg(Color::White))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!("Editing: {} (Press Enter to save)", field_name))
                .border_style(Style::default().fg(Color::Yellow)),
        );
    f.render_widget(input, chunks[1]);
}
