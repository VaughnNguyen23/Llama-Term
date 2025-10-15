pub mod app;
pub mod ui;

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use ratatui::{Terminal, backend::Backend};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

use crate::app::{App, AppMode};
use crate::ui::ui;

pub async fn run_app<B: Backend>(
    terminal: &mut Terminal<B>,
    app_arc: Arc<Mutex<App>>,
) -> Result<()> {
    loop {
        {
            let app = app_arc.lock().await;
            terminal.draw(|f| ui(f, &app))?;
        }

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

                // Vim-like key handling pre-processing for Chat mode
                if app.mode == AppMode::Chat && app.vim_mode {
                    // Esc/i to switch modes
                    if let KeyCode::Esc = key.code {
                        app.vim_insert = false;
                        app.pending_g = false;
                        app.status_message = "Normal mode".into();
                        continue;
                    }
                    if matches!(key.code, KeyCode::Char('i')) && key.modifiers.is_empty() && !app.vim_insert {
                        app.vim_insert = true;
                        app.status_message = "Insert mode".into();
                        continue;
                    }

                    if !app.vim_insert {
                        match key.code {
                            KeyCode::Char('j') => { app.scroll_down(); continue; }
                            KeyCode::Char('k') => { app.scroll_up(); continue; }
                            KeyCode::Char('g') => {
                                if app.pending_g { app.scroll_top(); app.pending_g = false; } else { app.pending_g = true; }
                                continue;
                            }
                            KeyCode::Char('G') => { app.scroll_bottom(); continue; }
                            // g-prefixed shortcuts for mode switching
                            KeyCode::Char('m') if app.pending_g => { let _ = app.fetch_models().await; app.switch_mode(AppMode::ModelSelection); app.pending_g = false; continue; }
                            KeyCode::Char('d') if app.pending_g => { app.switch_mode(AppMode::ModelDownload); app.pending_g = false; continue; }
                            KeyCode::Char('s') if app.pending_g => { app.update_system_info(); app.switch_mode(AppMode::SystemMonitor); app.pending_g = false; continue; }
                            KeyCode::Char('h') if app.pending_g => { let _ = app.load_chat_history(); app.history_list_state.select(Some(0)); app.switch_mode(AppMode::ChatHistory); app.pending_g = false; continue; }
                            KeyCode::Char('c') if app.pending_g => { app.config_input = app.get_current_config_value(); app.switch_mode(AppMode::ModelConfig); app.pending_g = false; continue; }
                            KeyCode::Char('w') => { let _ = app.save_current_chat(); continue; }
                            _ => { app.pending_g = false; }
                        }
                    }
                }

                match app.mode {
                    AppMode::Chat => match key.code {
                        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            return Ok(());
                        }
                        KeyCode::F(1) => {
                            app.status_message = "Vim: Esc/i modes | j/k scroll | gg top | G bottom | gm models | gd download | gs monitor | gh history | gc config | gw save | Enter send | Ctrl+C quit".to_string();
                        }
                        KeyCode::F(2) => { let _ = app.fetch_models().await; app.switch_mode(AppMode::ModelSelection); }
                        KeyCode::F(3) => { app.switch_mode(AppMode::ModelDownload); }
                        KeyCode::F(4) => { app.update_system_info(); app.switch_mode(AppMode::SystemMonitor); }
                        KeyCode::F(5) => { let _ = app.load_chat_history(); app.history_list_state.select(Some(0)); app.switch_mode(AppMode::ChatHistory); }
                        KeyCode::F(6) => { let _ = app.save_current_chat(); }
                        KeyCode::F(7) => { app.clear_chat(); }
                        KeyCode::F(8) => { app.config_input = app.get_current_config_value(); app.switch_mode(AppMode::ModelConfig); }
                        KeyCode::Char('s') if key.modifiers.contains(KeyModifiers::CONTROL) => { app.select_last_message(); }
                        KeyCode::Char('y') if key.modifiers.contains(KeyModifiers::CONTROL) => { app.copy_to_clipboard(); }
                        KeyCode::Enter => { app.start_message_stream(Arc::clone(&app_arc)); }
                        KeyCode::Char(c) => { app.input.push(c); }
                        KeyCode::Backspace => { app.input.pop(); }
                        KeyCode::Up => { app.scroll_up(); }
                        KeyCode::Down => { app.scroll_down(); }
                        _ => {}
                    },
                    AppMode::ModelSelection => match key.code {
                        KeyCode::Esc => { app.switch_mode(AppMode::Chat); }
                        KeyCode::Up => { if let Some(selected) = app.model_list_state.selected() { if selected > 0 { app.model_list_state.select(Some(selected - 1)); } } }
                        KeyCode::Down => { if let Some(selected) = app.model_list_state.selected() { if selected < app.available_models.len().saturating_sub(1) { app.model_list_state.select(Some(selected + 1)); } } }
                        KeyCode::Enter => { if let Some(selected) = app.model_list_state.selected() { if let Some(model) = app.available_models.get(selected).cloned() { app.current_model = model.clone(); app.status_message = format!("Model changed to: {}", model); app.switch_mode(AppMode::Chat); } } }
                        _ => {}
                    },
                    AppMode::ModelDownload => match key.code {
                        KeyCode::Esc => { app.download_input.clear(); app.switch_mode(AppMode::Chat); }
                        KeyCode::Enter => { let model_name = app.download_input.clone(); app.download_input.clear(); let _ = app.download_model(model_name).await; app.switch_mode(AppMode::Chat); }
                        KeyCode::Char(c) => { app.download_input.push(c); }
                        KeyCode::Backspace => { app.download_input.pop(); }
                        _ => {}
                    },
                    AppMode::SystemMonitor => match key.code {
                        KeyCode::Esc => { app.switch_mode(AppMode::Chat); }
                        KeyCode::Up => { if app.process_scroll > 0 { app.process_scroll -= 1; } }
                        KeyCode::Down => { app.process_scroll += 1; }
                        _ => {}
                    },
                    AppMode::ChatHistory => match key.code {
                        KeyCode::Esc => { app.switch_mode(AppMode::Chat); }
                        KeyCode::Up => { if let Some(selected) = app.history_list_state.selected() { if selected > 0 { app.history_list_state.select(Some(selected - 1)); } } }
                        KeyCode::Down => { if let Some(selected) = app.history_list_state.selected() { if selected < app.chat_history.len().saturating_sub(1) { app.history_list_state.select(Some(selected + 1)); } } }
                        KeyCode::Enter => { let _ = app.load_selected_chat(); }
                        _ => {}
                    },
                    AppMode::ModelConfig => match key.code {
                        KeyCode::Esc => { app.switch_mode(AppMode::Chat); }
                        KeyCode::Up => { app.prev_config_field(); app.config_input = app.get_current_config_value(); }
                        KeyCode::Down | KeyCode::Tab => { app.next_config_field(); app.config_input = app.get_current_config_value(); }
                        KeyCode::Enter => { let value = app.config_input.clone(); app.update_config_field(value); let _ = app.save_config(); app.config_input.clear(); }
                        KeyCode::Char(c) => { app.config_input.push(c); }
                        KeyCode::Backspace => { app.config_input.pop(); }
                        _ => {}
                    },
                }
            }
        }
    }
}
