use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Row, Table, Wrap, ListState},
};

use crate::app::{App, AppMode, ConfigField};

pub fn ui(f: &mut Frame, app: &App) {
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
    .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
    .block(Block::default().borders(Borders::ALL));
    f.render_widget(title, chunks[0]);

    match app.mode {
        AppMode::Chat => { render_chat(f, app, chunks[1]); render_input(f, app, chunks[2]); }
        AppMode::ModelSelection => { render_model_selection(f, app, chunks[1]); }
        AppMode::ModelDownload => { render_model_download(f, app, chunks[1]); }
        AppMode::SystemMonitor => { render_system_monitor(f, app, chunks[1]); }
        AppMode::ChatHistory => { render_chat_history(f, app, chunks[1]); }
        AppMode::ModelConfig => { render_model_config(f, app, chunks[1]); }
    }

    let status = Paragraph::new(app.status_message.as_str()).style(Style::default().fg(Color::Yellow));
    f.render_widget(status, chunks[3]);
}

fn render_chat(f: &mut Frame, app: &App, area: Rect) {
    let mut text = Vec::new();

    for (i, (role, content)) in app.messages.iter().enumerate() {
        let style = if role == "user" {
            Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Blue).add_modifier(Modifier::BOLD)
        };

        // Check if this is the last message and we're thinking
        let is_last = i == app.messages.len() - 1;
        let is_thinking_message = is_last && app.is_thinking && content.is_empty();

        if is_thinking_message {
            text.push(Line::from(vec![
                Span::styled(format!("{}: ", role), style),
                Span::styled(
                    format!("{} Thinking...", app.get_thinking_spinner()),
                    Style::default().fg(Color::Yellow).add_modifier(Modifier::ITALIC),
                ),
            ]));
        } else {
            text.push(Line::from(vec![Span::styled(format!("{}: ", role), style)]));
            if !content.is_empty() { text.push(Line::from(content.clone())); }
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
        .block(Block::default().borders(Borders::ALL).title("Input (Press Enter to send)"));
    f.render_widget(input, area);
}

fn render_model_selection(f: &mut Frame, app: &App, area: Rect) {
    let items: Vec<ListItem> = app
        .available_models
        .iter()
        .map(|model| {
            let style = if model == &app.current_model {
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)
            } else { Style::default() };
            ListItem::new(model.as_str()).style(style)
        })
        .collect();

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title("Select Model (Enter to select, Esc to cancel)"))
        .highlight_style(Style::default().bg(Color::DarkGray).add_modifier(Modifier::BOLD))
        .highlight_symbol(">> ");

    let mut state = app.model_list_state.clone();
    f.render_stateful_widget(list, area, &mut state);
}

fn render_model_download(f: &mut Frame, app: &App, area: Rect) {
    let download = Paragraph::new(app.download_input.as_str())
        .style(Style::default().fg(Color::White))
        .block(Block::default().borders(Borders::ALL).title("Download Model (Enter model name, e.g., 'llama2:latest')"));
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

    // CPU
    let cpu_percent = app.cpu_usage.min(100.0);
    let cpu_color = if cpu_percent > 80.0 { Color::Red } else if cpu_percent > 50.0 { Color::Yellow } else { Color::Cyan };
    let cpu_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(Span::styled("━━━ CPU ━━━", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))).border_style(Style::default().fg(Color::Cyan)))
        .gauge_style(Style::default().fg(cpu_color).bg(Color::Black).add_modifier(Modifier::BOLD))
        .percent(cpu_percent as u16)
        .label(Span::styled(format!("{:.1}%", cpu_percent), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)));
    f.render_widget(cpu_gauge, chunks[0]);

    // Memory
    let memory_percent = if app.memory_total > 0 { ((app.memory_usage as f64 / app.memory_total as f64) * 100.0) as u16 } else { 0 };
    let memory_gb_used = app.memory_usage as f64 / 1024.0 / 1024.0 / 1024.0;
    let memory_gb_total = app.memory_total as f64 / 1024.0 / 1024.0 / 1024.0;
    let mem_color = if memory_percent > 80 { Color::Red } else if memory_percent > 50 { Color::Yellow } else { Color::Magenta };
    let memory_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(Span::styled("━━━ MEMORY ━━━", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD))).border_style(Style::default().fg(Color::Magenta)))
        .gauge_style(Style::default().fg(mem_color).bg(Color::Black).add_modifier(Modifier::BOLD))
        .percent(memory_percent)
        .label(Span::styled(format!("{:.1} GB / {:.1} GB", memory_gb_used, memory_gb_total), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)));
    f.render_widget(memory_gauge, chunks[1]);

    // GPU
    let gpu_lines = if let Some(ref gpu_info) = app.gpu_info {
        let parts: Vec<&str> = gpu_info.trim().split(',').collect();
        if parts.len() >= 4 {
            let gpu_util = parts[0].trim();
            let mem_used = parts[1].trim();
            let mem_total = parts[2].trim();
            let temp = parts[3].trim();
            vec![
                Line::from(vec![Span::styled("  Utilization: ", Style::default().fg(Color::Gray)), Span::styled(format!("{}%", gpu_util), Style::default().fg(Color::Green).add_modifier(Modifier::BOLD))]),
                Line::from(vec![Span::styled("  VRAM: ", Style::default().fg(Color::Gray)), Span::styled(format!("{} / {} MB", mem_used, mem_total), Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))]),
                Line::from(vec![Span::styled("  Temperature: ", Style::default().fg(Color::Gray)), Span::styled(format!("{}°C", temp), Style::default().fg(Color::Red).add_modifier(Modifier::BOLD))]),
            ]
        } else { vec![Line::from("GPU detected")] }
    } else { vec![Line::from(Span::styled("  No GPU detected", Style::default().fg(Color::DarkGray)))] };

    let gpu_widget = Paragraph::new(gpu_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(Span::styled("━━━ GPU ━━━", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)))
                .border_style(Style::default().fg(Color::Green)),
        );
    f.render_widget(gpu_widget, chunks[2]);

    // Top Processes
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
        [Constraint::Percentage(60), Constraint::Percentage(20), Constraint::Percentage(20)],
    )
    .header(
        Row::new(vec!["Process", "CPU", "Memory"]).style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)).bottom_margin(1),
    )
    .block(
        Block::default().borders(Borders::ALL).title(Span::styled("━━━ TOP PROCESSES ━━━", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))).border_style(Style::default().fg(Color::Yellow)),
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
                format!("{} - {} msgs - {}", session.timestamp, msg_count, preview_text)
            } else { format!("{} - {} msgs", session.timestamp, msg_count) };
            ListItem::new(preview).style(Style::default().fg(Color::White))
        })
        .collect();

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title("Chat History (Enter to load, Esc to cancel)"))
        .highlight_style(Style::default().bg(Color::DarkGray).add_modifier(Modifier::BOLD))
        .highlight_symbol(">> ");

    let mut state = app.history_list_state.clone();
    f.render_stateful_widget(list, area, &mut state);
}

fn render_model_config(f: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(3)])
        .split(area);

    // Fields
    let config_items = vec![
        // Temperature
        Line::from(vec![
            Span::styled("  Temperature ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::styled(
                format!("[{}]", app.model_config.temperature),
                if matches!(app.config_field, ConfigField::Temperature) { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::White) },
            ),
        ]),
        Line::from("    Controls randomness. Lower = more focused, Higher = more creative"),
        Line::from("    Range: 0.0 - 2.0, Default: 0.8"),
        Line::from(""),
        // Top P
        Line::from(vec![
            Span::styled("  Top P ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::styled(
                format!("[{}]", app.model_config.top_p),
                if matches!(app.config_field, ConfigField::TopP) { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::White) },
            ),
        ]),
        Line::from("    Nucleus sampling. Controls diversity of responses"),
        Line::from("    Range: 0.0 - 1.0, Default: 0.9"),
        Line::from(""),
        // Top K
        Line::from(vec![
            Span::styled("  Top K ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::styled(
                format!("[{}]", app.model_config.top_k),
                if matches!(app.config_field, ConfigField::TopK) { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::White) },
            ),
        ]),
        Line::from("    Limits token selection to top K options"),
        Line::from("    Range: 1+, Default: 40"),
        Line::from(""),
        // Repeat Penalty
        Line::from(vec![
            Span::styled("  Repeat Penalty ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::styled(
                format!("[{}]", app.model_config.repeat_penalty),
                if matches!(app.config_field, ConfigField::RepeatPenalty) { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::White) },
            ),
        ]),
        Line::from("    Penalizes repetition. Higher = less repetition"),
        Line::from("    Range: 0.0 - 2.0, Default: 1.1"),
        Line::from(""),
        // Context Window
        Line::from(vec![
            Span::styled("  Context Window ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::styled(
                format!("[{}]", app.model_config.num_ctx),
                if matches!(app.config_field, ConfigField::ContextWindow) { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::White) },
            ),
        ]),
        Line::from("    Number of tokens in context window"),
        Line::from("    Range: 512 - 32768, Default: 2048"),
        Line::from(""),
        // System Prompt
        Line::from(vec![
            Span::styled("  System Prompt ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::styled(
                format!(
                    "[{}]",
                    if app.model_config.system_prompt.len() > 30 { format!("{}...", &app.model_config.system_prompt[..30]) } else { app.model_config.system_prompt.clone() }
                ),
                if matches!(app.config_field, ConfigField::SystemPrompt) { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::White) },
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
        .block(Block::default().borders(Borders::ALL).title(Span::styled("━━━ MODEL CONFIGURATION ━━━", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD))).border_style(Style::default().fg(Color::Magenta)))
        .wrap(Wrap { trim: false });

    f.render_widget(config_widget, chunks[0]);

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
        .block(Block::default().borders(Borders::ALL).title(format!("Editing: {} (Press Enter to save)", field_name)).border_style(Style::default().fg(Color::Yellow)));
    f.render_widget(input, chunks[1]);
}
