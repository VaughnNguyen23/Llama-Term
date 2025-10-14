# Ollama TUI Chat 🦙💬

[![Rust CI](https://github.com/yourusername/ollama-tui/actions/workflows/rust.yml/badge.svg)](https://github.com/yourusername/ollama-tui/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org)

A powerful, feature-rich terminal user interface (TUI) for interacting with Ollama models. Built with Rust for performance and reliability.

![Ollama TUI Demo](https://via.placeholder.com/800x400.png?text=Ollama+TUI+Demo+Screenshot)

> ⚡ Real-time streaming • 🎛️ Model fine-tuning • 📊 System monitoring • 💾 Session management

## Features

- **Interactive Chat**: Chat with any Ollama model in a clean TUI
- **Real-time Streaming**: Tokens appear as they're generated from the LLM
- **Animated Spinner**: Smooth thinking animation while waiting for responses
- **Non-blocking UI**: The interface stays responsive during generation
- **Model Fine-tuning**: Configure temperature, top_p, top_k, context window, system prompts, and more
- **Chat Management**: Save, load, and clear chat sessions
- **Copy/Paste Support**: Select and copy messages to clipboard
- **System Monitor**: Real-time CPU, Memory, GPU monitoring with top processes
- **Model Management**: Switch between installed models on the fly
- **Model Downloads**: Download new models directly from the TUI
- **Docker Support**: Run Ollama in a Docker container with GPU acceleration
- **Keyboard Navigation**: Fully keyboard-driven interface

## Prerequisites

- Rust (2021 edition or later)
- Docker and Docker Compose (for containerized Ollama)
- Or Ollama installed locally

## Quick Start

### 1. Start Ollama (Docker)

```bash
# Start Ollama container
docker-compose up -d

# Pull a model (do this once)
docker exec -it ollama ollama pull llama2:latest
```

### 2. Build and Run the TUI

```bash
# Build the application
cargo build --release

# Run the application
cargo run --release
```

## Usage

### Keyboard Shortcuts

**Chat Mode:**
- Type your message and press `Enter` to send
- `Up/Down` - Scroll through chat history
- `F1` - Show help
- `F2` - Open model selection
- `F3` - Download new model
- `F4` - Open system monitor
- `F5` - Browse chat history
- `F6` - Save current chat
- `F7` - Clear current chat
- `F8` - Open model configuration
- `Ctrl+S` - Select last message
- `Ctrl+Y` - Copy selected message to clipboard
- `Ctrl+C` - Quit application

**Model Selection Mode:**
- `Up/Down` - Navigate models
- `Enter` - Select model
- `Esc` - Return to chat

**Model Download Mode:**
- Type the model name (e.g., `llama2:latest`, `mistral:latest`)
- `Enter` - Start download
- `Esc` - Cancel

**System Monitor Mode:**
- Shows real-time CPU, Memory, GPU stats and top processes
- `Up/Down` - Scroll through process list
- Updates every 100ms
- `Esc` - Return to chat

**Chat History Mode:**
- `Up/Down` - Navigate saved chats
- `Enter` - Load selected chat
- `Esc` - Return to chat

**Model Configuration Mode (F8):**
- `Up/Down` or `Tab` - Navigate between fields
- Type value and press `Enter` - Update field
- Auto-saves on Enter
- `Esc` - Return to chat

### Configurable Parameters

- **Temperature** (0.0-2.0): Controls randomness. Lower = more focused, Higher = more creative
- **Top P** (0.0-1.0): Nucleus sampling for diversity control
- **Top K** (1+): Limits token selection to top K options
- **Repeat Penalty** (0.0-2.0): Penalizes repetition. Higher = less repetition
- **Context Window** (512-32768): Number of tokens in context
- **System Prompt**: Custom instructions for the model's behavior

## Docker Configuration

The `docker-compose.yml` file sets up Ollama with:
- Port 11434 exposed for API access
- Persistent volume for model storage
- **GPU support enabled by default** (uses all available NVIDIA GPUs)

### GPU Support

GPU support is **already enabled** in the docker-compose.yml file. To use it, you need:

1. **NVIDIA GPU** in your system
2. **NVIDIA drivers** installed on your host
3. **NVIDIA Container Toolkit** installed

#### Installing NVIDIA Container Toolkit

```bash
# For Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### Verify GPU is Working

```bash
# Check if GPU is available in container
docker exec -it ollama nvidia-smi

# You should see your GPU listed
```

If you **don't have a GPU** or want to use CPU only, comment out the `deploy` section in `docker-compose.yml`.

### Managing Docker Ollama

```bash
# View logs
docker-compose logs -f

# Stop Ollama
docker-compose down

# Stop and remove volumes (deletes downloaded models)
docker-compose down -v

# Pull models manually
docker exec -it ollama ollama pull <model-name>

# List installed models
docker exec -it ollama ollama list
```

## Connecting to Remote Ollama

If Ollama is running on a different host, modify the Ollama connection in `src/main.rs`:

```rust
let ollama = Ollama::new("http://your-host:11434".to_string());
```

## Popular Models to Try

- `llama2:latest` - Meta's Llama 2
- `mistral:latest` - Mistral AI model
- `codellama:latest` - Code-focused Llama
- `deepseek-coder:latest` - DeepSeek code model
- `phi:latest` - Microsoft's Phi model
- `gemma:latest` - Google's Gemma model

## Troubleshooting

**Connection refused error:**
- Ensure Ollama is running: `docker-compose ps`
- Check if port 11434 is accessible: `curl http://localhost:11434/api/tags`

**Model not found:**
- Pull the model first: `docker exec -it ollama ollama pull <model-name>`
- Or use F3 in the TUI to download

**Slow responses:**
- Larger models require more resources
- Try smaller models like `phi:latest` for faster responses
- Consider GPU support for better performance

## File Storage

- **Chat sessions**: `~/.ollama_tui/chats/` - Saved when you press F6
- **Model config**: `~/.ollama_tui/model_config.json` - Auto-saved when you edit settings

Each chat session includes timestamp, model used, and all messages.
Model configuration persists across sessions and is automatically loaded on startup.

## Building for Production

```bash
# Build optimized binary
cargo build --release

# Binary location
./target/release/ollama_testing
```

## Screenshots

*Coming soon - Add screenshots of your TUI in action!*

## Roadmap

- [ ] Multi-model conversation support
- [ ] Export chats to markdown/PDF
- [ ] Custom themes and color schemes
- [ ] Plugin system for extensions
- [ ] Voice input/output support
- [ ] RAG (Retrieval Augmented Generation) support
- [ ] Model comparison mode

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a history of changes to this project.

## Acknowledgments

- [Ollama](https://ollama.ai/) - For the amazing local LLM platform
- [Ratatui](https://github.com/ratatui-org/ratatui) - For the excellent TUI framework
- [Tokio](https://tokio.rs/) - For the async runtime
- All contributors and users of this project

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/ollama-tui&type=Date)](https://star-history.com/#yourusername/ollama-tui&Date)

## Support

If you encounter any issues or have questions:
- 🐛 [Report a bug](https://github.com/yourusername/ollama-tui/issues/new?template=bug_report.md)
- 💡 [Request a feature](https://github.com/yourusername/ollama-tui/issues/new?template=feature_request.md)
- 💬 [Start a discussion](https://github.com/yourusername/ollama-tui/discussions)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  Made with ❤️ by the Ollama TUI community
  <br>
  <sub>If you find this project useful, please consider giving it a ⭐!</sub>
</div>
