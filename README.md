# AI Chat Log Analyzer

A Python tool that analyzes chat logs between users and AI assistants, extracting conversation statistics, keywords, and main topics using natural language processing techniques.

## Features

- **Message Statistics**: Count total messages, user messages, and AI responses
- **Keyword Extraction**: Identify the most common keywords using TF-IDF analysis
- **Topic Detection**: Determine the main conversation topic using bigram analysis
- **Batch Processing**: Analyze multiple chat log files in a folder at once
- **Clean Output**: Generate human-readable summaries for each conversation

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/chat-log-analyzer.git
cd chat-log-analyzer
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Chat Log Format

Your chat log files should be in `.txt` format with the following structure:

```
User: Hello!
AI: Hi! How can I assist you today?
User: Can you explain what machine learning is?
AI: Certainly! Machine learning is a field of AI that allows systems to
learn from data.
```

**Important**: Each speaker (User/AI) should be followed by a colon, and messages should be clearly separated.

## Usage

### Basic Usage

1. Place your chat log files (`.txt` format) in the `chat_logs` folder
2. Run the analyzer:

```bash
python task.py
```

## Sample Output

```
Processing chat logs in folder: ./chat_logs

Summary for 'chat.txt':
Summary:
- The conversation had 4 exchanges.(user:2, AI:2)
- The user asked mainly about machine learning.
- Most common keywords: ai, allows, data, learning, machine.
```

## Project Structure

```
chat-log-analyzer/
├── task.py              # Main analyzer script
├── requirements.txt     # Python dependencies
├── chat_logs/          # Folder for chat log files
│   ├── conversation1.txt
│   ├── conversation2.txt
│   └── ...
└── README.md           # This file
```

## How It Works

1. **Parsing**: The tool uses regex patterns to extract messages from each speaker
2. **Text Preprocessing**: Messages are cleaned and combined for analysis
3. **Keyword Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency) identifies important terms
4. **Topic Detection**: Bigram analysis finds the most frequent two-word phrases to determine main topics
5. **Summary Generation**: Creates a structured summary with statistics and insights

## Dependencies

- `nltk`: Natural Language Toolkit for text processing and stopwords
- `scikit-learn`: Machine learning library for TF-IDF and text vectorization
- `re`: Regular expressions (built-in)
- `os`: Operating system interface (built-in)
- `typing`: Type hints (built-in)

## Customization

You can modify the analysis by adjusting these parameters in the code:

- `top_k` in `extract_keywords_and_topic()`: Number of top keywords to extract (default: 5)
- `CHAT_LOGS_FOLDER`: Default folder path for chat logs
- Stopwords: Add custom stopwords to the `STOP_WORDS` list

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).

## Support

If you encounter any issues or have questions, please open an issue on the GitHub repository.

---

**Note**: This tool is designed for analyzing structured chat logs. For best results, ensure your chat logs follow the specified format with clear User/AI speaker labels.
