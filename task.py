import os
import re
from typing import Dict, List, Tuple

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Load English stopwords from NLTK
nltk.download('stopwords', quiet=True)
STOP_WORDS = stopwords.words('english')

def parse_chat_log(file_path: str) -> Dict[str, List[str]]:
    """
    Parse chat log file to extract messages grouped by speaker (User or AI).
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex matches each message block by User or AI until next speaker or EOF
    pattern = re.compile(r"(User:|AI:)(.*?)(?=(User:|AI:|$))", re.DOTALL)
    matches = pattern.findall(content)

    chat_data = {'User': [], 'AI': []}

    for speaker_tag, message, _ in matches:
        speaker = speaker_tag.rstrip(':')
        # Clean up whitespace/newlines inside messages
        cleaned_message = ' '.join(message.strip().split())
        if cleaned_message:
            chat_data[speaker].append(cleaned_message)

    return chat_data

def message_statistics(chat_data: Dict[str, List[str]]) -> Dict[str, int]:
    """
    Calculate total and per-speaker message counts.
    """
    user_count = len(chat_data.get('User', []))
    ai_count = len(chat_data.get('AI', []))
    total_count = user_count + ai_count

    return {
        'total_messages': total_count,
        'user_messages': user_count,
        'ai_messages': ai_count
    }

def extract_keywords(chat_data: Dict[str, List[str]], top_n: int = 5) -> List[str]:
    """
    Extract top N keywords from combined chat messages using TF-IDF.
    """
    combined_text = ' '.join(chat_data.get('User', []) + chat_data.get('AI', []))

    vectorizer = TfidfVectorizer(stop_words=STOP_WORDS, max_features=top_n)
    tfidf_matrix = vectorizer.fit_transform([combined_text])

    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]

    keyword_score_pairs = list(zip(feature_names, scores))
    # Sort descending by TF-IDF score
    keyword_score_pairs.sort(key=lambda x: x[1], reverse=True)

    return [kw for kw, _ in keyword_score_pairs]

def generate_summary(chat_data: Dict[str, List[str]]) -> str:
    """
    Generate a human-readable summary of the chat log.
    """
    stats = message_statistics(chat_data)
    keywords = extract_keywords(chat_data, top_n=5)

    keyword_str = ', '.join(keywords) if keywords else 'No significant keywords found'

    summary_lines = [
        "Summary:",
        f"- The conversation had {stats['total_messages']} exchanges.",
        f"- The user sent {stats['user_messages']} messages; the AI sent {stats['ai_messages']} messages.",
        f"- Most common keywords: {keyword_str}.",
    ]

    if keywords:
        summary_lines.append(f"- The user asked mainly about {keywords[0]} and related topics.")

    return '\n'.join(summary_lines)

def summarize_all_chats(folder_path: str) -> str:
    """
    Process and summarize all chat log files (.txt) in a folder.
    """
    if not os.path.isdir(folder_path):
        return f"Error: Provided path '{folder_path}' is not a valid directory."

    chat_files = sorted(f for f in os.listdir(folder_path) if f.lower().endswith('.txt'))

    if not chat_files:
        return f"No .txt chat log files found in directory '{folder_path}'."

    summaries = []
    for filename in chat_files:
        full_path = os.path.join(folder_path, filename)
        chat_data = parse_chat_log(full_path)
        summary = generate_summary(chat_data)
        header = f"\nSummary for '{filename}':\n"
        summaries.append(header)
        summaries.append(summary)

    return '\n\n'.join(summaries)

if __name__ == '__main__':
    # folder path -  chat logs folder
    CHAT_LOGS_FOLDER = "./chat_logs"
    # Ensure the folder exists
    if not os.path.exists(CHAT_LOGS_FOLDER):
        print(f"Error: The folder '{CHAT_LOGS_FOLDER}' does not exist.")
    else:
        print(f"Processing chat logs in folder: {CHAT_LOGS_FOLDER}")
        print(summarize_all_chats(CHAT_LOGS_FOLDER))
