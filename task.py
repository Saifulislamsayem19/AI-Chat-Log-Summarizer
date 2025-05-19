import os
import re
from typing import Dict, List, Tuple

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Load English stopwords from NLTK
nltk.download('stopwords', quiet=True)
STOP_WORDS = stopwords.words('english')

def parse_chat_log(file_path: str) -> Dict[str, List[str]]:
    """
    Parse a chat log file to extract messages grouped by speaker (User or AI).
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex matches each message block by User or AI until the next speaker tag or EOF
    pattern = re.compile(r"(User:|AI:)(.*?)(?=(User:|AI:|$))", re.DOTALL)
    matches = pattern.findall(content)

    chat_data = {'User': [], 'AI': []}

    for speaker_tag, message, _ in matches:
        speaker = speaker_tag.rstrip(':')   
        # Clean up whitespace/newlines inside messages                    
        cleaned = ' '.join(message.strip().split())            
        if cleaned:
            chat_data[speaker].append(cleaned)

    return chat_data

def message_statistics(chat_data: Dict[str, List[str]]) -> Tuple[int, int, int]:
    """
    Calculate total, user, and AI message counts.
    """
    user_count = len(chat_data.get('User', []))
    ai_count   = len(chat_data.get('AI', []))
    total      = user_count + ai_count
    return total, user_count, ai_count

def extract_keywords_and_topic(
    texts: List[str],
    top_k: int = 5
) -> Tuple[List[str], str]:
    """
    Extract top N keywords from combined chat messages using TF-IDF.
    """
    if not texts:
        return [], ''

    # Combine all messages into one string
    full_text = ' '.join(texts)

    # TF-IDF vectorizer for top unigrams
    tfidf = TfidfVectorizer(
        stop_words=STOP_WORDS,
        lowercase=True,
        token_pattern=r'\b[a-zA-Z]{2,}\b',
        max_features=top_k
    )
    tfidf.fit([full_text])
    keywords = list(tfidf.get_feature_names_out())

    # CountVectorizer for bigrams to approximate user topic
    cv = CountVectorizer(
        stop_words=STOP_WORDS,
        lowercase=True,
        ngram_range=(2, 2),
        token_pattern=r'\b[a-zA-Z]{2,}\b'
    )
    # Only use the same combined text to find bigram frequencies
    bigram_matrix = cv.fit_transform([full_text])
    bigram_counts = bigram_matrix.toarray()[0]
    bigram_features = cv.get_feature_names_out()

    if bigram_counts.sum() > 0:
        # Select the most frequent bigram as the "main topic"
        main_idx = bigram_counts.argmax()
        main_topic = bigram_features[main_idx]
    else:
        main_topic = keywords[0] if keywords else ''

    return keywords, main_topic

def generate_summary(chat_data: Dict[str, List[str]]) -> str:
    """
    Generate a human-readable summary of the chat log.
    """
    total, user_messages, ai_messages = message_statistics(chat_data)
    # Combine both User and AI messages for keyword/topic extraction
    combined = chat_data.get('User', []) + chat_data.get('AI', [])
    keywords, topic = extract_keywords_and_topic(combined, top_k=5)

    # Format the summary exactly as requested
    summary_lines = [
        "Summary:",
        f"- The conversation had {total} exchanges.",
        f"- The user asked mainly about {topic}.",
        f"- Most common keywords: {', '.join(keywords)}."
    ]
    return '\n'.join(summary_lines)

def summarize_all_chats(folder_path: str) -> str:
    """
    Process and summarize all chat log files (.txt) in a folder.
    """
    if not os.path.isdir(folder_path):
        return f"Error: '{folder_path}' is not a valid directory."

    # List and sort only .txt files
    files = sorted(f for f in os.listdir(folder_path) if f.lower().endswith('.txt'))
    if not files:
        return f"No .txt chat log files found in '{folder_path}'."

    output = []
    for filename in files:
        path = os.path.join(folder_path, filename)
        data = parse_chat_log(path)
        summary = generate_summary(data)
        output.append(f"\nSummary for '{filename}':\n{summary}")

    return '\n'.join(output)

if __name__ == '__main__':
    # folder path -  chat logs folder
    CHAT_LOGS_FOLDER = "./chat_logs"
    # Ensure the folder exists
    if not os.path.exists(CHAT_LOGS_FOLDER):
        print(f"Error: The folder '{CHAT_LOGS_FOLDER}' does not exist.")
    else:
        print(f"Processing chat logs in folder: {CHAT_LOGS_FOLDER}")
        print(summarize_all_chats(CHAT_LOGS_FOLDER))
