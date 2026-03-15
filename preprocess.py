import os

def load_full_and_summary(full_dir, summary_dir):
    full_texts = []
    summaries = []

    full_files = sorted(os.listdir(full_dir))
    summary_files = sorted(os.listdir(summary_dir))

    print("DEBUG: Full files count:", len(full_files))
    print("DEBUG: Summary files count:", len(summary_files))

    for fname in full_files:
        full_path = os.path.join(full_dir, fname)
        summary_path = os.path.join(summary_dir, fname)

        if not os.path.exists(summary_path):
            continue

        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            full_text = f.read().strip()

        with open(summary_path, "r", encoding="utf-8", errors="ignore") as f:
            summary_text = f.read().strip()

        if full_text and summary_text:
            full_texts.append(full_text)
            summaries.append(summary_text)

    print("DEBUG: Loaded full_texts:", len(full_texts))
    print("DEBUG: Loaded summaries:", len(summaries))

    return full_texts, summaries
