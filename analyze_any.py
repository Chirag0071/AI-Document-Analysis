"""
analyze_any.py — Analyze ANY document from your PC without changing any code.

Usage:
    python analyze_any.py                    → shows menu to pick file
    python analyze_any.py myfile.pdf         → analyze specific file
    python analyze_any.py invoice.pdf pdf    → specify type manually

Supports: PDF, DOCX, JPG, JPEG, PNG, BMP, TIFF, WEBP
"""

import base64
import json
import sys
import os
import urllib.request
import urllib.error

API_URL = "http://127.0.0.1:8000/api/document-analyze"
API_KEY = "sk_track2_987654321"

# File extension to fileType mapping
EXT_MAP = {
    ".pdf":  "pdf",
    ".docx": "docx",
    ".doc":  "docx",
    ".jpg":  "image",
    ".jpeg": "image",
    ".png":  "image",
    ".bmp":  "image",
    ".tiff": "image",
    ".tif":  "image",
    ".webp": "image",
}

GREEN  = "\033[92m"
RED    = "\033[91m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def detect_type(file_path: str) -> str:
    """Auto-detect fileType from extension."""
    ext = os.path.splitext(file_path)[1].lower()
    ft = EXT_MAP.get(ext)
    if not ft:
        print(f"{RED}Unsupported file type: {ext}{RESET}")
        print(f"Supported: {', '.join(EXT_MAP.keys())}")
        sys.exit(1)
    return ft


def analyze_file(file_path: str, file_type: str = None):
    """Analyze a single file and print results."""

    # Validate file exists
    if not os.path.exists(file_path):
        print(f"{RED}File not found: {file_path}{RESET}")
        sys.exit(1)

    # Auto-detect type if not provided
    if not file_type:
        file_type = detect_type(file_path)

    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{CYAN}{BOLD}Analyzing: {file_name}{RESET}")
    print(f"Type    : {file_type.upper()}")
    print(f"Size    : {file_size / 1024:.1f} KB")
    print(f"{'='*60}{RESET}")

    # Read and encode file
    print("Reading file...")
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    b64 = base64.b64encode(file_bytes).decode("utf-8")
    print(f"Sending to API... (this may take 5-30 seconds)")

    # Send to API
    payload = {
        "fileName": file_name,
        "fileType": file_type,
        "fileBase64": b64,
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            API_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "x-api-key": API_KEY,
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read().decode())

    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"{RED}HTTP {e.code} Error: {body}{RESET}")
        return
    except urllib.error.URLError:
        print(f"{RED}Cannot connect to server!{RESET}")
        print("Make sure server is running: uvicorn main:app --reload")
        return
    except Exception as e:
        print(f"{RED}Error: {e}{RESET}")
        return

    # Print results
    print(f"\n{GREEN}{BOLD}✓ Analysis Complete!{RESET}")
    print(f"Processing time: {result.get('processing_time_seconds', '?')}s\n")

    print(f"{BOLD}📋 DOCUMENT TYPE:{RESET}")
    print(f"   {result.get('document_type', 'Unknown')}")

    print(f"\n{BOLD}📝 SUMMARY:{RESET}")
    print(f"   {result.get('summary', 'No summary available')}")

    print(f"\n{BOLD}😊 SENTIMENT:{RESET}")
    label = result.get('sentiment', 'Unknown')
    scores = result.get('sentiment_scores', {})
    color = GREEN if label == "Positive" else RED if label == "Negative" else YELLOW
    print(f"   {color}{BOLD}{label}{RESET}", end="")
    if scores:
        print(f"  (Pos:{scores.get('Positive',0):.0%}  Neu:{scores.get('Neutral',0):.0%}  Neg:{scores.get('Negative',0):.0%})")
    else:
        print()

    print(f"\n{BOLD}🏷️  ENTITIES:{RESET}")
    ents = result.get("entities", {})
    entity_labels = {
        "names":         "👤 Names",
        "organizations": "🏢 Organizations",
        "dates":         "📅 Dates",
        "locations":     "📍 Locations",
        "amounts":       "💰 Amounts",
        "percentages":   "📊 Percentages",
        "emails":        "📧 Emails",
        "phones":        "📞 Phones",
        "urls":          "🔗 URLs",
    }
    found_any = False
    for key, label in entity_labels.items():
        values = ents.get(key, [])
        if values:
            print(f"   {label}: {', '.join(str(v) for v in values)}")
            found_any = True
    if not found_any:
        print("   No entities detected")

    kp = result.get("key_phrases", [])
    if kp:
        print(f"\n{BOLD}🔑 KEY PHRASES:{RESET}")
        print(f"   {', '.join(kp[:8])}")

    stats = result.get("document_stats", {})
    if stats:
        readability = stats.get("readability", {})
        print(f"\n{BOLD}📊 DOCUMENT STATS:{RESET}")
        print(f"   Words        : {stats.get('word_count', 0)}")
        print(f"   Sentences    : {stats.get('sentence_count', 0)}")
        print(f"   Reading time : {stats.get('reading_time_minutes', 0)} min")
        print(f"   Language     : {stats.get('language', 'en').upper()}")
        print(f"   Readability  : {readability.get('interpretation', '?')} "
              f"(Flesch: {readability.get('flesch_reading_ease', 0)})")

    print(f"\n{BOLD}{'='*60}{RESET}")

    # Ask to save JSON
    save = input(f"\nSave full JSON result to file? (y/n): ").strip().lower()
    if save == "y":
        out_file = f"{os.path.splitext(file_name)[0]}_result.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"{GREEN}Saved to: {out_file}{RESET}")


def show_menu():
    """Interactive menu to pick a file from current directory."""
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{CYAN}{BOLD}  AI Document Analyzer — Interactive Mode{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    print(f"Supported formats: PDF, DOCX, JPG, PNG, BMP, TIFF, WEBP\n")

    # List supported files in current directory
    files = [
        f for f in os.listdir(".")
        if os.path.splitext(f)[1].lower() in EXT_MAP
        and os.path.isfile(f)
    ]

    if files:
        print(f"{BOLD}Files found in current folder:{RESET}")
        for i, f in enumerate(files, 1):
            size = os.path.getsize(f) / 1024
            ext = os.path.splitext(f)[1].lower()
            ftype = EXT_MAP.get(ext, "?")
            print(f"  [{i}] {f}  ({ftype.upper()}, {size:.1f} KB)")
        print(f"  [0] Enter a custom file path")
        print()

        choice = input("Enter number (or 0 for custom path): ").strip()
        if choice == "0":
            file_path = input("Enter full file path: ").strip().strip('"')
        elif choice.isdigit() and 1 <= int(choice) <= len(files):
            file_path = files[int(choice) - 1]
        else:
            print(f"{RED}Invalid choice{RESET}")
            return
    else:
        print("No supported files found in current folder.")
        file_path = input("Enter full file path: ").strip().strip('"')

    analyze_file(file_path)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments — show interactive menu
        show_menu()
    elif len(sys.argv) == 2:
        # One argument — file path
        analyze_file(sys.argv[1])
    elif len(sys.argv) == 3:
        # Two arguments — file path + type
        analyze_file(sys.argv[1], sys.argv[2])
    else:
        print("Usage:")
        print("  python analyze_any.py                  → interactive menu")
        print("  python analyze_any.py myfile.pdf       → analyze file")
        print("  python analyze_any.py myfile.png image → specify type")