# ══════════════════════════════════════════════════════════
# LOCAL BPE TRAINING SCRIPT
# Run this on MacBook M2 before Kaggle training
# ══════════════════════════════════════════════════════════

import os
import random
import sentencepiece as spm
from pathlib import Path

print("=" * 60)
print("LOCAL BPE TOKENIZER TRAINING")
print("=" * 60)

SEED = 42
random.seed(SEED)

# Dataset path (from your download)
DATASET_PATH = Path("/Users/rishitladdha/.cache/kagglehub/datasets/mathurinache/samanantar/versions/1/final_data/en-gu")

EN_FILE = DATASET_PATH / "train.en"
GU_FILE = DATASET_PATH / "train.gu"

print(f"English file: {EN_FILE}")
print(f"Gujarati file: {GU_FILE}")

# Check files exist
if not EN_FILE.exists() or not GU_FILE.exists():
    print("ERROR: Files not found!")
    exit(1)

print("\n" + "=" * 60)
print("LOADING DATA")
print("=" * 60)

print("Loading sentences...")
with open(EN_FILE, 'r', encoding='utf-8') as f:
    english_sentences = [line.strip() for line in f.readlines()]
with open(GU_FILE, 'r', encoding='utf-8') as f:
    gujarati_sentences = [line.strip() for line in f.readlines()]

print(f"Total sentences loaded: {len(english_sentences)}")

# Filter by length and take 300k samples
MAX_CHAR_LENGTH = 200
TOTAL_SAMPLES = 300000

valid_pairs = []
for en, gu in zip(english_sentences, gujarati_sentences):
    if 0 < len(en) < MAX_CHAR_LENGTH and 0 < len(gu) < MAX_CHAR_LENGTH:
        valid_pairs.append((en, gu))
    if len(valid_pairs) >= TOTAL_SAMPLES:
        break

print(f"Valid sentence pairs: {len(valid_pairs)}")

# Shuffle
random.shuffle(valid_pairs)

# Split into train/val (95%/5%)
VAL_SIZE = 15000
train_pairs = valid_pairs[:-VAL_SIZE]
val_pairs = valid_pairs[-VAL_SIZE:]

train_en = [p[0] for p in train_pairs]
train_gu = [p[1] for p in train_pairs]
val_en = [p[0] for p in val_pairs]
val_gu = [p[1] for p in val_pairs]

print(f"Training samples: {len(train_en)}")
print(f"Validation samples: {len(val_en)}")

# Save validation data for Kaggle
print("\nSaving validation data...")
with open("val_en.txt", "w", encoding="utf-8") as f:
    for line in val_en:
        f.write(line + "\n")
with open("val_gu.txt", "w", encoding="utf-8") as f:
    for line in val_gu:
        f.write(line + "\n")
print("✓ Saved val_en.txt and val_gu.txt")

# ══════════════════════════════════════════════════════════
# TRAIN BPE TOKENIZERS
# ══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("TRAINING BPE TOKENIZERS")
print("=" * 60)

VOCAB_SIZE = 16000
BPE_TRAIN_SAMPLES = 100000

print(f"\nPreparing tokenizer training data ({BPE_TRAIN_SAMPLES} samples)...")

with open("bpe_train_en.txt", "w", encoding="utf-8") as f:
    for line in train_en[:BPE_TRAIN_SAMPLES]:
        f.write(line + "\n")

with open("bpe_train_gu.txt", "w", encoding="utf-8") as f:
    for line in train_gu[:BPE_TRAIN_SAMPLES]:
        f.write(line + "\n")

print("✓ Training data files created")

# Train English BPE
print("\nTraining English BPE tokenizer...")
spm.SentencePieceTrainer.Train(
    input="bpe_train_en.txt",
    model_prefix="en_bpe",
    vocab_size=VOCAB_SIZE,
    character_coverage=1.0,
    model_type="bpe",
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    num_threads=8
)
print("✓ English BPE training complete!")

# Train Gujarati BPE
print("\nTraining Gujarati BPE tokenizer...")
spm.SentencePieceTrainer.Train(
    input="bpe_train_gu.txt",
    model_prefix="gu_bpe",
    vocab_size=VOCAB_SIZE,
    character_coverage=1.0,
    model_type="bpe",
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    num_threads=8
)
print("✓ Gujarati BPE training complete!")

# Load and test tokenizers
print("\n" + "=" * 60)
print("TESTING TOKENIZERS")
print("=" * 60)

en_sp = spm.SentencePieceProcessor(model_file="en_bpe.model")
gu_sp = spm.SentencePieceProcessor(model_file="gu_bpe.model")

print(f"\nEnglish vocab size: {en_sp.GetPieceSize()}")
print(f"Gujarati vocab size: {gu_sp.GetPieceSize()}")

test_en = "Hello, how are you today?"
test_gu = "તમે આજે કેમ છો?"

print(f"\nTest English: '{test_en}'")
print(f"  Encoded: {en_sp.Encode(test_en)}")
print(f"  Decoded: '{en_sp.Decode(en_sp.Encode(test_en))}'")

print(f"\nTest Gujarati: '{test_gu}'")
print(f"  Encoded: {gu_sp.Encode(test_gu)}")
print(f"  Decoded: '{gu_sp.Decode(gu_sp.Encode(test_gu))}'")

# Clean up temp files
os.remove("bpe_train_en.txt")
os.remove("bpe_train_gu.txt")

# ══════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("LOCAL TRAINING COMPLETE!")
print("=" * 60)
print("\nFiles created in ~/Desktop/translation_project/:")
print("  1. en_bpe.model")
print("  2. en_bpe.vocab")
print("  3. gu_bpe.model")
print("  4. gu_bpe.vocab")
print("  5. val_en.txt")
print("  6. val_gu.txt")
print("\nNext steps:")
print("  1. Go to kaggle.com/datasets → New Dataset")
print("  2. Upload all 6 files above")
print("  3. Name it 'en-gu-bpe-tokenizers'")
print("  4. Use the Kaggle notebook code")
