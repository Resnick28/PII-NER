"""
generate_data.py

A production-grade utility for generating synthetic PII (Personally Identifiable Information)
data for NER model training. Specifically designed to mimic noisy Speech-to-Text (STT) 
transcripts with Indian context (Hinglish, specific entity formats).

Usage:
    python generate_data.py --train_count 1000 --dev_count 200 --output_dir data/
"""

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Callable, Any, Optional

try:
    from faker import Faker
except ImportError:
    print("Critical Error: 'faker' library is missing. Install it via 'pip install faker'.")
    sys.exit(1)

# --- Configuration & Constants ---

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Constants for Entity Types
class EntityType:
    PERSON = "PERSON_NAME"
    CITY = "CITY"
    LOCATION = "LOCATION"
    PHONE = "PHONE"
    EMAIL = "EMAIL"
    DATE = "DATE"
    CREDIT_CARD = "CREDIT_CARD"

# Mapping digits to spoken words for "Noisy STT" simulation
DIGIT_MAP = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
}

@dataclass
class EntityTemplate:
    """Represents a template for generating a specific entity type."""
    label: str
    is_pii: bool
    generator: Callable[[], str]
    patterns: List[str]

# --- Helper Functions ---

def to_spoken_digits(text_num: str) -> str:
    """Converts '123' -> 'one two three' to mimic STT transcription."""
    return " ".join([DIGIT_MAP.get(c, c) for c in text_num if c.isdigit()])

def to_spaced_digits(text_num: str) -> str:
    """Converts '9876543210' -> '98765 43210' for realistic mobile number spacing."""
    s = str(text_num).replace("-", "").replace(" ", "")
    if len(s) == 10:
        return f"{s[:5]} {s[5:]}"
    return " ".join(s)

def noisify_email(email: str) -> str:
    """
    Applies STT noise to emails.
    Example: 'john@gmail.com' -> 'john at gmail dot com'
    """
    email_lower = email.lower()
    # 10% chance to spell out the domain (e.g., "g m a i l")
    if random.random() < 0.1:
        try:
            user_part, domain_part = email_lower.split('@')
            spaced_domain = " ".join(list(domain_part))
            email_lower = f"{user_part}@{spaced_domain}"
        except ValueError:
            pass # Handle edge cases where split might fail
    
    return email_lower.replace("@", " at ").replace(".", " dot ")

# --- Main Generator Class ---

class SyntheticDataGenerator:
    """
    Engine for generating synthetic PII datasets.
    Uses 'en_IN' locale to match the assignment's specific demographic requirements.
    """
    
    def __init__(self, seed: int = 42):
        self.fake = Faker('en_IN')
        Faker.seed(seed)
        random.seed(seed)
        self.templates = self._build_templates()
        
        # Negative examples (sentences with NO entities) to improve precision
        self.negative_sentences = [
            "this is regarding order id {0} and i checked it two three times already",
            "tomorrow problem delivery please information payment issue",
            "checking plan status yesterday balance complaint",
            "the connection is very poor can you hear me",
            "complaint order tomorrow ticket feedback please support",
            "i want to cancel my subscription immediately"
        ]

    def _generate_phone(self) -> str:
        """Generates noisy phone numbers (spoken, spaced, or standard)."""
        # Indian mobile format usually starts with 6-9
        ph = f"{random.randint(6,9)}{random.randint(100000000, 999999999)}"
        choice = random.random()
        if choice < 0.3:
            return to_spoken_digits(ph)
        elif choice < 0.7:
            return to_spaced_digits(ph)
        return ph

    def _generate_cc(self) -> str:
        """Generates noisy credit card numbers."""
        cc = self.fake.credit_card_number()
        # 50% chance of spoken digits (high frequency in stress data)
        return to_spoken_digits(cc) if random.random() < 0.5 else cc

    def _generate_date(self) -> str:
        """Generates dates in various spoken/written formats."""
        d = self.fake.future_date(end_date="+3y")
        choice = random.random()
        
        if choice < 0.4: 
            return d.strftime("%d/%m/%Y")
            
        if choice < 0.7: 
            return f"{d.day} {d.strftime('%B %Y')}".lower()
            
        return f"{d.day} of {d.strftime('%B %Y')}".lower()

    def _build_templates(self) -> List[EntityTemplate]:
        """Define the schema and patterns for all entity types."""
        return [
            EntityTemplate(
                label=EntityType.PERSON, is_pii=True, 
                generator=lambda: self.fake.name(),
                patterns=["this is {0}", "my name is {0}", "i am {0}", "haan so my naam is {0}"]
            ),
            EntityTemplate(
                label=EntityType.CITY, is_pii=False,
                generator=lambda: random.choice(["hyderabad", "pune", "mumbai", "bengaluru", "kolkata", "lucknow", "surat", "jaipur"]),
                patterns=["from {0}", "travelling to {0}", "rehte in {0}", "office is in {0}"]
            ),
            EntityTemplate(
                label=EntityType.LOCATION, is_pii=False,
                generator=lambda: random.choice(["koramangala", "hitech city", "mg road", "old airport road", "whitefield", "banjara hills"]),
                patterns=["near {0}", "work in {0}", "office is near {0}"]
            ),
            EntityTemplate(
                label=EntityType.PHONE, is_pii=True, 
                generator=self._generate_phone,
                patterns=["my phone is {0}", "number is {0}", "call me on {0}"]
            ),
            EntityTemplate(
                label=EntityType.EMAIL, is_pii=True, 
                generator=lambda: noisify_email(self.fake.email()),
                patterns=["email is {0}", "send email to {0} please"]
            ),
            EntityTemplate(
                label=EntityType.DATE, is_pii=True, 
                generator=self._generate_date,
                patterns=["meet on {0}", "travelling on {0}", "expires on {0}"]
            ),
            EntityTemplate(
                label=EntityType.CREDIT_CARD, is_pii=True, 
                generator=self._generate_cc,
                patterns=["card number is {0}", "old card number maybe is {0}"]
            )
        ]

    def create_example(self, idx: int) -> Dict[str, Any]:
        """Creates a single training example (dictionary)."""
        
        # 15% chance to generate a Negative Example (No Entities)
        if random.random() < 0.15:
            text = random.choice(self.negative_sentences).format(random.randint(100000, 999999))
            return {"id": f"gen_{idx}", "text": text, "entities": []}

        # Select primary entity
        t1 = random.choice(self.templates)
        val1 = t1.generator()
        
        # robust pattern splitting
        pattern = random.choice(t1.patterns)
        if "{0}" not in pattern:
            # Fallback if pattern is malformed
            pattern = "value is {0}" 
            
        pre_text = pattern.split("{0}")[0]
        full_text = f"{pre_text}{val1}"
        
        entities = [{
            "start": len(pre_text),
            "end": len(pre_text) + len(val1),
            "label": t1.label
        }]

        # 30% chance to append a second entity (Complex sentence)
        if random.random() < 0.3:
            t2 = random.choice(self.templates)
            val2 = t2.generator()
            connector = random.choice([" and ", " ", ". ", " also "])
            
            pre_t2 = random.choice(t2.patterns).split("{0}")[0]
            start_offset = len(full_text) + len(connector) + len(pre_t2)
            
            full_text += f"{connector}{pre_t2}{val2}"
            entities.append({
                "start": start_offset,
                "end": start_offset + len(val2),
                "label": t2.label
            })

        return {
            "id": f"gen_{idx}",
            "text": full_text.lower(), # STT standard: lowercase
            "entities": entities
        }

    def generate_file(self, output_path: Path, count: int):
        """Generates a dataset file with `count` examples."""
        logger.info(f"Generating {count} examples into {output_path}...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i in range(count):
                example = self.create_example(i)
                f.write(json.dumps(example) + "\n")
                
        logger.info(f"Successfully created {output_path}")

# --- CLI Entry Point ---

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic PII training data.")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save output files")
    parser.add_argument("--train_count", type=int, default=800, help="Number of training examples")
    parser.add_argument("--dev_count", type=int, default=200, help="Number of dev examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generator = SyntheticDataGenerator(seed=args.seed)
    
    # Generate Files
    generator.generate_file(output_dir / "train_generated.jsonl", args.train_count)
    generator.generate_file(output_dir / "dev_generated.jsonl", args.dev_count)
    
    print("\n" + "="*50)
    print("Data Generation Complete")
    print(f"Output Directory: {output_dir.absolute()}")
    print("Next Step: Run the following commands to merge data:")
    print("cat data/train.jsonl data/train_generated.jsonl > data/final_train.jsonl")
    print("cat data/dev.jsonl data/dev_generated.jsonl > data/final_dev.jsonl")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()