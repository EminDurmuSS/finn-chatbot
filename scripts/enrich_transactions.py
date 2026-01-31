#!/usr/bin/env python
"""Data enrichment for bank transactions using OpenRouter LLM."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

logger = logging.getLogger("enrich_transactions")

HEADER_MARKERS = ["Date/Time", "Transaction Type", "Description", "Transaction Amount*"]
COLUMN_MAP = {
    "Date/Time": "date_time",
    "Value Date": "value_date",
    "Channel/Branch": "channel",
    "Transaction Amount*": "amount",
    "Balance": "balance",
    "Overdraft Balance": "overdraft_balance",
    "Transaction Code": "transaction_code",
    "Transaction Type": "transaction_type",
    "Description": "description",
    "Reference": "reference",
}

PERSON_INDICATORS = ["FROM", "SENT BY", "TRANSFERRED", "PAYEE", "P2P"]
TRANSFER_TYPES = {"TRANSFER", "WIRE", "ACH", "SEPA"}
ATM_TYPES = {"ATM", "CASH WITHDRAWAL"}
INCOME_TYPES = {"DEPOSIT", "CASH DEPOSIT", "SALARY"}
FEE_TYPES = {"FEE", "CHARGE"}
FX_TYPES = {"CURRENCY", "FX", "EXCHANGE"}
INVESTMENT_TYPES = {"INVESTMENT", "BROKERAGE"}
BILL_TYPES = {"BILL", "UTILITY", "INVOICE"}

COUNTRY_RULES = [
    {
        "code": "NL",
        "name": "Netherlands",
        "strong_keywords": [
            "NETHERLANDS",
            "NEDERLAND",
            "HOLLAND",
            "HOLLANDA",
            "AMSTERDAM",
            "ROTTERDAM",
            "THE HAGUE",
            "DEN HAAG",
            "UTRECHT",
            "EINDHOVEN",
            "GRONINGEN",
            "MAASTRICHT",
            "HAARLEM",
            "TILBURG",
            "ENSCHEDE",
            "NIJMEGEN",
            "ZWOLLE",
            "LEEUWARDEN",
            "ALMERE",
            "SCHIPHOL",
        ],
        "weak_keywords": ["NL", "NLD"],
    },
    {
        "code": "TR",
        "name": "Turkey",
        "strong_keywords": [
            "TURKEY",
            "TURKIYE",
            "ADANA",
            "ADIYAMAN",
            "AFYONKARAHISAR",
            "AGRI",
            "AMASYA",
            "ANKARA",
            "ANTALYA",
            "ARTVIN",
            "AYDIN",
            "BALIKESIR",
            "BILECIK",
            "BINGOL",
            "BITLIS",
            "BOLU",
            "BURDUR",
            "BURSA",
            "CANAKKALE",
            "CANKIRI",
            "CORUM",
            "DENIZLI",
            "DIYARBAKIR",
            "EDIRNE",
            "ELAZIG",
            "ERZINCAN",
            "ERZURUM",
            "ESKISEHIR",
            "GAZIANTEP",
            "GIRESUN",
            "GUMUSHANE",
            "HAKKARI",
            "HATAY",
            "ISPARTA",
            "MERSIN",
            "ISTANBUL",
            "IZMIR",
            "KARS",
            "KASTAMONU",
            "KAYSERI",
            "KIRKLARELI",
            "KIRSEHIR",
            "KOCAELI",
            "KONYA",
            "KUTAHYA",
            "MALATYA",
            "MANISA",
            "KAHRAMANMARAS",
            "MARDIN",
            "MUGLA",
            "MUS",
            "NEVSEHIR",
            "NIGDE",
            "ORDU",
            "RIZE",
            "SAKARYA",
            "SAMSUN",
            "SIIRT",
            "SINOP",
            "SIVAS",
            "TEKIRDAG",
            "TOKAT",
            "TRABZON",
            "TUNCELI",
            "SANLIURFA",
            "USAK",
            "VAN",
            "YOZGAT",
            "ZONGULDAK",
            "AKSARAY",
            "BAYBURT",
            "KARAMAN",
            "KIRIKKALE",
            "BATMAN",
            "SIRNAK",
            "BARTIN",
            "ARDAHAN",
            "IGDIR",
            "YALOVA",
            "KARABUK",
            "KILIS",
            "OSMANIYE",
            "DUZCE",
        ],
        "weak_keywords": ["TR"],
    },
]

def setup_logging(level: str, log_file: Optional[Path] = None) -> None:
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.insert(0, logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )

def load_dotenv(dotenv_path: Optional[Path] = None, override: bool = False) -> Optional[Path]:
    candidates: List[Path] = []
    if dotenv_path:
        candidates.append(dotenv_path)
    else:
        candidates.append(Path.cwd())
        candidates.append(Path(__file__).resolve().parent)

    seen: set[Path] = set()
    resolved: List[Path] = []
    for start in candidates:
        for parent in [start, *start.parents]:
            if parent in seen:
                continue
            seen.add(parent)
            resolved.append(parent)

    env_file: Optional[Path] = None
    for parent in resolved:
        candidate = parent / ".env"
        if candidate.is_file():
            env_file = candidate
            break

    if not env_file:
        return None

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        if not override and key in os.environ:
            continue
        os.environ[key] = value

    return env_file



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich transactions with LLM categories.")
    parser.add_argument("--data-dir", default="data", help="Folder with .xls/.xlsx files")
    parser.add_argument("--output", default="data/enriched_transactions.jsonl", help="Output path")
    parser.add_argument("--taxonomy", default="data/category_taxonomy_v1.json", help="Taxonomy JSON path")
    parser.add_argument("--batch-size", type=int, default=50, help="LLM batch size")
    parser.add_argument("--max-rows", type=int, default=0, help="Max rows to process (0 = all)")
    parser.add_argument("--format", choices=["jsonl", "csv"], default="jsonl", help="Output format")
    parser.add_argument("--dry-run", action="store_true", help="Parse only, no LLM calls")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--log-file", default="", help="Optional log file path")
    return parser.parse_args()


def find_header_row(raw_df: pd.DataFrame) -> Optional[int]:
    for idx, row in raw_df.iterrows():
        row_text = " | ".join([str(x) for x in row.tolist()])
        if all(marker in row_text for marker in HEADER_MARKERS[:2]):
            return idx
    return None


def normalize_reference(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def parse_amount(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    text = text.replace(" ", "")
    if "," in text and "." in text:
        text = text.replace(".", "").replace(",", ".")
    elif "," in text:
        text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def parse_datetime(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    try:
        dt = pd.to_datetime(text, dayfirst=True, errors="coerce")
        if pd.isna(dt):
            return text
        return dt.isoformat()
    except Exception:
        return text


def normalize_location_text(text: str) -> str:
    if not text:
        return ""
    normalized = text.upper().translate(TURKISH_CHAR_MAP)
    normalized = re.sub(r"[^A-Z0-9 ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def keyword_in_text(normalized_text: str, tokens: set[str], keyword: str) -> bool:
    if not keyword:
        return False
    if " " in keyword:
        return f" {keyword} " in f" {normalized_text} "
    return keyword in tokens


def detect_country(description: str) -> Tuple[str, str, float]:
    normalized = normalize_location_text(description)
    if not normalized:
        return "", "", 0.0

    tokens = set(normalized.split())

    for rule in COUNTRY_RULES:
        for keyword in rule.get("strong_keywords", []):
            if keyword_in_text(normalized, tokens, keyword):
                return rule["code"], rule["name"], 0.9

    for rule in COUNTRY_RULES:
        for keyword in rule.get("weak_keywords", []):
            if keyword_in_text(normalized, tokens, keyword):
                return rule["code"], rule["name"], 0.6

    return "", "", 0.0


def infer_direction(amount: Optional[float], tx_type: str) -> str:
    if amount is None:
        return "neutral"
    direction = "neutral"
    if amount < 0:
        direction = "expense"
    elif amount > 0:
        direction = "income"
    if (tx_type or "").upper() in TRANSFER_TYPES:
        return "transfer"
    return direction


def detect_special_cases(description: str, tx_type: str, amount: float) -> Optional[Dict[str, Any]]:
    desc_upper = description.upper()
    tx_type_upper = (tx_type or "").upper()

    if tx_type_upper == "BLOKAJ" or "BLK KAYD" in desc_upper or "BLK İPTL" in desc_upper:
        return {
            "merchant_norm": "AUTHORIZATION_HOLD",
            "category": "internal_banking",
            "subcategory": "authorization_hold",
            "confidence": 1.0,
            "tags": ["internal"],
            "reasoning": "Authorization hold / blokaj"
        }

    if tx_type_upper in FEE_TYPES:
        return {
            "merchant_norm": "BANK_FEE",
            "category": "financial_services",
            "subcategory": "bank_fees",
            "confidence": 1.0,
            "tags": ["bank"],
            "reasoning": "Bank fee"
        }

    if tx_type_upper in FX_TYPES:
        return {
            "merchant_norm": "CURRENCY_EXCHANGE_INTERNAL",
            "category": "internal_banking",
            "subcategory": "currency_exchange_internal",
            "confidence": 1.0,
            "tags": ["internal"],
            "reasoning": "Internal currency exchange"
        }

    if tx_type_upper in INVESTMENT_TYPES:
        return {
            "merchant_norm": "INVESTMENT",
            "category": "financial_services",
            "subcategory": "investment",
            "confidence": 0.95,
            "tags": ["investment"],
            "reasoning": "Investment transaction"
        }

    if tx_type_upper in ATM_TYPES:
        return {
            "merchant_norm": "ATM_WITHDRAWAL",
            "category": "financial_services",
            "subcategory": "atm_withdrawal",
            "confidence": 0.98,
            "tags": ["atm"],
            "reasoning": "ATM withdrawal"
        }

    if tx_type_upper in INCOME_TYPES:
        return {
            "merchant_norm": "CASH_DEPOSIT",
            "category": "income",
            "subcategory": "other_income",
            "confidence": 0.9,
            "tags": ["cash_deposit"],
            "reasoning": "Cash deposit"
        }

    if tx_type_upper in BILL_TYPES:
        return {
            "merchant_norm": "UTILITY_BILL",
            "category": "utilities",
            "subcategory": "other_utilities",
            "confidence": 0.9,
            "tags": ["bill"],
            "reasoning": "Utility bill"
        }

    if "DÖVİZ" in desc_upper or "CURRENCY" in desc_upper or "MAXİ İLE DÖVİZ" in desc_upper:
        return {
            "merchant_norm": "CURRENCY_EXCHANGE_INTERNAL",
            "category": "internal_banking",
            "subcategory": "currency_exchange_internal",
            "confidence": 1.0,
            "tags": ["internal"],
            "reasoning": "Internal currency exchange"
        }

    if "KAMBIYO" in desc_upper or "MUAM" in desc_upper or "VERGİSİ" in desc_upper:
        return {
            "merchant_norm": "BANK_FEE_CURRENCY_TAX",
            "category": "financial_services",
            "subcategory": "bank_fees",
            "confidence": 1.0,
            "tags": ["bank"],
            "reasoning": "Bank fee / tax"
        }

    if re.search(r"ÜCRET\s+H\d+", desc_upper):
        return {
            "merchant_norm": "BANK_FEE_TRANSFER",
            "category": "financial_services",
            "subcategory": "bank_fees",
            "confidence": 1.0,
            "tags": ["bank"],
            "reasoning": "Transfer fee"
        }

    if "KAREKOD" in desc_upper and "PARA ÇEKME" in desc_upper:
        return {
            "merchant_norm": "ATM_QR_WITHDRAWAL",
            "category": "financial_services",
            "subcategory": "atm_withdrawal",
            "confidence": 1.0,
            "tags": ["atm"],
            "reasoning": "ATM withdrawal"
        }

    if "ORTAK ATM" in desc_upper or "BANKAMATIK" in desc_upper:
        return {
            "merchant_norm": "ATM_WITHDRAWAL",
            "category": "financial_services",
            "subcategory": "atm_withdrawal",
            "confidence": 0.98,
            "tags": ["atm"],
            "reasoning": "ATM withdrawal"
        }

    is_p2p = tx_type_upper in TRANSFER_TYPES and any(ind in desc_upper for ind in PERSON_INDICATORS)
    if is_p2p:
        direction = "p2p_sent" if amount < 0 else "p2p_received"
        return {
            "merchant_norm": "P2P_TRANSFER",
            "category": "transfers",
            "subcategory": direction,
            "confidence": 0.95,
            "tags": ["p2p"],
            "reasoning": "P2P transfer"
        }

    if tx_type_upper in TRANSFER_TYPES:
        return {
            "merchant_norm": "INTERNAL_TRANSFER",
            "category": "transfers",
            "subcategory": "internal_transfer",
            "confidence": 0.85,
            "tags": ["transfer"],
            "reasoning": "Transfer transaction"
        }

    if "/FATURA" in desc_upper or "FATURA NO:" in desc_upper:
        return {
            "merchant_norm": "UTILITY_BILL",
            "category": "utilities",
            "subcategory": "other_utilities",
            "confidence": 0.9,
            "tags": ["bill"],
            "reasoning": "Utility bill"
        }

    return None


def log_tx_classification(
    tx: Dict[str, Any],
    source: str,
    category: str,
    subcategory: str,
    merchant_norm: str,
    confidence: float,
    tags: List[str],
    reasoning: str,
) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    logger.debug(
        "tx_id=%s source=%s category=%s/%s merchant=%s confidence=%.2f amount=%s type=%s tags=%s reason=%s desc=%s",
        tx.get("tx_id"),
        source,
        category,
        subcategory,
        merchant_norm,
        confidence,
        tx.get("amount"),
        tx.get("transaction_type"),
        ",".join(tags) if isinstance(tags, list) else tags,
        reasoning,
        tx.get("description"),
    )

def load_taxonomy(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compact_taxonomy(taxonomy: Dict[str, Any]) -> Dict[str, Any]:
    if not taxonomy:
        return {}
    compact = {"categories": {}}
    special_section = taxonomy.get("special_categories", {}) or {}
    special_ids = set(special_section.get("category_ids") or [])
    legacy_special = special_section.get("categories") or {}

    def add_category(cat_id: str, cat: Dict[str, Any], special: bool = False) -> None:
        subcats = {}
        for sub_id, sub in (cat.get("subcategories") or {}).items():
            subcats[sub_id] = {
                "display_name": sub.get("display_name"),
                "keywords": sub.get("keywords", []),
                "merchants_examples": sub.get("merchants_examples", []),
            }
        compact["categories"][cat_id] = {
            "display_name": cat.get("display_name"),
            "keywords": cat.get("keywords", []),
            "subcategories": subcats,
            "special": special,
        }

    for cat_id, cat in (taxonomy.get("categories") or {}).items():
        add_category(cat_id, cat, special=(cat_id in special_ids))

    for cat_id, cat in legacy_special.items():
        if cat_id in compact["categories"]:
            compact["categories"][cat_id]["special"] = True
            continue
        add_category(cat_id, cat, special=True)

    return compact


def build_taxonomy_index(taxonomy: Dict[str, Any]) -> Tuple[set, Dict[str, set]]:
    categories = taxonomy.get("categories", {})
    cat_set = set(categories.keys())
    sub_map = {
        cat_id: set((cat.get("subcategories") or {}).keys())
        for cat_id, cat in categories.items()
    }
    return cat_set, sub_map


def normalize_tags(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(tag).strip() for tag in value if str(tag).strip()]
    if isinstance(value, str):
        return [tag.strip() for tag in value.split(",") if tag.strip()]
    return []


def validate_taxonomy_item(category: str, subcategory: str, taxonomy_index: Optional[Tuple[set, Dict[str, set]]]) -> bool:
    if not taxonomy_index:
        return True
    categories, sub_map = taxonomy_index
    if not category or category not in categories:
        return False
    if subcategory and subcategory not in sub_map.get(category, set()):
        return False
    return True


def read_transactions_from_file(path: Path) -> List[Dict[str, Any]]:
    logger.info("Reading transactions from %s", path.name)
    raw_df = pd.read_excel(path, header=None)
    header_row = find_header_row(raw_df)
    if header_row is None:
        raise ValueError(f"Header row not found in {path}")

    df = pd.read_excel(path, header=header_row, dtype=str)
    df = df.loc[:, [col for col in df.columns if not str(col).startswith("Unnamed")]]

    records: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        date_time = parse_datetime(row.get("Tarih/Saat"))
        if not date_time:
            continue
        amount = parse_amount(row.get("İşlem Tutarı*"))
        description = str(row.get("Açıklama") or "").strip()
        tx_type = str(row.get("İşlem Tipi") or "").strip()
        reference = normalize_reference(row.get("Referans"))
        country_code, country_name, country_confidence = detect_country(description)

        record = {
            "tx_id": reference or f"{path.stem}-{idx}",
            "date_time": date_time,
            "value_date": parse_datetime(row.get("Valör")),
            "channel": str(row.get("Kanal/Şube") or "").strip(),
            "amount": amount,
            "balance": parse_amount(row.get("Bakiye")),
            "overdraft_balance": parse_amount(row.get("Ek Hesap\nBakiyesi")),
            "transaction_code": str(row.get("İşlem") or "").strip(),
            "transaction_type": tx_type,
            "description": description,
            "reference": reference,
            "source_file": path.name,
            "country_code": country_code,
            "country_name": country_name,
            "country_confidence": country_confidence,
        }
        record["direction"] = infer_direction(amount, tx_type)
        records.append(record)
    return records


def load_transactions(data_dir: Path) -> List[Dict[str, Any]]:
    files = sorted([p for p in data_dir.glob("*.xls*") if not p.name.startswith("~$")])
    if not files:
        raise FileNotFoundError(f"No .xls/.xlsx files found in {data_dir}")

    logger.info("Found %d input file(s) in %s", len(files), data_dir)
    logger.info("Input files: %s", ", ".join([p.name for p in files]))

    all_records: List[Dict[str, Any]] = []
    seen_keys = set()
    duplicate_count = 0

    for path in files:
        records = read_transactions_from_file(path)
        logger.info("Parsed %d row(s) from %s", len(records), path.name)
        for record in records:
            key = record.get("reference") or (record.get("date_time"), record.get("amount"), record.get("description"))
            if key in seen_keys:
                duplicate_count += 1
                continue
            seen_keys.add(key)
            all_records.append(record)
    logger.info(
        "After deduplication: %d transaction(s), %d duplicate(s) skipped",
        len(all_records),
        duplicate_count,
    )
    return all_records


def call_openrouter(messages: List[Dict[str, str]], model: str, api_key: str) -> Dict[str, Any]:
    url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    http_referer = os.getenv("OPENROUTER_HTTP_REFERER")
    if http_referer:
        headers["HTTP-Referer"] = http_referer
    app_title = os.getenv("OPENROUTER_APP_TITLE")
    if app_title:
        headers["X-Title"] = app_title

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
    }

    for attempt in range(5):
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in {429, 500, 502, 503}:
            logger.warning(
                "OpenRouter error %s on attempt %d/5; retrying in %s sec",
                resp.status_code,
                attempt + 1,
                2 ** attempt,
            )
            time.sleep(2 ** attempt)
            continue
        logger.error("OpenRouter error %s: %s", resp.status_code, resp.text)
        raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text}")

    raise RuntimeError("OpenRouter request failed after retries")


def extract_json(content: str) -> Any:
    for candidate in iter_json_candidates(content):
        parsed = try_parse_json(candidate)
        if parsed is not None:
            return parsed

    raise ValueError("Could not parse JSON from LLM response")


def iter_json_candidates(content: str) -> Iterable[str]:
    seen: set[str] = set()
    if content:
        cleaned = content.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            yield cleaned

    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)```", content, flags=re.S | re.I)
    for block in fenced_blocks:
        cleaned = block.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            yield cleaned

    xml_blocks = re.findall(r"<json>\s*(.*?)</json>", content, flags=re.S | re.I)
    for block in xml_blocks:
        cleaned = block.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            yield cleaned

    balanced = find_balanced_json(content)
    if balanced:
        cleaned = balanced.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            yield cleaned


def find_balanced_json(text: str) -> Optional[str]:
    if not text:
        return None

    stack: List[str] = []
    start: Optional[int] = None
    in_string = False
    escape = False

    for idx, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "\"":
                in_string = False
            continue

        if ch == "\"":
            in_string = True
            continue

        if ch in "[{":
            if not stack:
                start = idx
            stack.append(ch)
            continue

        if ch in "]}":
            if not stack:
                continue
            last = stack[-1]
            if (ch == "]" and last == "[") or (ch == "}" and last == "{"):
                stack.pop()
                if not stack and start is not None:
                    return text[start : idx + 1]
            else:
                stack.clear()
                start = None

    return None


def try_parse_json(text: str) -> Optional[Any]:
    for candidate in iter_json_variants(text):
        parsed = loads_json(candidate)
        if parsed is not None:
            return parsed
    return None


def iter_json_variants(text: str) -> Iterable[str]:
    cleaned = text.strip()
    if not cleaned:
        return
    if cleaned.endswith(";"):
        yield cleaned[:-1].strip()
    yield cleaned
    repaired = repair_json_candidate(cleaned)
    if repaired != cleaned:
        yield repaired


def loads_json(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return json.loads(text, strict=False)
        except Exception:
            return None
    except Exception:
        return None


def repair_json_candidate(text: str) -> str:
    repaired = text
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = re.sub(r"}\s*{", "},{", repaired)
    repaired = re.sub(r"]\s*{", "],{", repaired)
    repaired = re.sub(r"}\s*\[", "},[", repaired)
    repaired = re.sub(r"]\s*\[", "],[", repaired)
    return repaired


def build_prompt(taxonomy: Dict[str, Any], batch: List[Dict[str, Any]]) -> str:
    taxonomy_text = json.dumps(taxonomy, ensure_ascii=False)
    batch_text = json.dumps(batch, ensure_ascii=False)

    return (
        "You are an expert financial transaction categorization system.\n\n"
        "Use the taxonomy JSON below. Choose the most specific category and subcategory.\nCategory must be a key in taxonomy.categories and subcategory must belong to that category.\n"
        "Return a JSON array of objects with fields: tx_id, merchant_norm, category, subcategory, "
        "confidence (0-1), tags (array), reasoning.\n"
        "Return ONLY valid JSON (no markdown, no code fences, no extra text). "
        "Use double quotes for all strings and do not include trailing commas.\n\n"
        f"Taxonomy JSON:\n{taxonomy_text}\n\n"
        "Transactions:\n"
        f"{batch_text}\n"
    )


def classify_batch(batch: List[Dict[str, Any]], taxonomy: Dict[str, Any], model: str, api_key: str) -> List[Dict[str, Any]]:
    prompt = build_prompt(taxonomy, batch)
    messages = [
        {"role": "system", "content": "You are a financial transaction categorization expert."},
        {"role": "user", "content": prompt},
    ]
    response = call_openrouter(messages, model=model, api_key=api_key)
    content = response["choices"][0]["message"]["content"]
    try:
        data = extract_json(content)
    except ValueError as exc:
        snippet = (content or "").strip().replace("\n", "\\n")
        if len(snippet) > 500:
            snippet = f"{snippet[:500]}...(truncated)"
        logger.warning("LLM response JSON parse failed: %s", exc)
        logger.warning("LLM response snippet: %s", snippet)
        return []

    if isinstance(data, dict):
        if "results" in data:
            data = data["results"]
        elif "transactions" in data:
            data = data["transactions"]
    if not isinstance(data, list):
        logger.warning("LLM response is not a list; falling back. type=%s", type(data).__name__)
        return []
    return data


def enrich_transactions(transactions: List[Dict[str, Any]], taxonomy: Dict[str, Any], batch_size: int, dry_run: bool) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    to_classify: List[Dict[str, Any]] = []
    taxonomy_index = build_taxonomy_index(taxonomy) if taxonomy else None
    rule_count = 0

    for tx in transactions:
        amount = tx.get("amount") or 0.0
        rule_result = detect_special_cases(tx.get("description", ""), tx.get("transaction_type", ""), amount)
        if rule_result:
            rule_count += 1
            record = {**tx, **rule_result, "source": "rule"}
            enriched.append(record)
            log_tx_classification(
                tx,
                source="rule",
                category=record.get("category", ""),
                subcategory=record.get("subcategory", ""),
                merchant_norm=record.get("merchant_norm", ""),
                confidence=float(record.get("confidence", 0.0) or 0.0),
                tags=normalize_tags(record.get("tags")),
                reasoning=record.get("reasoning", ""),
            )
        else:
            to_classify.append(tx)

    logger.info("Rule-based classified: %d; LLM to classify: %d", rule_count, len(to_classify))

    if dry_run:
        logger.info("Dry-run enabled: skipping LLM calls")
        dry_records = []
        for tx in to_classify:
            record = {
                **tx,
                "merchant_norm": "",
                "category": "",
                "subcategory": "",
                "confidence": 0.0,
                "tags": [],
                "reasoning": "",
                "source": "dry_run",
            }
            dry_records.append(record)
            log_tx_classification(
                tx,
                source="dry_run",
                category="",
                subcategory="",
                merchant_norm="",
                confidence=0.0,
                tags=[],
                reasoning="",
            )
        return enriched + dry_records

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required")

    model = os.getenv("OPENROUTER_MODEL", "google/gemini-3-flash-preview")

    total_batches = max(1, math.ceil(len(to_classify) / batch_size)) if to_classify else 0
    fallback_missing = 0
    fallback_invalid = 0

    for batch_index, start in enumerate(range(0, len(to_classify), batch_size), start=1):
        batch = to_classify[start : start + batch_size]
        batch_timer = time.perf_counter()
        logger.info(
            "Classifying batch %d/%d (size=%d)",
            batch_index,
            total_batches,
            len(batch),
        )
        batch_payload = [
            {
                "tx_id": tx["tx_id"],
                "description": tx.get("description", ""),
                "amount": tx.get("amount"),
                "transaction_type": tx.get("transaction_type", ""),
            }
        for tx in batch
        ]
        llm_results = classify_batch(batch_payload, taxonomy, model=model, api_key=api_key)
        result_map = {str(item.get("tx_id")): item for item in llm_results if item}

        for tx in batch:
            item = result_map.get(str(tx["tx_id"]))
            if not item:
                fallback_missing += 1
                record = {
                    **tx,
                    "merchant_norm": "UNKNOWN",
                    "category": "uncategorized",
                    "subcategory": "pending_review",
                    "confidence": 0.0,
                    "tags": [],
                    "reasoning": "Missing LLM result",
                    "source": "fallback",
                }
                enriched.append(record)
                log_tx_classification(
                    tx,
                    source="fallback",
                    category=record.get("category", ""),
                    subcategory=record.get("subcategory", ""),
                    merchant_norm=record.get("merchant_norm", ""),
                    confidence=0.0,
                    tags=[],
                    reasoning=record.get("reasoning", ""),
                )
                continue

            category = str(item.get("category", "")).strip()
            subcategory = str(item.get("subcategory", "")).strip()
            if taxonomy_index and not validate_taxonomy_item(category, subcategory, taxonomy_index):
                fallback_invalid += 1
                record = {
                    **tx,
                    "merchant_norm": "UNKNOWN",
                    "category": "uncategorized",
                    "subcategory": "pending_review",
                    "confidence": 0.0,
                    "tags": [],
                    "reasoning": "Invalid taxonomy category/subcategory",
                    "source": "fallback",
                }
                enriched.append(record)
                log_tx_classification(
                    tx,
                    source="fallback",
                    category=record.get("category", ""),
                    subcategory=record.get("subcategory", ""),
                    merchant_norm=record.get("merchant_norm", ""),
                    confidence=0.0,
                    tags=[],
                    reasoning=record.get("reasoning", ""),
                )
                continue

            try:
                confidence = float(item.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0

            record = {
                **tx,
                "merchant_norm": item.get("merchant_norm", ""),
                "category": category,
                "subcategory": subcategory,
                "confidence": confidence,
                "tags": normalize_tags(item.get("tags")),
                "reasoning": item.get("reasoning", ""),
                "source": "llm",
            }
            enriched.append(record)
            log_tx_classification(
                tx,
                source="llm",
                category=record.get("category", ""),
                subcategory=record.get("subcategory", ""),
                merchant_norm=record.get("merchant_norm", ""),
                confidence=confidence,
                tags=normalize_tags(record.get("tags")),
                reasoning=record.get("reasoning", ""),
            )

        logger.info(
            "Batch %d/%d completed in %.2fs",
            batch_index,
            total_batches,
            time.perf_counter() - batch_timer,
        )

    if fallback_missing or fallback_invalid:
        logger.warning(
            "Fallbacks applied: missing=%d invalid_taxonomy=%d",
            fallback_missing,
            fallback_invalid,
        )

    return enriched

def write_output(records: List[Dict[str, Any]], output_path: Path, fmt: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonl":
        logger.info("Writing JSONL output to %s", output_path)
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return

    if fmt == "csv":
        import csv

        logger.info("Writing CSV output to %s", output_path)
        fieldnames = sorted({key for record in records for key in record.keys()})
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                row = record.copy()
                if isinstance(row.get("tags"), list):
                    row["tags"] = ",".join(row["tags"])
                writer.writerow(row)
        return

    raise ValueError(f"Unknown format: {fmt}")


def main() -> None:
    env_path = load_dotenv()
    args = parse_args()
    log_file = Path(args.log_file) if args.log_file else None
    setup_logging(args.log_level, log_file)
    if env_path:
        logger.info("Loaded environment from %s", env_path)
    start_time = time.perf_counter()
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    taxonomy_path = Path(args.taxonomy)

    logger.info(
        "Starting enrichment: data_dir=%s output=%s taxonomy=%s batch_size=%d max_rows=%d format=%s dry_run=%s",
        data_dir,
        output_path,
        taxonomy_path,
        args.batch_size,
        args.max_rows,
        args.format,
        args.dry_run,
    )

    taxonomy_raw = load_taxonomy(taxonomy_path)
    taxonomy = compact_taxonomy(taxonomy_raw) if taxonomy_raw else {}
    if not taxonomy:
        logger.warning(
            "Warning: taxonomy file not found or empty. LLM will classify without taxonomy guidance.",
        )
    else:
        categories = taxonomy.get("categories", {})
        subcategory_count = sum(len(cat.get("subcategories") or {}) for cat in categories.values())
        logger.info("Loaded taxonomy: %d categories, %d subcategories", len(categories), subcategory_count)

    transactions = load_transactions(data_dir)
    if args.max_rows and args.max_rows > 0:
        transactions = transactions[: args.max_rows]
        logger.info("Applied max_rows: %d transaction(s) kept", len(transactions))

    enriched = enrich_transactions(transactions, taxonomy, batch_size=args.batch_size, dry_run=args.dry_run)
    write_output(enriched, output_path, args.format)

    logger.info("Enriched %d transaction(s) -> %s", len(enriched), output_path)
    logger.info("Completed in %.2fs", time.perf_counter() - start_time)


if __name__ == "__main__":
    main()
