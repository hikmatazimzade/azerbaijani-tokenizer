from typing import List, Generator, Optional, Tuple
from time import perf_counter
from pathlib import Path
import os

from lingua import (
    Language,
    LanguageDetectorBuilder
)
from nltk.tokenize import sent_tokenize
from datasets import load_dataset, get_dataset_config_names
from huggingface_hub import login
from decouple import AutoConfig
import nltk

from utils.logger import get_logger
from utils.config import ROOT_DIR

nltk.download('punkt_tab')
logger = get_logger("utils.clean")


LANGUAGES = [Language.ENGLISH, Language.AZERBAIJANI]
LANGUAGE_DETECTOR = LanguageDetectorBuilder.from_languages(*LANGUAGES).build()


AZ_CHARS = set('əıöüşğç')
AZ_COMMON_WORDS = set(['və', 'bu', 'ki', 'ilə', 'üçün','olan', 'ola',
                       'daha', 'çox', 'bir', 'da', 'də', 'isə', 'ona',
                       'onun', 'belə'])

ENGLISH_COMMON_WORDS = ['the', 'is', 'are', 'was', 'were', 'have',
                        'has', 'had', 'to']


config = AutoConfig()
HUGGINGFACE_ACCESS_TOKEN = config("HUGGINGFACE_ACCESS_TOKEN")
login(HUGGINGFACE_ACCESS_TOKEN)


def quick_filter(sentence: str) -> Optional[bool]:
    """Filter based on latin characters, azerbaijani characters
    common words and sentence length"""
    sentence_lower = sentence.lower()
    words = sentence_lower.split()

    if len(sentence_lower) < 50:
        return False

    if any(char in AZ_CHARS for char in sentence_lower):
        return True

    if any(word in AZ_COMMON_WORDS for word in words):
        return True

    if any(word in ENGLISH_COMMON_WORDS for word in words):
        return False
    
    return None


def split_into_sentences(text: str) -> List[str]:
    """Split given text into list of sentences"""
    sentences = sent_tokenize(text, language="turkish")
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences


def process_chunk(text: str) -> None:
    """Process and clean the given text for the tokenizer training"""
    azerbaijani_sentences: List[str] = []

    start_time = perf_counter()
    sentences = split_into_sentences(text)
    
    logger.info(f"{len(sentences)} sentences split in " 
                f"{perf_counter() - start_time}")
    
    for sentence in sentences:
        filter_res = quick_filter(sentence)

        if filter_res is not None:
            if filter_res:
                azerbaijani_sentences.append(sentence)
            
            continue

        language = LANGUAGE_DETECTOR.detect_language_of(sentence)

        if language and language.name == "AZERBAIJANI":
            azerbaijani_sentences.append(sentence)
    
    append_file(" ".join(azerbaijani_sentences))


def clean_dataset(file: open, row_number_per_chunk: int=1_000_000
                  ) -> Generator[None]:
    """Split file content into chunks based on given
    row number and pass each chunk for processing"""
    chunk_lines = []
    logger.info("Dataset cleaning started...")

    for line in file:
        chunk_lines.append(line)

        if len(chunk_lines) >= row_number_per_chunk:
            chunk_text = "".join(chunk_lines)
            yield process_chunk(chunk_text)
            chunk_lines = []

    if chunk_lines:
        chunk_text = "".join(chunk_lines)
        yield process_chunk(chunk_text)


def append_file(text: str,
            file_name: str=f"{ROOT_DIR}/data/azerbaijani_data.txt") -> None:
    """Append text content to the given file"""
    with open(file=file_name, mode="a", encoding="utf-8") as azerbaijani_file:
        azerbaijani_file.write(text)
        logger.info(f"Successfully written data to {file_name} file")


def prune_file(file_path: str=f"{ROOT_DIR}"
                    "/data/azerbaijani_data.txt") -> None:
    """Prune all the content of the given file"""
    with open(file_path, "w") as file:
        file.write("")
    logger.info(f"Pruned content of {file_path}")


def clean_file(file_name: str) -> None:
    """Open file and pass it for processing to clean"""
    with open(file_name, "r", encoding="utf-8") as file:
        for _ in clean_dataset(file):
            pass


def create_file(file_path: str) -> None:
    """Create file if it doesn't exist"""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if not file_path.exists():
        file_path.touch()


def install_configs(dataset_name: str, configs: List[str],
                    file_path: str) -> None:
    """Install configs text data from given dataset"""
    for config in configs:
        logger.info(f"  Downloading config: {config}")
        dataset = load_dataset(
            dataset_name,
            config,
            cache_dir=f"{ROOT_DIR}/data"
        )

        write_dataset_data(file_path, dataset)


def write_dataset_data(file_path: str, dataset: load_dataset) -> None:
    """Write dataset text data to given file path"""
    with open(file_path, "a", encoding="utf-8") as f:
        for item in dataset["train"]:
            f.write(item["text"] + "\n")


def install_dataset(dataset_name: str, file_path: str) -> None:
    """Install dataset to the given file path"""
    logger.info(f"Downloading {dataset_name}")
    configs = get_dataset_config_names(dataset_name)

    if configs:
        install_configs(dataset_name, configs, file_path)

    else:
        dataset = load_dataset(dataset_name,
                               cache_dir=f"{ROOT_DIR}/data")

        write_dataset_data(file_path, dataset)


def install_datasets(dataset_names: Tuple[str,...]) -> List[str]:
    """Install datasets to data folder from hugging face"""
    file_paths = []
    for dataset_name in dataset_names:
        new_dataset_name = dataset_name.replace("/", "_")
        file_path = os.path.join(ROOT_DIR, "data", f"{new_dataset_name}.txt")

        create_file(file_path)
        prune_file(file_path)

        try:
            install_dataset(dataset_name, file_path)

            logger.info(f"Successfully wrote text data to {file_path}")
            file_paths.append(file_path)

        except Exception as install_error:
            logger.error(f"Error occurred while installing "
                         f"{dataset_name}: {install_error}")

        return file_paths


def get_azerbaijani_dataset(dataset_names: Tuple[str,...]) -> None:
    """Get azerbaijani_data.txt file that contains cleaned text"""
    create_file(f"{ROOT_DIR}/data/azerbaijani_data.txt")
    prune_file()
    file_paths = install_datasets(dataset_names)

    for file_path in file_paths:
        start = perf_counter()
        clean_file(file_path)

        logger.info(f"Time took to clean the dataset: {perf_counter() - start}")


if __name__ == "__main__":
    dataset_names = ("LocalDoc/AzTC", "allmalab/DOLLMA")
    get_azerbaijani_dataset(dataset_names)