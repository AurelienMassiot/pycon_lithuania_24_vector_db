from pathlib import Path

# Directories
ROOT_DIR: Path = Path(__file__).parent.parent.resolve()

# Files
IMAGES_PATH: str = str(ROOT_DIR / './data/images')
GC_LOGO_PATH: str = str(ROOT_DIR / './data/assets/gc_logo.webp')
#LOGO_PATH: str = str(ROOT_DIR / './data/assets/pycon_lt_logo.png')
# LOGO_PATH: str = str(ROOT_DIR / './data/assets/logo_nai.png')
LOGO_PATH: str = str(ROOT_DIR / './data/assets/data_days_logo.png')
DICT_IDS_EMBEDDINGS_PATH: str = str(ROOT_DIR / './data/dict_ids_embeddings.pickle')
DICT_IDS_EMBEDDINGS_FULL_PATH: str = str(ROOT_DIR / './data/dict_ids_embeddings_full.pickle')