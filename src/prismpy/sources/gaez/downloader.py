"""
GAEZ (Global Agro-Ecological Zones) data downloader.

Downloads GAEZ v4 data from FAO S3 bucket for crop suitability
and potential yield estimates used by ACEA.

Reference: FAO GAEZ Data Portal - https://gaez.fao.org/
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# GAEZ cultivar name to 4-letter code mapping
GAEZ_CROP_CODES = {
    # Maize cultivars
    'Highland maize': 'hmze',
    'Lowland maize': 'lmze',
    'Temperate maize': 'tmze',
    'Maize': 'maiz',
    'Highland_maize': 'hmze',
    'Lowland_maize': 'lmze',
    'Temperate_maize': 'tmze',
    # Wheat cultivars
    'Spring wheat': 'swhe',
    'Winter wheat': 'wwhe',
    'Spring_wheat': 'swhe',
    'Winter_wheat': 'wwhe',
    # Rice cultivars
    'Wetland rice': 'wric',
    'Dryland rice': 'dric',
    'Wetland_rice': 'wric',
    'Dryland_rice': 'dric',
    # Sorghum cultivars
    'Highland sorghum': 'hsor',
    'Lowland sorghum': 'lsor',
    'Highland_sorghum': 'hsor',
    'Lowland_sorghum': 'lsor',
    # Millet
    'Pearl millet': 'pmlt',
    'Foxtail millet': 'fmlt',
    'Pearl_millet': 'pmlt',
    'Foxtail_millet': 'fmlt',
}


# Crop name to GAEZ cultivar mapping
GAEZ_CULTIVAR_MAP = {
    'Maize': ['Highland_maize', 'Lowland_maize', 'Temperate_maize', 'Maize'],
    'Corn': ['Highland_maize', 'Lowland_maize', 'Temperate_maize', 'Maize'],
    'Wheat': ['Spring_wheat', 'Winter_wheat'],
    'Rice': ['Wetland_rice', 'Dryland_rice'],
    'Sorghum': ['Highland_sorghum', 'Lowland_sorghum'],
    'Millet': ['Pearl_millet', 'Foxtail_millet'],
}


# GAEZ variable codes
GAEZ_VARIABLES = {
    'LUT': 'idx',           # Suitability class
    'PotentialYield': 'yld',  # Potential yield (kg/ha)
    'F3': 'fc3',            # Constraint factor
}


# FAO S3 base URL
BASE_URL = "https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/res02/CRUTS32/Hist"


# Input levels
INPUT_LEVELS = {
    'High': '8110H',
    'Intermediate': '8110I',
    'Low': '8110L',
}


class GAEZDownloader:
    """Downloads and manages GAEZ crop suitability data.

    GAEZ provides global crop suitability and potential yield estimates
    at various input levels (management intensity).

    Attributes:
        cache_dir: Directory to cache downloaded files
    """

    DEFAULT_CACHE_DIR = Path.home() / ".prismpy" / "cache" / "gaez"

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the GAEZ downloader.

        Args:
            cache_dir: Directory for caching downloads
        """
        self.cache_dir = Path(cache_dir) if cache_dir else self.DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cultivars_for_crop(self, crop_name: str) -> List[str]:
        """Get GAEZ cultivar names for a crop.

        Args:
            crop_name: Crop name (e.g., 'Wheat', 'Maize')

        Returns:
            List of cultivar names
        """
        return GAEZ_CULTIVAR_MAP.get(crop_name, [crop_name])

    def download_cultivar(
        self,
        cultivar: str,
        water_supply: str = 'irr',
        input_levels: Optional[List[str]] = None,
        overwrite: bool = False,
    ) -> Dict[str, Path]:
        """Download GAEZ data for a cultivar.

        Args:
            cultivar: Cultivar name (e.g., 'Spring_wheat')
            water_supply: 'irr' (irrigated) or 'rf' (rainfed)
            input_levels: List of input levels to download
            overwrite: Whether to re-download existing files

        Returns:
            Dictionary mapping variable name to downloaded file path
        """
        try:
            import requests
        except ImportError:
            logger.error("requests library required for GAEZ download")
            return {}

        if input_levels is None:
            input_levels = ['High', 'Low']

        crop_code = GAEZ_CROP_CODES.get(cultivar)
        if not crop_code:
            logger.warning(f"No GAEZ code for cultivar: {cultivar}")
            return {}

        water_code = 'a' if water_supply == 'irr' else 'b'
        downloaded = {}

        for input_level in input_levels:
            input_path = INPUT_LEVELS.get(input_level)
            if not input_path:
                continue

            for acea_var, gaez_var in GAEZ_VARIABLES.items():
                # Source URL
                gaez_filename = f"{crop_code}200{water_code}_{gaez_var}.tif"
                url = f"{BASE_URL}/{input_path}/{gaez_filename}"

                # Target filename (ACEA naming convention)
                cultivar_fname = cultivar.replace(' ', '_')
                acea_filename = f"{acea_var}_{cultivar_fname}_{water_supply}_{input_level}_1981_2010.tif"

                # Cache path
                cache_path = self.cache_dir / acea_var / acea_filename

                if cache_path.exists() and not overwrite:
                    logger.debug(f"Using cached: {cache_path}")
                    downloaded[f"{acea_var}_{cultivar}_{water_supply}_{input_level}"] = cache_path
                    continue

                # Download
                cache_path.parent.mkdir(parents=True, exist_ok=True)

                try:
                    logger.info(f"Downloading {acea_filename}...")
                    response = requests.get(url, timeout=120, stream=True)
                    response.raise_for_status()

                    with open(cache_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    downloaded[f"{acea_var}_{cultivar}_{water_supply}_{input_level}"] = cache_path
                    logger.info(f"Downloaded: {cache_path}")

                except Exception as e:
                    logger.error(f"Failed to download {url}: {e}")

        return downloaded

    def download_for_crop(
        self,
        crop_name: str,
        water_supply: str = 'irr',
        input_levels: Optional[List[str]] = None,
    ) -> Dict[str, Path]:
        """Download GAEZ data for all cultivars of a crop.

        Args:
            crop_name: Crop name (e.g., 'Wheat')
            water_supply: 'irr' or 'rf'
            input_levels: Input levels to download

        Returns:
            Dictionary mapping file keys to paths
        """
        all_downloaded = {}
        cultivars = self.get_cultivars_for_crop(crop_name)

        for cultivar in cultivars:
            downloaded = self.download_cultivar(
                cultivar,
                water_supply=water_supply,
                input_levels=input_levels,
            )
            all_downloaded.update(downloaded)

        return all_downloaded

    def copy_to_output(
        self,
        crop_name: str,
        output_dir: Path,
        water_supplies: Optional[List[str]] = None,
        input_levels: Optional[List[str]] = None,
    ) -> List[Path]:
        """Download GAEZ data and copy to output directory.

        Args:
            crop_name: Crop name
            output_dir: Output directory (gaez/)
            water_supplies: List of water supply types
            input_levels: List of input levels

        Returns:
            List of output file paths
        """
        import shutil

        if water_supplies is None:
            water_supplies = ['irr', 'rf']

        if input_levels is None:
            input_levels = ['High', 'Low']

        output_files = []

        for ws in water_supplies:
            downloaded = self.download_for_crop(
                crop_name,
                water_supply=ws,
                input_levels=input_levels,
            )

            for key, src_path in downloaded.items():
                if not src_path.exists():
                    continue

                # Determine subdirectory (LUT, PotentialYield, F3)
                for var in GAEZ_VARIABLES.keys():
                    if key.startswith(var):
                        subdir = var
                        break
                else:
                    subdir = "other"

                dst_dir = output_dir / subdir
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst_path = dst_dir / src_path.name

                shutil.copy2(src_path, dst_path)
                output_files.append(dst_path)
                logger.debug(f"Copied {src_path} to {dst_path}")

        return output_files

    def check_cached(self, crop_name: str) -> Dict[str, bool]:
        """Check which GAEZ files are already cached.

        Args:
            crop_name: Crop name

        Returns:
            Dictionary mapping file keys to existence status
        """
        cached = {}
        cultivars = self.get_cultivars_for_crop(crop_name)

        for cultivar in cultivars:
            cultivar_fname = cultivar.replace(' ', '_')

            for ws in ['irr', 'rf']:
                for level in ['High', 'Low']:
                    for var in GAEZ_VARIABLES.keys():
                        filename = f"{var}_{cultivar_fname}_{ws}_{level}_1981_2010.tif"
                        cache_path = self.cache_dir / var / filename
                        key = f"{var}_{cultivar}_{ws}_{level}"
                        cached[key] = cache_path.exists()

        return cached
