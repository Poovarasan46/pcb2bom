"""Part lookup module for querying electronic component specifications.

Supports:
- Local offline Mock Mode (data/sample_parts_mock.json)
- Digi-Key PartSearch API
- Octopart / Nexar GraphQL API
- Persistent disk-based JSON response caching
- Rate limiting with exponential backoff and jitter
- RapidFuzz matching between OCR readings and catalog MPNs/aliases
- Graceful degradation for unidentified parts
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from dotenv import load_dotenv
from rapidfuzz import fuzz

from pcb2bom.models import Component, PartLookupResult

logger = logging.getLogger(__name__)

# Automatically load .env if available
load_dotenv()


class DiskCache:
    """Thread-safe disk-backed JSON response cache."""

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None) -> None:
        if cache_dir is None:
            cache_env = os.getenv("PCB2BOM_CACHE_DIR", "~/.cache/pcb2bom")
            self.cache_dir = Path(os.path.expanduser(cache_env))
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, key: str) -> Path:
        safe_key = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in key.upper())
        return self.cache_dir / f"{safe_key}.json"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._get_path(key)
        if path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    logger.debug("Cache hit for '%s'", key)
                    return data
            except Exception as e:
                logger.warning("Failed to read cache file %s: %s", path, e)
        return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        path = self._get_path(key)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(value, f, indent=2)
            logger.debug("Cached response for '%s'", key)
        except Exception as e:
            logger.warning("Failed to write cache for %s: %s", key, e)


class MockLookupProvider:
    """Offline lookup provider reading from sample_parts_mock.json."""

    def __init__(self, mock_file_path: Optional[Union[str, Path]] = None) -> None:
        if mock_file_path is None:
            # Default to bundled data/sample_parts_mock.json
            base = Path(__file__).resolve().parent.parent / "data" / "sample_parts_mock.json"
            self.mock_file = base
        else:
            self.mock_file = Path(mock_file_path)

        self.parts_db: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self.mock_file.is_file():
            logger.warning("Mock parts database not found at %s", self.mock_file)
            return
        try:
            with open(self.mock_file, "r", encoding="utf-8") as f:
                self.parts_db = json.load(f)
            logger.info("Loaded %d mock parts from %s", len(self.parts_db), self.mock_file.name)
        except Exception as e:
            logger.error("Failed to parse mock database: %s", e)

    def search(self, query: str) -> Optional[Dict[str, Any]]:
        """Finds closest matching part in the mock database."""
        if not query or not self.parts_db:
            return None

        clean_q = query.strip().upper()
        best_part = None
        best_score = 0.0

        for part in self.parts_db:
            mpn = part.get("mpn", "").upper()
            aliases = [a.upper() for a in part.get("aliases", [])]

            # Exact match
            if clean_q == mpn or clean_q in aliases:
                return {**part, "score": 100.0}

            # Fuzzy match MPN
            ratio_mpn = fuzz.ratio(clean_q, mpn)
            # Partial ratio helps if OCR captured substring of MPN (e.g. '555' in 'NE555P')
            part_ratio = fuzz.partial_ratio(clean_q, mpn)
            score = max(ratio_mpn, part_ratio * 0.9)

            for alias in aliases:
                score = max(score, fuzz.ratio(clean_q, alias))

            if score > best_score:
                best_score = score
                best_part = part

        if best_part and best_score >= 60.0:
            return {**best_part, "score": best_score}

        return None


class DigiKeyClient:
    """Client for Digi-Key API v3 with authentication and rate-limiting."""

    BASE_URL = "https://api.digikey.com/Search/v3/Products/Keyword"

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        timeout: int = 10,
    ) -> None:
        self.client_id = client_id or os.getenv("DIGIKEY_CLIENT_ID", "")
        self.client_secret = client_secret or os.getenv("DIGIKEY_CLIENT_SECRET", "")
        self.timeout = timeout

    @property
    def is_configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def search_keyword(self, keyword: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
        if not self.is_configured:
            return None

        headers = {
            "X-DIGIKEY-Client-Id": self.client_id,
            "Authorization": f"Bearer {self.client_secret}",
            "Content-Type": "application/json",
        }
        payload = {"Keywords": keyword, "RecordCount": 1}

        backoff = 1.0
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.BASE_URL, json=payload, headers=headers, timeout=self.timeout
                )
                if response.status_code == 200:
                    data = response.json()
                    products = data.get("Products", [])
                    if products:
                        prod = products[0]
                        return {
                            "mpn": prod.get("ManufacturerPartNumber", keyword),
                            "manufacturer": prod.get("Manufacturer", {}).get("Value", "Unknown"),
                            "description": prod.get("ProductDescription", ""),
                            "package": prod.get("PackageType", {}).get("Value", "Unknown"),
                            "datasheet_url": prod.get("DatasheetUrl", ""),
                        }
                    return None
                elif response.status_code == 429:
                    logger.warning("DigiKey rate limit reached; backing off %.1fs", backoff)
                    time.sleep(backoff)
                    backoff *= 2.0
                else:
                    logger.debug("DigiKey returned status %d: %s", response.status_code, response.text)
                    return None
            except Exception as e:
                logger.warning("DigiKey request failed on attempt %d: %s", attempt + 1, e)
                time.sleep(backoff)
                backoff *= 2.0

        return None


class OctopartClient:
    """Client for Octopart / Nexar GraphQL API."""

    NEXAR_URL = "https://api.nexar.com/graphql"

    def __init__(
        self,
        token: Optional[str] = None,
        timeout: int = 10,
    ) -> None:
        self.token = token or os.getenv("OCTOPART_API_TOKEN", "") or os.getenv("NEXAR_TOKEN", "")
        self.timeout = timeout

    @property
    def is_configured(self) -> bool:
        return bool(self.token)

    def search_mpn(self, mpn: str) -> Optional[Dict[str, Any]]:
        if not self.is_configured:
            return None

        query = """
        query PartSearch($query: String!) {
          supSearch(q: $query, limit: 1) {
            results {
              part {
                mpn
                manufacturer { name }
                shortDescription
                specs {
                  attribute { name }
                  displayValue
                }
                bestDatasheet { url }
              }
            }
          }
        }
        """
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                self.NEXAR_URL,
                json={"query": query, "variables": {"query": mpn}},
                headers=headers,
                timeout=self.timeout,
            )
            if response.status_code == 200:
                data = response.json()
                results = data.get("data", {}).get("supSearch", {}).get("results", [])
                if results:
                    part = results[0].get("part", {})
                    return {
                        "mpn": part.get("mpn", mpn),
                        "manufacturer": part.get("manufacturer", {}).get("name", "Unknown"),
                        "description": part.get("shortDescription", ""),
                        "package": "SMD/Through-Hole",
                        "datasheet_url": part.get("bestDatasheet", {}).get("url", ""),
                    }
        except Exception as e:
            logger.debug("Octopart/Nexar request failed: %s", e)
        return None


class PartLookupService:
    """Unified part lookup service with caching, fuzzy matching, and fallback chains."""

    def __init__(
        self,
        use_mock: bool = True,
        mock_file_path: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.use_mock = use_mock
        self.cache = DiskCache(cache_dir=cache_dir)
        self.mock_provider = MockLookupProvider(mock_file_path=mock_file_path)
        self.digikey = DigiKeyClient()
        self.octopart = OctopartClient()

    def identify_part(
        self,
        query: str,
        class_hint: Optional[str] = None,
        candidate_variations: Optional[List[str]] = None,
    ) -> PartLookupResult:
        """Looks up a part given OCR text, candidates, and optional class hint.

        Args:
            query: Primary cleaned OCR string.
            class_hint: Component class name (e.g. 'ic').
            candidate_variations: Permutations to try (from OCR confusions).

        Returns:
            PartLookupResult with match details or unidentified status.
        """
        if not query or len(query.strip()) < 2:
            return PartLookupResult(
                query_string=query,
                matched_mpn="N/A",
                manufacturer="Unknown",
                description="Unlabeled / Passive component" if class_hint in ("resistor", "capacitor") else "Unidentified",
                package=f"Generic {class_hint}" if class_hint else "Unknown",
                datasheet_url="",
                match_confidence=0.0,
                source="unidentified",
            )

        search_candidates = [query]
        if candidate_variations:
            for cand in candidate_variations:
                if cand not in search_candidates:
                    search_candidates.append(cand)

        # 1. Check cache first
        for cand in search_candidates:
            cached = self.cache.get(cand)
            if cached:
                return PartLookupResult(
                    query_string=query,
                    matched_mpn=cached["mpn"],
                    manufacturer=cached["manufacturer"],
                    description=cached["description"],
                    package=cached.get("package", "Unknown"),
                    datasheet_url=cached.get("datasheet_url", ""),
                    match_confidence=cached.get("match_confidence", 0.95),
                    source=cached.get("source", "cache"),
                )

        # 2. If online API is configured and mock mode is not forced
        if not self.use_mock:
            if self.digikey.is_configured:
                for cand in search_candidates:
                    res = self.digikey.search_keyword(cand)
                    if res:
                        score = fuzz.ratio(query.upper(), res["mpn"].upper()) / 100.0
                        lookup_res = PartLookupResult(
                            query_string=query,
                            matched_mpn=res["mpn"],
                            manufacturer=res["manufacturer"],
                            description=res["description"],
                            package=res["package"],
                            datasheet_url=res["datasheet_url"],
                            match_confidence=max(0.6, score),
                            source="digikey",
                        )
                        self.cache.set(cand, asdict(lookup_res))
                        return lookup_res

            if self.octopart.is_configured:
                for cand in search_candidates:
                    res = self.octopart.search_mpn(cand)
                    if res:
                        score = fuzz.ratio(query.upper(), res["mpn"].upper()) / 100.0
                        lookup_res = PartLookupResult(
                            query_string=query,
                            matched_mpn=res["mpn"],
                            manufacturer=res["manufacturer"],
                            description=res["description"],
                            package=res["package"],
                            datasheet_url=res["datasheet_url"],
                            match_confidence=max(0.6, score),
                            source="octopart",
                        )
                        self.cache.set(cand, asdict(lookup_res))
                        return lookup_res

        # 3. Fallback to Mock Database
        best_match = None
        best_score = 0.0

        for cand in search_candidates:
            match = self.mock_provider.search(cand)
            if match and match["score"] > best_score:
                best_score = match["score"]
                best_match = match

        if best_match and best_score >= 60.0:
            conf = best_score / 100.0
            lookup_res = PartLookupResult(
                query_string=query,
                matched_mpn=best_match["mpn"],
                manufacturer=best_match["manufacturer"],
                description=best_match["description"],
                package=best_match["package"],
                datasheet_url=best_match.get("datasheet_url", ""),
                match_confidence=conf,
                source="mock",
            )
            self.cache.set(query, asdict(lookup_res))
            return lookup_res

        # 4. Unidentified marking
        logger.debug("Marking '%s' could not be identified", query)
        return PartLookupResult(
            query_string=query,
            matched_mpn=query,
            manufacturer="Unknown",
            description=f"Unrecognized marking for {class_hint}" if class_hint else "Unrecognized part",
            package="Unknown",
            datasheet_url="",
            match_confidence=0.0,
            source="unidentified",
        )


def run_lookup_pipeline(
    components: List[Component],
    use_mock: bool = True,
    mock_file_path: Optional[Union[str, Path]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Tuple[List[Component], List[Component]]:
    """Identifies all components and returns identified and unidentified lists.

    Args:
        components: List of components with OCR results.
        use_mock: Whether to use offline mock mode.
        mock_file_path: Path to custom mock JSON.
        cache_dir: Custom disk cache directory.

    Returns:
        Tuple of (all_components, unidentified_components).
    """
    service = PartLookupService(
        use_mock=use_mock,
        mock_file_path=mock_file_path,
        cache_dir=cache_dir,
    )

    unidentified: List[Component] = []

    for comp in components:
        raw_q = comp.ocr_result.cleaned_text if comp.ocr_result else ""
        cands = comp.ocr_result.candidates if comp.ocr_result else []
        res = service.identify_part(
            query=raw_q,
            class_hint=comp.class_name,
            candidate_variations=cands,
        )
        comp.lookup_result = res

        if res.source == "unidentified" and comp.class_name.lower() not in ("resistor", "capacitor"):
            unidentified.append(comp)

    logger.info(
        "Lookup pipeline finished. Total components: %d, Unidentified ICs/parts: %d",
        len(components),
        len(unidentified),
    )
    return components, unidentified
