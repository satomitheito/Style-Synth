"""
Pytest configuration for frontend tests.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest


@pytest.fixture
def sample_wardrobe_items():
    """Sample wardrobe items for testing."""
    return [
        {
            "id": 1,
            "category": "Tops",
            "subcategory": "T-shirt",
            "colors": ["Blue"],
            "occasions": ["Casual"],
            "season": ["Summer"],
            "brand": "Nike",
        },
        {
            "id": 2,
            "category": "Bottoms",
            "subcategory": "Jeans",
            "colors": ["Black"],
            "occasions": ["Formal", "Casual"],
            "season": ["All-Season"],
            "brand": "Levi's",
        },
        {
            "id": 3,
            "category": "Shoes",
            "subcategory": "Sneakers",
            "colors": ["White", "Red"],
            "occasions": ["Any Occasion"],
            "season": ["Spring", "Fall"],
            "brand": "Adidas",
        },
    ]
